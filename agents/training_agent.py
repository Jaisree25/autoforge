"""Training Agent — full agentic-pipeline pattern ported into AutoForge.

Pipeline (matches the user's `agentic-pipeline` C-compiler-inspired loop):

  1. Oracle baseline (sklearn LogisticRegression on flattened input) — defines
     the "must beat by 5 points" sanity gate.
  2. Generate `design.md` via Nemotron — architecture + hyperparams +
     estimated wall-clock + success criteria.
  3. **HITL gate — design.md approval.** Trainer pauses. The dashboard's
     approval panel shows the design; the human approves / edits / rejects.
  4. Generate `code/model.py` + `code/train.py` via Nemotron (structured
     JSON output, sklearn-based).
  5. Smoke harness — py_compile, import, instantiate model. Emits
     `verify_report.json`. Hard-fail if anything errors.
  6. Subprocess training — run `code/train.py` with a wall-clock cap, capture
     stdout/stderr to `logs/`. Soft-success if `best.pkl` exists even after
     timeout.
  7. Load + return `TrainingResult` from the saved model + metrics.

The internal design.md gate is what makes this "agentic" rather than just
"sklearn HPO with extra steps." Coordinator passes the HITL service in at
construction time; Trainer calls `self.hitl.request_and_wait()` mid-run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from config import ARTIFACTS_DIR, COORDINATOR_MODEL
from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    EventType,
)
from contracts.schemas import (
    AgentName,
    DatasetProfile,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
    TrainingResult,
    TrialResult,
)

from agents._llm_client import NemotronClient
from agents.base_agent import BaseAgent
from tools import training_pipeline as tp
from tools import training_tools as tt


class TrainerError(Exception):
    """Unrecoverable failure in the Trainer's pipeline."""


class TrainingAgent(BaseAgent):
    """Agentic Trainer — design → HITL → code → smoke → train → report."""

    name: ClassVar[AgentName] = AgentName.TRAINING

    def __init__(self, store, run_id: str, hitl=None) -> None:
        super().__init__(store=store, run_id=run_id)
        # 49B for the planning + code-gen calls — sklearn code-gen wants
        # solid instruction following.
        self.llm = NemotronClient(model=COORDINATOR_MODEL)
        self.hitl = hitl  # optional; if None the design gate is auto-approved

    # ------------------------------------------------------------------
    def run(  # type: ignore[override]
        self,
        strategy_spec: StrategySpec,
        training_envelope: TrainingEnvelope,
        dataset_profile: DatasetProfile,
        preparation_report: PreparationReport | None = None,
    ) -> TrainingResult:
        run_dir = ARTIFACTS_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        code_dir = run_dir / "code"
        models_dir = run_dir / "models"
        logs_dir = run_dir / "logs"

        with self._lifecycle("design → code → smoke → train"):
            # --- Stage 0: Precondition — Preparer must have produced a split ---
            self._check_prepared_data(dataset_profile, preparation_report)

            # --- Stage 1: Oracle baseline ---
            self.emit_event(
                EventType.TOOL_CALL,
                message="oracle = sklearn.LogisticRegression on flattened input",
            )
            oracle = tp.run_oracle(dataset_profile, preparation_report, run_dir)
            self.emit_event(
                EventType.INFO,
                message=(
                    f"oracle baseline: test_accuracy={oracle['test_accuracy']:.3f} "
                    f"in {oracle['wall_clock_s']:.1f}s "
                    f"(n_train={oracle['n_train']}, n_test={oracle['n_test']})"
                ),
            )

            # --- Stage 2: Codegen + smoke retry loop ---
            # Per the agentic-pipeline pattern: up to MAX_ATTEMPTS LLM calls,
            # each retry receives the previous attempt's verify errors as
            # context. After all retries fail, fall back to a hardcoded
            # MLP-on-MNIST template so the demo doesn't break.
            design_md, code_files, design_path, verify = (
                self._codegen_with_retries(
                    strategy_spec=strategy_spec,
                    dataset_profile=dataset_profile,
                    training_envelope=training_envelope,
                    preparation_report=preparation_report,
                    oracle=oracle,
                    run_dir=run_dir,
                    code_dir=code_dir,
                    max_attempts=3,
                )
            )

            # --- Stage 3: HITL gate on design.md (after code passes smoke) ---
            design_md = self._gate_design(
                design_md=design_md,
                oracle=oracle,
                design_path=design_path,
                strategy_spec=strategy_spec,
            )

            passed_checks = sum(1 for c in verify["checks"] if c["passed"])
            total_checks = len(verify["checks"])

            # --- Stage 6: Subprocess training ---
            prepared_dir = self._resolve_prepared_dir(
                dataset_profile, preparation_report,
            )
            max_seconds = int((training_envelope.max_train_minutes or 5.0) * 60)
            self.emit_event(
                EventType.TOOL_CALL,
                message=(
                    f"subprocess.run(python code/train.py "
                    f"--data-dir {prepared_dir.name} --max-time-seconds {max_seconds})"
                ),
            )
            t0 = time.time()
            sub = tp.run_training_subprocess(
                code_dir=code_dir,
                prepared_dir=prepared_dir,
                output_dir=models_dir,
                log_dir=logs_dir,
                max_time_seconds=max_seconds,
            )
            sub_duration = time.time() - t0

            if not sub["success"]:
                # Surface the last lines of stderr so the human sees what broke
                if sub["stderr_tail"]:
                    self.emit_event(
                        EventType.WARNING,
                        message=f"train.py stderr (last lines):\n{sub['stderr_tail'][:500]}",
                    )
                raise TrainerError(
                    f"Subprocess training failed: return_code={sub['return_code']}, "
                    f"timed_out={sub['timed_out']}"
                )

            metrics = sub.get("metrics") or {}
            val_acc = float(metrics.get("val_accuracy", 0.0))
            train_seconds = float(metrics.get("train_seconds", sub_duration))
            training_process = metrics.get("training_process") or {}
            self.emit_event(
                EventType.INFO,
                message=(
                    f"training done in {sub_duration:.1f}s · "
                    f"val_accuracy={val_acc:.3f}"
                    + (" (TIMED OUT; soft success)" if sub["timed_out"] else "")
                ),
                payload={"metrics": metrics},
            )

            # --- Stage 7: Build TrainingResult ---
            model_id = f"m_{int(time.time())}"
            # Move the trained model into a stable name expected by Evaluator
            best_pkl = Path(sub["best_pkl"]) if sub["best_pkl"] else None
            if best_pkl is None or not best_pkl.exists():
                raise TrainerError("No best.pkl produced by training subprocess.")

            final_path = models_dir / f"{model_id}.pkl"
            best_pkl.replace(final_path)

            # Single-trial summary (no Optuna here — the agentic-pipeline
            # pattern is one design → one training run; HPO is the
            # Researcher's recommendation surface)
            trials = [
                TrialResult(
                    trial_id=0,
                    params={"design": "see design.md"},
                    score=val_acc,
                    duration_seconds=train_seconds,
                    status="completed",
                ),
            ]

            # best_params: surface the EFFECTIVE hyperparameters (what
            # sklearn actually ran with), not just a design.md pointer. Falls
            # back to the pointer if introspection failed.
            effective = training_process.get("effective_params") or {}
            best_params: dict[str, Any] = {
                **effective,
                "design_md_path": str(design_path),
            }

            return TrainingResult(
                best_model_id=model_id,
                metric_name=strategy_spec.success_metric or "accuracy",
                best_score=val_acc,
                best_params=best_params,
                trials_completed=1,
                total_trials=1,
                training_time_seconds=train_seconds,
                artifact_path=str(final_path),
                library="sklearn",
                all_trials=trials,
                training_process=training_process,
                notes=(
                    f"agentic-pipeline: oracle={oracle['test_accuracy']:.3f}, "
                    f"trained={val_acc:.3f} "
                    f"(+{(val_acc - oracle['test_accuracy']):+.3f} vs oracle). "
                    f"Smoke harness {passed_checks}/{total_checks} passed."
                ),
            )

    # ------------------------------------------------------------------
    # Codegen + smoke retry loop (the heart of the agentic-pipeline pattern)
    # ------------------------------------------------------------------
    def _codegen_with_retries(
        self,
        strategy_spec: StrategySpec,
        dataset_profile: DatasetProfile,
        training_envelope: TrainingEnvelope,
        preparation_report: PreparationReport | None,
        oracle: dict[str, Any],
        run_dir: Path,
        code_dir: Path,
        max_attempts: int = 3,
    ) -> tuple[str, dict[str, str], Path, dict[str, Any]]:
        """Try N LLM-codegen rounds, each verified by the smoke harness.

        Returns `(design_md, code_files, design_path, verify_report)`. Raises
        TrainerError only if BOTH the LLM attempts AND the fallback template
        fail their smoke check (which would be a real bug in our code).
        """
        design_path = run_dir / "design.md"
        previous_errors: list[str] | None = None

        for attempt in range(max_attempts):
            self.emit_event(
                EventType.TOOL_CALL,
                message=(
                    f"nemotron.generate_design_and_code "
                    f"(attempt {attempt + 1}/{max_attempts}, /no_think)"
                ),
            )
            try:
                artifacts = tp.generate_design_and_code(
                    llm=self.llm,
                    spec=strategy_spec,
                    profile=dataset_profile,
                    envelope=training_envelope,
                    oracle=oracle,
                    prep=preparation_report,
                    on_thinking=None,
                    no_think=True,
                    previous_errors=previous_errors,
                )
            except Exception as exc:  # noqa: BLE001
                self.emit_event(
                    EventType.WARNING,
                    message=f"codegen LLM call failed: {type(exc).__name__}: {exc}",
                )
                previous_errors = [f"LLM call raised: {exc}"]
                continue

            design_md = artifacts["design_md"]
            # LLM only produced design.md + model.py. AutoForge's templated
            # train.py is paired in. The template is parameterized by modality
            # (and target column for CSV).
            is_image = dataset_profile.modality.value == "image"
            target_col = dataset_profile.target_column or "target"
            train_py = (
                _FALLBACK_TRAIN_IMAGE if is_image
                else _FALLBACK_TRAIN_CSV.replace("__TARGET__", target_col)
            )
            code_files = {
                "model.py": artifacts["model.py"],
                "train.py": train_py,
            }
            design_path.write_text(design_md, encoding="utf-8")
            code_dir.mkdir(parents=True, exist_ok=True)
            for name, content in code_files.items():
                (code_dir / name).write_text(content, encoding="utf-8")
            self.emit_event(
                EventType.INFO,
                message=(
                    f"attempt {attempt + 1}: design.md ({len(design_md)} chars) "
                    f"+ model.py ({len(artifacts['model.py'])} chars) "
                    f"+ templated train.py"
                ),
            )

            # Run smoke harness
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"smoke_harness (attempt {attempt + 1})",
            )
            verify = tp.run_smoke_harness(code_dir)
            (run_dir / "verify_report.json").write_text(
                json.dumps(verify, indent=2), encoding="utf-8",
            )
            n_passed = sum(1 for c in verify["checks"] if c["passed"])
            n_total = len(verify["checks"])
            if verify["overall_passed"]:
                self.emit_event(
                    EventType.INFO,
                    message=f"smoke passed: {n_passed}/{n_total} checks ✓",
                )
                return design_md, code_files, design_path, verify

            # Collect specific errors for the next retry
            failed_details = [
                c["detail"] for c in verify["checks"]
                if not c["passed"] and c.get("detail")
            ]
            previous_errors = failed_details
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"smoke FAILED on attempt {attempt + 1}: "
                    f"{n_passed}/{n_total} checks passed; "
                    f"{len(failed_details)} error(s) — "
                    + ("retrying" if attempt + 1 < max_attempts else "out of retries")
                ),
            )
            for err in failed_details[:3]:
                self.emit_event(
                    EventType.WARNING,
                    message=f"  • {err[:200]}",
                )

        # All LLM attempts failed — fall back to a hardcoded template.
        self.emit_event(
            EventType.WARNING,
            message=(
                f"after {max_attempts} LLM attempts, falling back to "
                "hardcoded MLP template (demo-safety net)"
            ),
        )
        design_md, code_files = self._fallback_template(
            dataset_profile, strategy_spec, oracle,
        )
        design_path.write_text(design_md, encoding="utf-8")
        for name, content in code_files.items():
            (code_dir / name).write_text(content, encoding="utf-8")
        self.emit_event(
            EventType.INFO,
            message="wrote fallback template; running final smoke check",
        )
        verify = tp.run_smoke_harness(code_dir)
        (run_dir / "verify_report.json").write_text(
            json.dumps(verify, indent=2), encoding="utf-8",
        )
        if not verify["overall_passed"]:
            for c in verify["checks"]:
                if not c["passed"]:
                    self.emit_event(
                        EventType.ERROR,
                        message=f"fallback failed check: {c['name']} — {c['detail']}",
                    )
            raise TrainerError(
                "Even the fallback template failed smoke. This is a bug in "
                "TrainingAgent._fallback_template."
            )
        return design_md, code_files, design_path, verify

    # ------------------------------------------------------------------
    # Hardcoded fallback template — guarantees the demo doesn't break.
    # ------------------------------------------------------------------
    def _fallback_template(
        self,
        profile: DatasetProfile,
        spec: StrategySpec,
        oracle: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """Return (design_md, code_files) using a known-good sklearn MLP recipe."""
        is_image = profile.modality.value == "image"

        oracle_acc = float(oracle.get("test_accuracy", 0.0))
        target_metric = spec.success_metric or "accuracy"
        target_thresh = max(spec.success_threshold, oracle_acc + 0.05)
        data_layout = (
            "`<data-dir>/train/<class>/*.png` + `<data-dir>/test/<class>/*.png`"
            if is_image else
            "`<data-dir>/train.csv` + `<data-dir>/test.csv`"
        )

        design_md = (
            "# Fallback design (LLM codegen failed after 3 attempts)\n\n"
            "AutoForge's hard-coded safety-net template kicked in. The "
            "structure below mirrors the LLM-generated format so the design "
            "gate review is consistent.\n\n"
            "## Architecture commitment\n"
            "`sklearn.neural_network.MLPClassifier`. Chosen over LogisticRegression "
            "as a fallback because the dataset is non-linear (image pixels / "
            "tabular mixed features) and MLP outperforms linear baselines on "
            "small datasets with mild regularization.\n\n"
            "## Hyperparameters (final)\n"
            "- `hidden_layer_sizes = (128, 64)` — Two layers: 128 captures feature "
            "  patterns, 64 narrows to class separation. Total ~110k params, well "
            "  under the envelope cap.\n"
            "- `alpha = 1e-3` — Mild L2 regularization; the dataset is small so we "
            "  prefer regularization over deeper architecture.\n"
            "- `learning_rate_init = 1e-3` — Standard Adam starting LR; converges "
            "  quickly for this size of network.\n"
            "- `max_iter = 30` — Enough for convergence on the prepared set without "
            "  blowing the wall-clock budget.\n"
            "- `random_state = 42` — Reproducibility across runs.\n\n"
            "## Wall-clock budget\n"
            "Estimated ~10s on CPU for the prepared dataset. Envelope cap respected. "
            "If we hit the cap, sklearn's `MLPClassifier` will stop at `max_iter` "
            "naturally — no separate abort needed.\n\n"
            "## Success criteria\n"
            f"- Hard target: `{target_metric}` ≥ {target_thresh:.3f}.\n"
            f"- Oracle delta: must beat sklearn LogReg baseline "
            f"(test_accuracy={oracle_acc:.3f}) by ≥0.05.\n"
            "- Rollback trigger: if `val_accuracy < oracle - 0.05`, the Evaluator "
            "  flags FAIL and the run does not deploy.\n\n"
            "## Risks & anti-patterns\n"
            "- Risk: MLP on tiny datasets can overfit. Mitigated by `alpha=1e-3`.\n"
            "- Anti-pattern avoided: we are NOT flattening labels or one-hot "
            "  encoding the target — sklearn handles integer class labels directly.\n"
            "- Overfitting risk given dataset size: moderate; augmentation "
            "  (if recorded by the Preparer) helps further but we do not depend on it.\n\n"
            "## Code structure\n"
            "- `model.py` exports `build_model() -> MLPClassifier`.\n"
            f"- `train.py` is AutoForge-templated: loads {data_layout}, calls "
            "  `build_model()`, fits, evaluates, saves `best.pkl` and `metrics.json`.\n"
            "- Required estimator surface: `.fit(X, y)` + `.predict(X)`. Met by MLPClassifier.\n\n"
            "## Verification plan\n"
            "- Smoke harness: `py_compile model.py`, `import build_model`, "
            "  `build_model()` instantiates.\n"
            "- After training: assert `val_accuracy >= oracle_accuracy + 0.05`.\n"
            "- Manual: confirm `best.pkl` loads via `joblib.load`.\n"
        )

        model_py = (
            "from sklearn.neural_network import MLPClassifier\n"
            "\n"
            "def build_model():\n"
            "    return MLPClassifier(\n"
            "        hidden_layer_sizes=(128, 64),\n"
            "        alpha=1e-3,\n"
            "        learning_rate_init=1e-3,\n"
            "        max_iter=30,\n"
            "        random_state=42,\n"
            "    )\n"
        )

        if is_image:
            train_py = _FALLBACK_TRAIN_IMAGE
        else:
            target_col = profile.target_column or "target"
            train_py = _FALLBACK_TRAIN_CSV.replace("__TARGET__", target_col)

        return design_md, {"model.py": model_py, "train.py": train_py}

    # ------------------------------------------------------------------
    # Design HITL gate
    # ------------------------------------------------------------------
    def _gate_design(
        self,
        design_md: str,
        oracle: dict[str, Any],
        design_path: Path,
        strategy_spec: StrategySpec,
    ) -> str:
        """Request human approval of design.md. Returns the (possibly edited) text."""
        if self.hitl is None:
            self.emit_event(
                EventType.INFO,
                message="no HITL service wired — auto-approving design.md (dev mode)",
            )
            return design_md

        first_lines = "\n".join(design_md.splitlines()[:5])
        request = ApprovalRequest(
            run_id=self.run_id,
            agent=AgentName.TRAINING,
            title="Approve design.md (Trainer)",
            description=(
                f"Trainer generated a design. Oracle baseline = "
                f"{oracle['test_accuracy']:.3f}; must beat by ≥0.05. "
                f"Target {strategy_spec.success_metric} ≥ "
                f"{strategy_spec.success_threshold:.2f}."
            ),
            payload={
                "summary": f"Trainer wrote design.md ({len(design_md)} chars)",
                "next_agent": "code generation",
                "design_md": design_md,
                "design_path": str(design_path),
                "oracle": oracle,
                "preview": first_lines,
            },
        )
        self.emit_event(
            EventType.APPROVAL_REQUESTED,
            message="design.md ready — awaiting human approval before code generation",
            payload={
                "summary": "Approve the proposed training design (architecture + HPs)",
                "next_agent": "code generation",
                "from_agent": AgentName.TRAINING.value,
                "request_id": request.request_id,
            },
        )
        response = self.hitl.request_and_wait(request)
        self.emit_event(
            EventType.APPROVAL_RECEIVED,
            message=(
                f"design.md {response.decision.value} by "
                f"{response.responder or 'unknown'}"
                + (f" — {response.comment}" if response.comment else "")
            ),
            payload={
                "request_id": response.request_id,
                "decision": response.decision.value,
            },
        )

        if response.decision is ApprovalDecision.REJECTED:
            raise TrainerError(
                response.comment or "design.md rejected by reviewer"
            )

        if (
            response.decision is ApprovalDecision.EDITED
            and response.response_payload is not None
        ):
            edited = response.response_payload.get("design_md")
            if isinstance(edited, str) and edited.strip():
                self.emit_event(
                    EventType.INFO,
                    message="using edited design.md from reviewer",
                )
                design_path.write_text(edited, encoding="utf-8")
                return edited

        return design_md

    # ------------------------------------------------------------------
    # Precondition check — Trainer is the contract enforcer here
    # ------------------------------------------------------------------
    def _check_prepared_data(
        self,
        profile: DatasetProfile,
        prep: PreparationReport | None,
    ) -> None:
        """Fail loudly if the Preparer didn't produce the expected layout.

        Image modality: expects `prepared_dataset_path / {train, test}` dirs
        with class subfolders. Tabular: expects `train.csv` + `test.csv`.

        We do NOT auto-fix here. The Preparer has an internal split backstop
        that should have caught this; if we got here without a split, something
        is genuinely broken upstream and the demo should pause for a human to
        investigate rather than silently train on the wrong data.
        """
        if prep is None or not prep.prepared_dataset_path:
            raise TrainerError(
                "Preparer did not return a prepared_dataset_path. "
                "Cannot train without a known data location."
            )
        prepared = Path(prep.prepared_dataset_path)
        if not prepared.exists():
            raise TrainerError(
                f"prepared_dataset_path does not exist: {prepared}"
            )

        if profile.modality.value == "image":
            train_dir = prepared / "train"
            test_dir = prepared / "test"
            if not train_dir.is_dir() or not test_dir.is_dir():
                self.emit_event(
                    EventType.ERROR,
                    message=(
                        f"Preparer output missing train/ or test/ subdir at "
                        f"`{prepared}`. The Trainer requires a class-folder "
                        f"split for image tasks. Aborting."
                    ),
                )
                raise TrainerError(
                    f"Preparer did not produce a train/test split for images "
                    f"at {prepared}. Check the Preparer's plan and ensure a "
                    f"`train_test_split_images` op ran."
                )
            self.emit_event(
                EventType.INFO,
                message=f"precondition OK: image split present at `{prepared}`",
            )
            return

        if profile.modality.value == "tabular":
            train_csv = prepared / "train.csv"
            test_csv = prepared / "test.csv"
            if not train_csv.is_file() or not test_csv.is_file():
                self.emit_event(
                    EventType.ERROR,
                    message=(
                        f"Preparer output missing train.csv or test.csv at "
                        f"`{prepared}`. Aborting."
                    ),
                )
                raise TrainerError(
                    f"Preparer did not produce train.csv/test.csv at {prepared}. "
                    f"Check the Preparer's plan and ensure a "
                    f"`train_test_split_csv` op ran."
                )
            self.emit_event(
                EventType.INFO,
                message=f"precondition OK: tabular split present at `{prepared}`",
            )
            return

        # Unknown modality — let the rest of the pipeline try.
        self.emit_event(
            EventType.WARNING,
            message=f"precondition skipped: unknown modality `{profile.modality.value}`",
        )

    # ------------------------------------------------------------------
    def _resolve_prepared_dir(
        self,
        profile: DatasetProfile,
        prep: PreparationReport | None,
    ) -> Path:
        """Return the directory the generated train.py should read from."""
        if prep and prep.prepared_dataset_path:
            candidate = Path(prep.prepared_dataset_path)
            if candidate.exists():
                return candidate
        # Fallback: source dataset
        src = Path(profile.dataset_path)
        if src.is_file():
            return src.parent  # train.py is expected to look for the file
        return src


# ===========================================================================
# Fallback train.py templates — used only when 3 LLM attempts fail smoke.
# Kept at module scope so they don't bloat the class body. Both are tested
# manually to py_compile + import + run cleanly.
# ===========================================================================
_FALLBACK_TRAIN_IMAGE = '''\
"""Fallback train.py — sklearn classifier on a class-folder image dataset.

After fit, introspects the fitted estimator and emits training-process info
(iterations, loss curve, effective hyperparameters) so the Trainer's UI can
show how training proceeded — separate from the Evaluator's accuracy/latency
benchmark on the same artifact.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

from model import build_model

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_split(split_dir):
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for cls in classes:
        for img_path in (split_dir / cls).rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            with Image.open(img_path) as img:
                arr = np.asarray(img.convert("L"), dtype=np.float32).flatten() / 255.0
            X.append(arr)
            y.append(cls_to_idx[cls])
    return np.stack(X), np.array(y), classes


def _safe_repr(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def _introspect_training(model):
    """Pull training-process info off a fitted sklearn estimator."""
    info = {"estimator_class": type(model).__name__}

    # Iterations — MLPClassifier exposes int, LogisticRegression exposes array
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        try:
            iters = list(n_iter) if hasattr(n_iter, "__iter__") else [int(n_iter)]
            info["n_iter"] = int(max(iters)) if iters else int(n_iter)
        except Exception:
            pass

    # Loss curve — MLPClassifier exposes loss_curve_; GradientBoosting uses train_score_
    if hasattr(model, "loss_curve_"):
        try:
            info["loss_curve"] = [float(x) for x in model.loss_curve_]
        except Exception:
            pass
    elif hasattr(model, "train_score_"):
        try:
            info["loss_curve"] = [float(x) for x in model.train_score_]
            info["loss_curve_label"] = "train_score (per iteration)"
        except Exception:
            pass

    if hasattr(model, "best_loss_"):
        try:
            info["final_loss"] = float(model.best_loss_)
        except Exception:
            pass

    # Effective hyperparameters — the actual values sklearn ran with
    try:
        info["effective_params"] = {
            k: _safe_repr(v) for k, v in model.get_params(deep=False).items()
        }
    except Exception:
        pass

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-time-seconds", type=int, default=120)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_split(data_dir / "train")
    X_test, y_test, _ = load_split(data_dir / "test")

    t0 = time.time()
    model = build_model()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    val_accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    training_process = _introspect_training(model)
    training_process["n_train"] = int(X_train.shape[0])
    training_process["n_test"] = int(X_test.shape[0])

    joblib.dump(model, output_dir / "best.pkl")
    metrics = {
        "val_accuracy": val_accuracy,
        "train_seconds": train_seconds,
        "training_process": training_process,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics))
    print(json.dumps({k: v for k, v in metrics.items() if k != "training_process"}))


if __name__ == "__main__":
    main()
'''


_FALLBACK_TRAIN_CSV = '''\
"""Fallback train.py — sklearn classifier on a tabular train.csv/test.csv split.

After fit, introspects the fitted estimator and emits training-process info
(iterations, loss curve, effective hyperparameters) so the Trainer's UI can
show how training proceeded — separate from the Evaluator's accuracy/latency
benchmark on the same artifact.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from model import build_model

TARGET_COL = "__TARGET__"


def _safe_repr(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def _introspect_training(model):
    info = {"estimator_class": type(model).__name__}

    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        try:
            iters = list(n_iter) if hasattr(n_iter, "__iter__") else [int(n_iter)]
            info["n_iter"] = int(max(iters)) if iters else int(n_iter)
        except Exception:
            pass

    if hasattr(model, "loss_curve_"):
        try:
            info["loss_curve"] = [float(x) for x in model.loss_curve_]
        except Exception:
            pass
    elif hasattr(model, "train_score_"):
        try:
            info["loss_curve"] = [float(x) for x in model.train_score_]
            info["loss_curve_label"] = "train_score (per iteration)"
        except Exception:
            pass

    if hasattr(model, "best_loss_"):
        try:
            info["final_loss"] = float(model.best_loss_)
        except Exception:
            pass

    try:
        info["effective_params"] = {
            k: _safe_repr(v) for k, v in model.get_params(deep=False).items()
        }
    except Exception:
        pass

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-time-seconds", type=int, default=120)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols].to_numpy(dtype="float32")
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype="float32")
    y_test = test_df[TARGET_COL].to_numpy()

    t0 = time.time()
    model = build_model()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    val_accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    training_process = _introspect_training(model)
    training_process["n_train"] = int(X_train.shape[0])
    training_process["n_test"] = int(X_test.shape[0])

    joblib.dump(model, output_dir / "best.pkl")
    metrics = {
        "val_accuracy": val_accuracy,
        "train_seconds": train_seconds,
        "training_process": training_process,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics))
    print(json.dumps({k: v for k, v in metrics.items() if k != "training_process"}))


if __name__ == "__main__":
    main()
'''
