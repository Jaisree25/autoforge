"""Training Agent — linear one-pass flow.

Pipeline:

  1. Oracle baseline (sklearn LogReg) — the "must beat by 5 points" gate.
  2. **Generate design.md** (LLM call 1) — Nemotron 9B writes a short
     markdown design doc using full upstream context.
  3. **HITL gate on design.md** — human approves the design BEFORE any
     code is generated. Editing the design changes what model.py commits to.
  4. **Generate model.py** (LLM call 2) — Nemotron 9B writes the sklearn
     estimator with hyperparameters matching the approved design.
  5. AutoForge writes the templated train.py (reads prep_config.json at
     runtime) and drops autoforge_helpers.py beside it.
  6. **Smoke harness** — py_compile, import, train.py --help, design.md
     structure check. One-shot — if smoke fails the Trainer aborts.
  7. **Subprocess training** — run train.py. If it fails, the Trainer aborts.
  8. **Build TrainingResult** — read best.pkl + metrics.json.

No retry loop, no fallback templates, no attempt state machine. One pass,
fail loudly on any error. Keeps the agent honest and the demo flow simple.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
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


# Source for the helper modules the Trainer drops into the training dir.
# autoforge_helpers.py = data loading + joblib output
# autoforge_optuna.py  = sklearn-aware HP search via Optuna
_HELPERS_SRC = (
    Path(__file__).resolve().parent.parent / "tools" / "train_helpers.py"
)
_OPTUNA_SRC = (
    Path(__file__).resolve().parent.parent / "tools" / "optuna_search.py"
)


class TrainerError(Exception):
    """Unrecoverable failure in the linear Trainer pipeline."""


class TrainingAgent(BaseAgent):
    """Linear Trainer — design → HITL → model → train.

    No retries: a one-pass agentic flow that fails loudly. The Researcher
    has already constrained the candidate to sklearn, the Preparer has
    already produced the split + prep_config — by the time we run, the
    LLM has a tight surface to misuse.
    """

    name: ClassVar[AgentName] = AgentName.TRAINING

    def __init__(self, store, run_id: str, hitl=None) -> None:
        super().__init__(store=store, run_id=run_id)
        # 49B model — slower (~30-60s per call) but reliably follows the
        # sklearn-only / rigid-shape constraints. The 9B dropped LightGBM
        # references and produced empty model.py outputs.
        self.llm = NemotronClient(model=COORDINATOR_MODEL)
        self.hitl = hitl  # optional; if None the design gate auto-approves

    # ------------------------------------------------------------------
    def run(  # type: ignore[override]
        self,
        strategy_spec: StrategySpec,
        training_envelope: TrainingEnvelope,
        dataset_profile: DatasetProfile,
        preparation_report: PreparationReport | None = None,
        previous_feedback: dict[str, Any] | None = None,
        attempt_num: int = 1,
    ) -> TrainingResult:
        run_dir = ARTIFACTS_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        training_root = run_dir / "training"
        training_root.mkdir(parents=True, exist_ok=True)
        # Each attempt lives in its own subdir so retries don't clobber prior
        # artifacts. The latest attempt-N directory is the canonical "current
        # result"; the Coordinator surfaces all attempts for the demo.
        code_dir = training_root / f"attempt-{attempt_num}"
        code_dir.mkdir(parents=True, exist_ok=True)
        models_dir = code_dir / "models"
        logs_dir = code_dir / "logs"

        lifecycle_label = (
            f"design → gate → model → train (attempt {attempt_num})"
            if previous_feedback is None
            else f"RETRY (attempt {attempt_num}) with Evaluator feedback"
        )

        with self._lifecycle(lifecycle_label):
            if previous_feedback:
                self.emit_event(
                    EventType.INFO,
                    message=(
                        f"received Evaluator feedback: "
                        f"failure_mode={previous_feedback.get('failure_mode')!r}, "
                        f"gap={previous_feedback.get('accuracy_gap')}, "
                        f"{len(previous_feedback.get('suggestions') or [])} suggestion(s)"
                    ),
                    payload={"feedback": previous_feedback},
                )
            self._check_prepared_data(dataset_profile, preparation_report)
            self._announce_helper_actions(dataset_profile, preparation_report)

            # Oracle baseline removed — the agent's design.md is now judged
            # purely against the StrategySpec.success_threshold. The stub
            # below keeps downstream signatures unchanged for now (callers
            # treat test_accuracy=0.0 as "no oracle context").
            oracle: dict[str, Any] = {
                "model": "(no oracle)",
                "test_accuracy": 0.0,
                "wall_clock_s": 0.0,
                "n_train": 0,
                "n_test": 0,
                "n_features": 0,
            }
            prep_config = self._load_prep_config(preparation_report)
            if prep_config:
                self.emit_event(
                    EventType.INFO,
                    message=(
                        f"loaded prep_config: {', '.join(prep_config.keys())}"
                    ),
                )

            # --- Stage 2: design.md ---
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"nemotron.design_md ({self.llm.model})",
            )
            design_md = tp.generate_design_md(
                llm=self.llm,
                spec=strategy_spec,
                profile=dataset_profile,
                envelope=training_envelope,
                oracle=oracle,
                prep=preparation_report,
                prep_config=prep_config,
                previous_feedback=previous_feedback,
            )
            design_path = code_dir / "design.md"
            design_path.write_text(design_md, encoding="utf-8")
            self.emit_event(
                EventType.INFO,
                message=f"design.md written ({len(design_md)} chars)",
            )

            # --- Stage 3: HITL gate on design.md ---
            approved_design = self._gate_design(
                design_md=design_md,
                oracle=oracle,
                design_path=design_path,
                strategy_spec=strategy_spec,
            )
            if approved_design != design_md:
                design_path.write_text(approved_design, encoding="utf-8")
                design_md = approved_design

            # --- Stage 4: model.py ---
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"nemotron.model_py ({self.llm.model})",
            )
            model_py = tp.generate_model_py(
                llm=self.llm,
                design_md=design_md,
                spec=strategy_spec,
                profile=dataset_profile,
            )
            (code_dir / "model.py").write_text(model_py, encoding="utf-8")
            self.emit_event(
                EventType.INFO,
                message=f"model.py written ({len(model_py)} chars)",
            )

            # --- Stage 5: templated train.py + helpers ---
            train_py = self._render_train_template(dataset_profile)
            (code_dir / "train.py").write_text(train_py, encoding="utf-8")
            shutil.copy2(_HELPERS_SRC, code_dir / "autoforge_helpers.py")
            shutil.copy2(_OPTUNA_SRC, code_dir / "autoforge_optuna.py")
            self.emit_event(
                EventType.INFO,
                message="wrote templated train.py + autoforge_helpers.py",
            )

            # --- Stage 6: smoke harness (one-shot, fail loudly) ---
            self.emit_event(EventType.TOOL_CALL, message="smoke_harness")
            verify = tp.run_smoke_harness(code_dir)
            (code_dir / "verify_report.json").write_text(
                json.dumps(verify, indent=2), encoding="utf-8",
            )
            n_pass = sum(1 for c in verify["checks"] if c["passed"])
            n_total = len(verify["checks"])
            if not verify["overall_passed"]:
                failed = [
                    c["detail"] for c in verify["checks"]
                    if not c["passed"] and c.get("detail")
                ]
                for err in failed[:3]:
                    self.emit_event(
                        EventType.WARNING, message=f"smoke FAILED — {err[:200]}",
                    )
                self._write_status(code_dir, "smoke_failed", "; ".join(failed)[:500])
                raise TrainerError(
                    f"Smoke harness failed ({n_pass}/{n_total}): "
                    + "; ".join(failed)[:500]
                )
            self.emit_event(
                EventType.INFO,
                message=f"smoke PASSED: {n_pass}/{n_total} ✓",
            )

            # --- Stage 7: subprocess training (one-shot, fail loudly) ---
            prepared_dir = self._resolve_prepared_dir(
                dataset_profile, preparation_report,
            )
            max_seconds = int((training_envelope.max_train_minutes or 5.0) * 60)
            self.emit_event(
                EventType.TOOL_CALL,
                message=(
                    f"subprocess.run(python train.py "
                    f"--data-dir {prepared_dir.name} "
                    f"--max-time-seconds {max_seconds})"
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
                stderr_tail = (
                    sub.get("stderr_tail") or sub.get("stdout_tail") or ""
                )
                if stderr_tail:
                    self.emit_event(
                        EventType.WARNING,
                        message=f"train.py stderr (last lines):\n{stderr_tail[:500]}",
                    )
                self._write_status(
                    code_dir, "training_failed",
                    stderr_tail[:1000] or "(no stderr)",
                )
                raise TrainerError(
                    f"Training subprocess failed "
                    f"(return_code={sub['return_code']}, "
                    f"timed_out={sub['timed_out']}). "
                    f"Last stderr:\n{stderr_tail[:1000] or '(none)'}"
                )

            metrics = sub.get("metrics") or {}
            val_acc = float(metrics.get("val_accuracy", 0.0))
            self.emit_event(
                EventType.INFO,
                message=(
                    f"training PASSED: val_accuracy={val_acc:.3f} in "
                    f"{sub_duration:.1f}s"
                    + (" (TIMED OUT; soft success)" if sub["timed_out"] else "")
                ),
                payload={"metrics": metrics},
            )
            self._write_status(
                code_dir, "success",
                f"val_accuracy={val_acc:.3f}, "
                f"train_seconds={metrics.get('train_seconds', sub_duration):.1f}",
            )

            # --- Stage 8: build TrainingResult ---
            return self._build_result(
                code_dir=code_dir,
                sub=sub,
                oracle=oracle,
                spec=strategy_spec,
                verify=verify,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write_status(self, code_dir: Path, status: str, reason: str) -> None:
        payload = {
            "status": status,
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        (code_dir / "status.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )

    def _load_prep_config(
        self, prep: PreparationReport | None,
    ) -> dict[str, Any] | None:
        if prep is None or not prep.prep_config_path:
            return None
        path = Path(prep.prep_config_path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.emit_event(
                EventType.WARNING,
                message=f"could not parse prep_config.json: {exc}",
            )
            return None

    # ------------------------------------------------------------------
    # Build TrainingResult from the training dir
    # ------------------------------------------------------------------
    def _build_result(
        self,
        code_dir: Path,
        sub: dict[str, Any],
        oracle: dict[str, Any],
        spec: StrategySpec,
        verify: dict[str, Any],
    ) -> TrainingResult:
        metrics = sub.get("metrics") or {}
        val_acc = float(metrics.get("val_accuracy", 0.0))
        train_seconds = float(metrics.get(
            "train_seconds", sub.get("duration_s", 0.0),
        ))
        training_process = metrics.get("training_process") or {}

        best_pkl_path = code_dir / "models" / "best.pkl"
        if not best_pkl_path.exists():
            raise TrainerError(
                f"Training reported success but best.pkl is missing at "
                f"{best_pkl_path}"
            )

        passed_checks = sum(1 for c in verify["checks"] if c["passed"])
        total_checks = len(verify["checks"])
        model_id = f"m_{int(time.time())}"

        effective = training_process.get("effective_params") or {}
        best_params: dict[str, Any] = {
            **effective,
            "design_md_path": str(code_dir / "design.md"),
        }

        trials = [
            TrialResult(
                trial_id=0,
                params={"design": "see design.md"},
                score=val_acc,
                duration_seconds=train_seconds,
                status="completed",
            ),
        ]

        return TrainingResult(
            best_model_id=model_id,
            metric_name=spec.success_metric or "accuracy",
            best_score=val_acc,
            best_params=best_params,
            trials_completed=1,
            total_trials=1,
            training_time_seconds=train_seconds,
            artifact_path=str(best_pkl_path),
            library="sklearn",
            all_trials=trials,
            training_process=training_process,
            notes=(
                f"linear-pipeline: trained={val_acc:.3f}. "
                f"Smoke {passed_checks}/{total_checks} passed."
            ),
        )

    # ------------------------------------------------------------------
    # HITL gate on design.md — fires once, before model.py is generated.
    # ------------------------------------------------------------------
    def _gate_design(
        self,
        design_md: str,
        oracle: dict[str, Any],
        design_path: Path,
        strategy_spec: StrategySpec,
    ) -> str:
        if self.hitl is None:
            self.emit_event(
                EventType.INFO,
                message=(
                    "no HITL service wired — auto-approving design.md (dev mode)"
                ),
            )
            return design_md

        first_lines = "\n".join(design_md.splitlines()[:5])
        request = ApprovalRequest(
            run_id=self.run_id,
            agent=AgentName.TRAINING,
            title="Approve design.md (Trainer)",
            description=(
                f"Trainer wrote a design before generating any code. "
                f"Target {strategy_spec.success_metric} ≥ "
                f"{strategy_spec.success_threshold:.2f}. Approve to generate "
                f"model.py and run training; edit to change hyperparameters."
            ),
            payload={
                "summary": f"Trainer wrote design.md ({len(design_md)} chars)",
                "next_agent": "model.py generation",
                "design_md": design_md,
                "design_path": str(design_path),
                "oracle": oracle,
                "preview": first_lines,
            },
        )
        self.emit_event(
            EventType.APPROVAL_REQUESTED,
            message="design.md ready — awaiting approval before model.py codegen",
            payload={
                "summary": (
                    "Approve the proposed design (architecture + HPs)"
                ),
                "next_agent": "model.py generation",
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
                return edited

        return design_md

    # ------------------------------------------------------------------
    # Precondition check — tabular split must exist.
    # ------------------------------------------------------------------
    def _check_prepared_data(
        self,
        profile: DatasetProfile,
        prep: PreparationReport | None,
    ) -> None:
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
                f"Preparer did not produce train.csv/test.csv at "
                f"{prepared}. Ensure `train_test_split_csv` ran."
            )
        self.emit_event(
            EventType.INFO,
            message=f"precondition OK: tabular split present at `{prepared}`",
        )

    # ------------------------------------------------------------------
    # Transparency for AutoForge's auto-fixes — announce what the helpers
    # will do BEFORE the Oracle / training subprocess runs, so the dashboard
    # and Slack feed show the safety nets firing.
    # ------------------------------------------------------------------
    def _announce_helper_actions(
        self,
        profile: DatasetProfile,
        prep: PreparationReport | None,
    ) -> None:
        if prep is None or not prep.prepared_dataset_path:
            return
        prepared = Path(prep.prepared_dataset_path)
        train_csv = prepared / "train.csv"
        if not train_csv.is_file():
            return
        try:
            import pandas as pd
            df = pd.read_csv(train_csv, nrows=200)
        except Exception:  # noqa: BLE001
            return
        target_col = profile.target_column
        id_tokens = ("_id", "customer_id", "user_id", "id_", "uuid")

        def _is_id_col(name: str) -> bool:
            n = name.lower()
            if n == "id":
                return True
            if any(t in n for t in id_tokens):
                return True
            return n.endswith("id") and len(n) > 3

        def _is_high_card(col_name: str) -> bool:
            try:
                n = len(df)
                if n == 0:
                    return False
                nunique = df[col_name].nunique(dropna=True)
                if nunique / n > 0.9:
                    return True
                if df[col_name].dtype == object and nunique > 50:
                    return True
            except Exception:  # noqa: BLE001
                pass
            return False

        # ID-like columns AutoForge will drop at load time.
        id_cols = [
            c for c in df.columns
            if c != target_col and _is_id_col(c)
        ]
        # High-cardinality / near-unique columns AutoForge will also drop.
        high_card_cols = [
            c for c in df.columns
            if c != target_col and c not in id_cols and _is_high_card(c)
        ]
        if id_cols:
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"AutoForge safety net: will drop ID-like column(s) "
                    f"{id_cols!r} (Preparer left them in; they would leak)"
                ),
                payload={"dropped_id_columns": id_cols},
            )
        if high_card_cols:
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"AutoForge safety net: will drop high-cardinality / "
                    f"near-unique column(s) {high_card_cols!r} (would "
                    f"explode one-hot encoding or behave like an ID)"
                ),
                payload={"dropped_high_card_columns": high_card_cols},
            )

        # Non-numeric feature columns AutoForge will auto-one-hot at load time.
        already_dropped = set(id_cols) | set(high_card_cols)
        obj_cols = [
            c for c in df.columns
            if c != target_col and c not in already_dropped
            and df[c].dtype == object
        ]
        if obj_cols:
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"AutoForge safety net: will auto-one-hot non-numeric "
                    f"column(s) {obj_cols!r} (Preparer's encode_categoricals "
                    f"did not cover them)"
                ),
                payload={"auto_encoded_columns": obj_cols},
            )

        # Missing-value fills AutoForge will apply.
        nan_cols = [
            c for c in df.columns
            if c != target_col and c not in already_dropped
            and df[c].isna().any()
        ]
        if nan_cols:
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"AutoForge safety net: will fill NaN in {nan_cols!r} "
                    f"(numeric → column median, object → mode/'missing')"
                ),
                payload={"nan_filled_columns": nan_cols},
            )

        # Non-numeric target → auto-encode to int labels.
        if target_col and target_col in df.columns and df[target_col].dtype == object:
            self.emit_event(
                EventType.WARNING,
                message=(
                    f"AutoForge safety net: will auto-encode non-numeric "
                    f"target `{target_col}` to int class labels"
                ),
            )

        if not (id_cols or high_card_cols or obj_cols or nan_cols):
            self.emit_event(
                EventType.INFO,
                message=(
                    "data inspection: prepared CSV is clean — "
                    "no safety nets needed"
                ),
            )

    # ------------------------------------------------------------------
    def _resolve_prepared_dir(
        self,
        profile: DatasetProfile,
        prep: PreparationReport | None,
    ) -> Path:
        if prep and prep.prepared_dataset_path:
            candidate = Path(prep.prepared_dataset_path)
            if candidate.exists():
                return candidate
        src = Path(profile.dataset_path)
        if src.is_file():
            return src.parent
        return src

    # ------------------------------------------------------------------
    # Templated train.py — AutoForge owns this so it's reliable.
    # ------------------------------------------------------------------
    def _render_train_template(self, profile: DatasetProfile) -> str:
        target_col = profile.target_column or "target"
        return _TRAIN_PY_TABULAR.replace("__TARGET__", target_col)


# ===========================================================================
# Templated train.py runner. AutoForge writes this directly into the
# training directory next to the LLM's model.py. It reads prep_config.json
# (the Preparer's output) at runtime and applies feature_scaling as recorded.
# The autoforge_helpers module (also dropped next to it) handles joblib +
# metrics boilerplate.
# ===========================================================================
_TRAIN_PY_TABULAR = '''\
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from autoforge_helpers import load_csv_split, save_outputs
from autoforge_optuna import run_optuna_search
from model import build_model


TARGET_COL = "__TARGET__"
N_OPTUNA_TRIALS = 10

_SCALER_BY_METHOD = {
    "standard": "StandardScaler",
    "minmax": "MinMaxScaler",
    "robust": "RobustScaler",
}


def _load_prep_config(data_dir: Path) -> dict:
    for cand in [
        data_dir.parent / "prep_config.json",
        data_dir / "prep_config.json",
    ]:
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return {}


def _build_scaler(method: str):
    import sklearn.preprocessing as pp
    cls = getattr(pp, _SCALER_BY_METHOD.get(method, "StandardScaler"), None)
    return cls() if cls is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-time-seconds", type=int, default=120)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_csv_split(data_dir, TARGET_COL)

    prep_config = _load_prep_config(data_dir)
    scaling = prep_config.get("feature_scaling") or {}
    method = scaling.get("method")
    if method:
        scaler = _build_scaler(method)
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    # Identify the sklearn class so Optuna can pick the right search space.
    # The Trainer's `build_model()` is generated with default kwargs; calling
    # it once with no overrides yields a base instance whose class we read.
    sample = build_model()
    sklearn_class = type(sample).__name__

    t0 = time.time()
    model, best_params, trials = run_optuna_search(
        build_model_fn=build_model,
        sklearn_class=sklearn_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        n_trials=N_OPTUNA_TRIALS,
    )
    train_seconds = time.time() - t0

    # `.score()` returns accuracy for classifiers, R² for regressors —
    # works for both task types without branching on metric name.
    val_accuracy = float(model.score(X_test, y_test))
    headline = save_outputs(
        model, output_dir, val_accuracy, train_seconds,
        n_train=len(X_train), n_test=len(X_test),
    )
    headline["best_optuna_params"] = best_params
    headline["n_optuna_trials"] = len(trials)
    print(json.dumps(headline))


if __name__ == "__main__":
    main()
'''
