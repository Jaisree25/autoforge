"""Agentic Trainer pipeline.
 
The TrainingAgent drives this in stages:
 
  1. Oracle baseline (sklearn LogReg) — the "must beat by 5 points" gate.
  2. **Generate design.md + model.py** — Nemotron writes both from the
     upstream Profiler/Researcher/Preparer context. AutoForge writes the
     templated train.py (modality-specific, reads prep_config.json at
     runtime to apply normalization or feature_scaling automatically).
  3. **Smoke harness** — verify generated code: py_compile, import,
     `build_model()` instantiates, `train.py --help` works, AND a
     semantic dry-run fit/predict on synthetic data catches hallucinated
     sklearn API calls that pass syntax. Failures feed back to the LLM
     as retry context.
  4. **HITL gate on design.md** — fires once, after the first smoke-passing
     attempt, before any training subprocess runs.
  5. **Subprocess training** — run train.py with a wall-clock cap. Capture
     stdout/stderr. Runtime failures also feed back as retry context.
  6. **Build + return TrainingResult** — read best.pkl + metrics.json from
     the accepted attempt directory.
 
Each attempt lives in its own folder under `training/in-progress/attempt-N/`
and is moved into `training/failed/` or `training/done/` based on outcome.
"""
from __future__ import annotations
 
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
 
from pydantic import BaseModel, ConfigDict, Field
 
from agents._llm_client import NemotronClient
from contracts.schemas import (
    DatasetProfile,
    Modality,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
)
from tools import training_tools as tt
 
 
# ===========================================================================
# Stage 1: Oracle baseline
# ===========================================================================
def run_oracle(
    profile: DatasetProfile,
    prep: PreparationReport | None,
    run_dir: Path,
) -> dict[str, Any]:
    """sklearn.LogisticRegression on flattened input. Save `oracle.json`.
 
    The oracle plays the same role as GCC in the C-compiler project: a
    known-good baseline the generated model must beat by ≥5 points.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
 
    t0 = time.time()
    if profile.modality == Modality.IMAGE:
        prepared = (
            Path(prep.prepared_dataset_path)
            if (prep and prep.prepared_dataset_path) else None
        )
        if prepared and (prepared / "train").is_dir() and (prepared / "test").is_dir():
            X_train, y_train, _ = tt.load_image_folder(prepared / "train")
            X_test, y_test, _ = tt.load_image_folder(prepared / "test")
        else:
            from sklearn.model_selection import train_test_split
            X, y, _ = tt.load_image_folder(Path(profile.dataset_path))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
    else:
        target = profile.target_column or "target"
        prepared_dir = (
            Path(prep.prepared_dataset_path)
            if (prep and prep.prepared_dataset_path) else None
        )
        X_train, y_train, X_test, y_test = tt.load_csv_split_or_full(
            prepared_dir=prepared_dir,
            fallback_csv=Path(profile.dataset_path),
            target_column=target,
        )
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
            )
 
    model = LogisticRegression(max_iter=300, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, model.predict(X_test)))
    wall = time.time() - t0
 
    oracle = {
        "model": "sklearn.linear_model.LogisticRegression",
        "test_accuracy": acc,
        "wall_clock_s": wall,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]) if X_train.ndim > 1 else 1,
    }
    (run_dir / "oracle.json").write_text(json.dumps(oracle, indent=2))
    return oracle
 
 
# ===========================================================================
# Stage 2: Generate design.md + model.py in ONE call.
# ===========================================================================
class _DesignAndCode(BaseModel):
    """Structured output: design.md + model.py."""
    model_config = ConfigDict(extra="forbid")
    design_md: str = Field(
        description=(
            "Markdown design doc (~400-700 words). MUST have all 7 sections "
            "listed in the system prompt. Concrete values everywhere — no "
            "ranges, no `tbd`, no `~`. Every hyperparameter line has a rationale."
        ),
    )
    model_py: str = Field(
        description=(
            "Contents of model.py. Defines exactly one function "
            "`build_model()` returning an unfitted sklearn estimator with the "
            "exact hyperparameter values committed to in design.md. "
            "No `if __name__ == '__main__'` — this file is imported only. "
            "Top-level code may import sklearn modules but MUST NOT do any I/O."
        ),
    )
 
 
def generate_design_and_code(
    llm: NemotronClient,
    spec: StrategySpec,
    profile: DatasetProfile,
    envelope: TrainingEnvelope,
    oracle: dict[str, Any],
    prep: PreparationReport | None,
    prep_config: dict[str, Any] | None = None,
    on_thinking=None,
    no_think: bool = True,
    previous_errors: list[str] | None = None,
) -> dict[str, str]:
    """ONE Nemotron call → {design_md, model.py}."""
    arch = spec.candidate_architectures[0] if spec.candidate_architectures else None
    prepared_path = (
        prep.prepared_dataset_path if (prep and prep.prepared_dataset_path)
        else profile.dataset_path
    )
    modality = profile.modality.value
 
    system = (
        "You are the Trainer agent. Produce TWO artifacts as one structured "
        "JSON object: design.md + model.py.\n\n"
        "AutoForge writes train.py (templated by modality, reads "
        "prep_config.json at runtime, applies normalization/feature_scaling "
        "automatically). You do NOT need to worry about data loading, "
        "preprocessing application, file saving, or argparse — that's all "
        "handled. Your job is the architectural choice and its concrete "
        "implementation.\n\n"
        "================================================================\n"
        "## CRITICAL CONTRACT — output rejected if you violate ANY:\n"
        "================================================================\n"
        "1. model.py defines a TOP-LEVEL FUNCTION `def build_model():`. "
        "   NOT a class. NOT a module-level `model = ...` variable. A FUNCTION.\n"
        "2. The function takes ZERO arguments and returns an unfitted sklearn "
        "estimator instance.\n"
        "3. model.py has NO `if __name__ == '__main__':` block. NO I/O. NO "
        "training. The Trainer's templated train.py calls build_model() then "
        "fits the result.\n"
        "4. Imports in model.py: ONLY `sklearn.*`. Nothing else.\n"
        "5. Hyperparameter values in model.py MUST match design.md exactly.\n\n"
        "Upstream agents have ALREADY produced: Profiler (DatasetProfile + "
        "TrainingEnvelope), Researcher (StrategySpec with one committed "
        "architecture), Preparer (PreparationReport + optional prep_config).\n"
        "DO NOT re-derive any of this. COMMIT and SHIP.\n\n"
        "================================================================\n"
        "## 1. design.md — Markdown commitment doc (~400-700 words)\n"
        "================================================================\n"
        "design_md MUST be plain Markdown prose. It is NOT a JSON dump of "
        "the inputs you received — write English sentences. The smoke harness "
        "REJECTS design.md unless it contains all seven required headers "
        "exactly as written below.\n\n"
        "Required structure (literal copy/paste — use these exact lines as "
        "section headers, then write 1-3 sentences under each):\n\n"
        "```markdown\n"
        "## Architecture commitment\n"
        "<one line: concrete sklearn class chosen (e.g. MLPClassifier).>\n"
        "<one line: why this beats the runner-up for THIS dataset.>\n"
        "\n"
        "## Hyperparameters (final)\n"
        "- `hidden_layer_sizes = (128, 64)` — Two-layer MLP captures pixel-level features.\n"
        "- `alpha = 1e-3` — Mild L2; dataset is small.\n"
        "- `random_state = 42` — Reproducibility.\n"
        "(One bullet per HP, EXACT value, one-line rationale each. NO ranges, no `tbd`, no `~`.)\n"
        "\n"
        "## Wall-clock budget\n"
        "<estimated train seconds + the envelope cap + what happens at the cap>\n"
        "\n"
        "## Success criteria\n"
        "<hard target (from StrategySpec) + oracle delta requirement (must beat by ≥ 0.05)\n"
        "+ one concrete rollback trigger>\n"
        "\n"
        "## Risks & anti-patterns\n"
        "<one risk for this architecture on this dataset shape>\n"
        "<one anti-pattern avoided and why>\n"
        "\n"
        "## Code structure\n"
        "- `model.py` exports `build_model() -> sklearn estimator`.\n"
        "- AutoForge's templated train.py loads the split, applies "
        "prep_config, fits, scores, saves `best.pkl` + `metrics.json`.\n"
        "\n"
        "## Verification plan\n"
        "- Smoke: py_compile model.py, import build_model, "
        "build_model() instantiates.\n"
        "- Post-train: val_accuracy >= oracle_accuracy + 0.05.\n"
        "```\n\n"
        "DO NOT emit a JSON object inside the design_md string. DO NOT echo "
        "the input data structure. WRITE ACTUAL MARKDOWN PROSE.\n\n"
        "================================================================\n"
        "## 2. model.py — minimal, sklearn-only\n"
        "================================================================\n"
        "Defines EXACTLY one function `build_model()` returning an unfitted "
        "estimator with the EXACT hyperparameter values from design.md's "
        "Hyperparameters section.\n\n"
        "### model.py shape (rigid)\n"
        "```python\n"
        "from sklearn.<your_module> import <YourClass>\n"
        "\n"
        "def build_model():\n"
        "    return <YourClass>(\n"
        "        <hp_1>=<value_1>,\n"
        "        <hp_2>=<value_2>,\n"
        "        random_state=42,\n"
        "    )\n"
        "```\n"
        "Allowed sklearn classes: MLPClassifier, LogisticRegression, "
        "RandomForestClassifier, GradientBoostingClassifier, SVC, "
        "KNeighborsClassifier, DecisionTreeClassifier.\n\n"
        "Imports in model.py: ONLY `sklearn.*`. No I/O. No top-level "
        "computation. No `if __name__ == '__main__'`.\n\n"
        "### Code quality (smoke harness will reject violations)\n"
        "- Every f-string properly terminated; every bracket matched.\n"
        "- model.py MUST `python -m py_compile` cleanly.\n"
        "- model.py MUST `import` cleanly and `build_model()` MUST return "
        "an estimator.\n"
        "- Hyperparameter values in model.py MUST match design.md exactly.\n\n"
        "### SEMANTIC DRY-RUN (smoke harness will reject violations)\n"
        "The smoke harness calls `.fit(X, y)` and `.predict(X)` on 10 rows "
        "of synthetic data BEFORE training. Your model MUST:\n"
        "- Accept `.fit(X, y)` where X is float32 numpy array, y is integer labels.\n"
        "- Return a 1-D array from `.predict(X)` with shape (n_samples,).\n"
        "- Have `.fit` and `.predict` methods (i.e. be a real sklearn estimator).\n"
        "Common hallucination traps that WILL be caught and rejected:\n"
        "- Nonexistent constructor kwargs (e.g. `n_neurons=` on MLPClassifier).\n"
        "- Wrong kwarg types (e.g. passing a string where a tuple is required).\n"
        "- Estimators that fit but return wrong output shape.\n"
        "- Pipeline objects that don't accept plain numpy input to `.fit`."
    )
 
    effective_target = max(
        spec.success_threshold, oracle["test_accuracy"] + 0.05,
    )
    user_lines = [
        f"## Inputs",
        f"- **Modality:** `{modality}`",
        f"- **Task type:** `{profile.task_type.value}`",
        f"- **Target column (tabular only):** `{profile.target_column}`",
        f"- **Prepared data dir (the value of --data-dir at runtime):** `{prepared_path}`",
        f"- **Samples:** {profile.n_rows:,} "
        + (f"(classes: {profile.n_classes})" if profile.n_classes else ""),
        f"- **Wall-clock cap:** {envelope.max_train_minutes * 60:.0f}s",
    ]
    if modality == "image":
        user_lines.append(
            f"- **Image channels:** {profile.image_channels} "
            f"(convert to grayscale with `.convert('L')` if channels=1, else 'RGB')"
        )
        if profile.image_resolutions:
            res_w, res_h = profile.image_resolutions[0]
            user_lines.append(
                f"- **First-sample resolution (W×H):** {res_w}×{res_h}. "
                f"Assume all images share this resolution; flatten to "
                f"{res_w * res_h * (profile.image_channels or 1)} features."
            )
    user_lines += [
        "",
        f"## Targets",
        f"- **Hard target:** `{spec.success_metric}` ≥ {spec.success_threshold:.3f}",
        f"- **Oracle baseline (sklearn LogReg):** test_accuracy = "
        f"{oracle['test_accuracy']:.3f} (measured in "
        f"{oracle.get('wall_clock_s', 0.0):.1f}s on "
        f"n_train={oracle.get('n_train', '?')} / n_test={oracle.get('n_test', '?')}).",
        f"- **Effective target** (max of hard target and oracle+0.05): "
        f"≥ {effective_target:.3f}.",
        "",
        "## Researcher's committed architecture (already picked by human at HITL)",
    ]
    if arch is not None:
        user_lines.append(f"- **Name:** `{arch.name}`")
        user_lines.append(f"- **Family:** `{arch.family}`")
        user_lines.append(f"- **Library:** `{arch.library}`")
        user_lines.append(f"- **Rationale (from Researcher):** {arch.rationale}")
        user_lines.append(f"- **Hyperparameter space:** `{arch.hyperparameter_space}`")
        user_lines.append("")
        user_lines.append(
            "→ Pick ONE concrete value per hyperparameter. Use these as your "
            "starting point; you may add `random_state=42` and override any "
            "value the search space gave as a range."
        )
    user_lines.append("")
    user_lines.append("## Preparer's applied operations")
    if prep is not None and prep.operations:
        for op in prep.operations:
            user_lines.append(f"- `{op}`")
    else:
        user_lines.append("- (no operations recorded — load the raw split as-is)")
    user_lines.append("")
    user_lines.append("## Preparer's prep_config (applied at runtime by train.py)")
    if prep_config:
        user_lines.append("```json")
        user_lines.append(json.dumps(prep_config, indent=2, default=str))
        user_lines.append("```")
        user_lines.append(
            "AutoForge's templated train.py will read this config and apply it "
            "(normalization mean/std for images, sklearn preprocessing scaler "
            "for tabular). You do NOT need to bake it into model.py — just "
            "design `build_model()` knowing the input arrives pre-normalized "
            "for images or pre-scaled for tabular."
        )
    else:
        user_lines.append(
            "- (no prep_config — no normalization or feature scaling will be applied)"
        )
    user_lines.append("")
    if previous_errors:
        user_lines.append("## PREVIOUS ATTEMPT FAILED")
        user_lines.append("Fix these specific errors in this attempt:")
        for err in previous_errors:
            user_lines.append(f"- {err}")
        user_lines.append("")
        user_lines.append(
            "Re-check every f-string, bracket, quote. Make sure model.py "
            "passes `python -m py_compile`. If the previous attempt's TRAIN "
            "SUBPROCESS failed and the error mentions `build_model()`, the "
            "bug is in model.py — check the function signature and return type."
        )
        user_lines.append("")
    user_lines.append(
        "Produce the structured JSON now: design.md (all 7 sections, "
        "concrete values, rationales on every hyperparameter line) + model.py "
        "(rigid shape from system prompt — `def build_model():` returning an "
        "unfitted sklearn estimator)."
    )
    user = "\n".join(user_lines)
 
    result = llm.think_and_answer_structured(
        system=system,
        user=user,
        schema=_DesignAndCode,
        on_thinking=on_thinking,
        max_tokens=12000,
        temperature=0.2,
        no_think=no_think,
    )
    return {
        "design_md": result.design_md,
        "model.py": result.model_py,
    }
 
 
# ===========================================================================
# Stage 3: Smoke harness
# ===========================================================================
 
# The set of sklearn constructor kwargs that are valid for each allowed class.
# Used in Check 5 to detect hallucinated kwargs before even running the LLM
# output — gives a precise error message rather than a cryptic TypeError.
_SKLEARN_VALID_PARAMS: dict[str, set[str]] = {
    "MLPClassifier": {
        "hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
        "learning_rate", "learning_rate_init", "power_t", "max_iter", "shuffle",
        "random_state", "tol", "verbose", "warm_start", "momentum",
        "nesterovs_momentum", "early_stopping", "validation_fraction",
        "beta_1", "beta_2", "epsilon", "n_iter_no_change", "max_fun",
    },
    "LogisticRegression": {
        "penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling",
        "class_weight", "random_state", "solver", "max_iter", "multi_class",
        "verbose", "warm_start", "n_jobs", "l1_ratio",
    },
    "RandomForestClassifier": {
        "n_estimators", "criterion", "max_depth", "min_samples_split",
        "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
        "max_leaf_nodes", "min_impurity_decrease", "bootstrap", "oob_score",
        "n_jobs", "random_state", "verbose", "warm_start", "class_weight",
        "ccp_alpha", "max_samples",
    },
    "GradientBoostingClassifier": {
        "loss", "learning_rate", "n_estimators", "subsample", "criterion",
        "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf",
        "max_depth", "min_impurity_decrease", "init", "random_state",
        "max_features", "verbose", "max_leaf_nodes", "warm_start",
        "validation_fraction", "n_iter_no_change", "tol", "ccp_alpha",
    },
    "SVC": {
        "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability",
        "tol", "cache_size", "class_weight", "verbose", "max_iter",
        "decision_function_shape", "break_ties", "random_state",
    },
    "KNeighborsClassifier": {
        "n_neighbors", "weights", "algorithm", "leaf_size", "p", "metric",
        "metric_params", "n_jobs",
    },
    "DecisionTreeClassifier": {
        "criterion", "splitter", "max_depth", "min_samples_split",
        "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
        "random_state", "max_leaf_nodes", "min_impurity_decrease",
        "class_weight", "ccp_alpha",
    },
}
 
 
def run_smoke_harness(
    code_dir: Path,
    n_features: int = 16,
    n_classes: int = 2,
) -> dict[str, Any]:
    """Verify generated code is syntactically, structurally, AND semantically sane.
 
    Checks (in order):
      1. File existence + py_compile for model.py and train.py.
      2. import model + build_model() instantiates.
      3. train.py --help (argparse contract).
      4. design.md has all 7 required Markdown headers (not a JSON dump).
      5. **Semantic dry-run** — fit + predict on 10 rows of synthetic data.
         Catches hallucinated constructor kwargs and wrong API calls that
         produce valid Python but fail at runtime.
 
    `n_features` and `n_classes` can be overridden by callers that know the
    real dataset shape (see `run_smoke_harness_for_profile` below). The
    defaults (16 features, 2 classes) are conservative enough to catch the
    most common hallucinations without being dataset-specific.
 
    Returns a dict with `overall_passed: bool` and per-check details.
    """
    checks: list[dict[str, Any]] = []
 
    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
 
    # ------------------------------------------------------------------
    # Check 1: file exists + py_compile
    # ------------------------------------------------------------------
    for filename in ("model.py", "train.py"):
        path = code_dir / filename
        if not path.exists():
            add(f"file_exists_{filename}", False, f"ERROR: {filename} missing")
            continue
        add(f"file_exists_{filename}", True)
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if compile_result.returncode != 0:
            add(f"syntax_{filename}", False,
                f"ERROR: {compile_result.stderr.strip()[:300]}")
        else:
            add(f"syntax_{filename}", True)
 
    syntax_ok = all(c["passed"] for c in checks if c["name"].startswith("syntax_"))
    if not syntax_ok:
        return {
            "overall_passed": False,
            "checks": checks,
            "reason": "ERROR: syntax check(s) failed; skipping import + build",
        }
 
    # ------------------------------------------------------------------
    # Check 2: import + has build_model
    # ------------------------------------------------------------------
    import_test = (
        f"import sys; sys.path.insert(0, r'{code_dir}'); "
        "from model import build_model; "
        "m = build_model(); "
        "print('IMPORT_OK type=' + type(m).__name__)"
    )
    import_result = subprocess.run(
        [sys.executable, "-c", import_test],
        capture_output=True, text=True, timeout=60,
    )
    if import_result.returncode != 0 or "IMPORT_OK" not in import_result.stdout:
        err = import_result.stderr.strip()[:500] or import_result.stdout.strip()[:500]
        add("import_build_model", False, f"ERROR: {err}")
    else:
        add("import_build_model", True, import_result.stdout.strip())
 
    # ------------------------------------------------------------------
    # Check 3: train.py --help (argparse contract)
    # ------------------------------------------------------------------
    train_help = subprocess.run(
        [sys.executable, str(code_dir / "train.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(code_dir),
    )
    if train_help.returncode != 0:
        err = train_help.stderr.strip()[:500] or train_help.stdout.strip()[:500]
        add("train_py_help", False, f"ERROR: train.py --help failed: {err}")
    else:
        usage = train_help.stdout
        missing = [
            f for f in ("--data-dir", "--output-dir", "--max-time-seconds")
            if f not in usage
        ]
        if missing:
            add(
                "train_py_help", False,
                f"ERROR: train.py --help missing required flags: {missing}",
            )
        else:
            add("train_py_help", True, "argparse contract OK")
 
    # ------------------------------------------------------------------
    # Check 4: design.md structure (real Markdown, not a JSON dump)
    # ------------------------------------------------------------------
    design_path = code_dir / "design.md"
    if design_path.exists():
        design_text = design_path.read_text(encoding="utf-8").strip()
        required_headers = [
            "## Architecture commitment",
            "## Hyperparameters",
            "## Wall-clock budget",
            "## Success criteria",
            "## Risks",
            "## Code structure",
            "## Verification plan",
        ]
        missing_headers = [h for h in required_headers if h not in design_text]
        starts_like_json = design_text.startswith("{") or design_text.startswith("[")
        if starts_like_json:
            add(
                "design_md_format", False,
                "ERROR: design.md starts with a JSON brace — must be plain "
                "Markdown prose. Re-read the system prompt's required "
                "section structure and write English text, not JSON.",
            )
        elif missing_headers:
            add(
                "design_md_format", False,
                f"ERROR: design.md missing required headers: {missing_headers}. "
                f"Each section MUST appear on its own line as a level-2 "
                f"heading (e.g. `## Architecture commitment`).",
            )
        else:
            add("design_md_format", True, "Markdown structure OK")
 
    # ------------------------------------------------------------------
    # Check 5: Semantic dry-run — fit + predict on synthetic data.
    #
    # This is the main fix for hallucinated sklearn API calls. The LLM
    # frequently generates valid-looking Python that references nonexistent
    # constructor kwargs, wrong kwarg types, or estimators with the wrong
    # output shape. All of these pass py_compile and import cleanly but fail
    # the moment sklearn tries to use them. We catch them here with a real
    # sklearn execution on 10 rows of throwaway data, before any training
    # subprocess runs.
    #
    # We run this in a subprocess (not in-process) so a broken model.py
    # can't corrupt the parent's sklearn state or leave half-fitted objects
    # behind.
    # ------------------------------------------------------------------
    dryrun_script = f"""
import sys
sys.path.insert(0, r'{code_dir}')
 
import numpy as np
import json
 
results = {{}}
 
try:
    from model import build_model
except Exception as e:
    print(json.dumps({{"ok": False, "stage": "import", "error": str(e)}}))
    sys.exit(0)
 
try:
    model = build_model()
except Exception as e:
    print(json.dumps({{"ok": False, "stage": "instantiate", "error": str(e)}}))
    sys.exit(0)
 
# Check estimator surface (fit + predict must exist)
for method in ("fit", "predict"):
    if not hasattr(model, method):
        print(json.dumps({{
            "ok": False,
            "stage": "api_surface",
            "error": f"build_model() returned {{type(model).__name__}} which has no .{{method}}() method"
        }}))
        sys.exit(0)
 
# Check for hallucinated kwargs using get_params()
try:
    actual_params = set(model.get_params(deep=False).keys())
    cls_name = type(model).__name__
    known_params = {json.dumps(list(_SKLEARN_VALID_PARAMS.get(cls_name, set())))}
    known_params = set(known_params) if known_params else None
    if known_params is not None:
        hallucinated = actual_params - known_params
        if hallucinated:
            print(json.dumps({{
                "ok": False,
                "stage": "param_check",
                "error": (
                    f"build_model() returned {{cls_name}} with unrecognised "
                    f"kwargs: {{sorted(hallucinated)}}. These are not valid "
                    f"sklearn {{cls_name}} parameters. Check the sklearn docs."
                )
            }}))
            sys.exit(0)
except Exception:
    pass  # get_params() failure is non-fatal here; fit() will catch it
 
# Synthetic fit + predict
n_samples = 10
n_features = {n_features}
n_classes = {n_classes}
rng = np.random.default_rng(0)
X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
y = rng.integers(0, n_classes, size=n_samples)
 
try:
    model.fit(X, y)
except Exception as e:
    print(json.dumps({{
        "ok": False,
        "stage": "fit",
        "error": (
            f"model.fit() raised {{type(e).__name__}}: {{e}}. "
            "This usually means a hallucinated constructor kwarg or wrong "
            "kwarg type. Check that every kwarg in build_model() is a valid "
            "sklearn parameter with the correct type."
        )
    }}))
    sys.exit(0)
 
try:
    preds = model.predict(X)
except Exception as e:
    print(json.dumps({{
        "ok": False,
        "stage": "predict",
        "error": f"model.predict() raised {{type(e).__name__}}: {{e}}"
    }}))
    sys.exit(0)
 
# Output shape check
if not hasattr(preds, "shape") or len(preds.shape) != 1:
    shape = getattr(preds, "shape", type(preds).__name__)
    print(json.dumps({{
        "ok": False,
        "stage": "predict_shape",
        "error": (
            f"model.predict() returned shape {{shape}}; expected 1-D array "
            f"of shape ({{n_samples}},). Ensure build_model() returns a "
            "classifier, not a transformer or multi-output estimator."
        )
    }}))
    sys.exit(0)
 
if preds.shape[0] != n_samples:
    print(json.dumps({{
        "ok": False,
        "stage": "predict_shape",
        "error": (
            f"model.predict() returned {{preds.shape[0]}} predictions for "
            f"{{n_samples}} inputs; expected {{n_samples}}."
        )
    }}))
    sys.exit(0)
 
print(json.dumps({{
    "ok": True,
    "estimator": type(model).__name__,
    "params": {{k: str(v) for k, v in model.get_params(deep=False).items()}},
    "n_features": n_features,
    "n_classes": n_classes,
    "pred_shape": list(preds.shape),
}}))
"""
 
    dryrun_result = subprocess.run(
        [sys.executable, "-c", dryrun_script],
        capture_output=True, text=True, timeout=60,
    )
 
    dryrun_stdout = dryrun_result.stdout.strip()
    if dryrun_result.returncode != 0 or not dryrun_stdout:
        # Subprocess itself crashed (OOM, killed, etc.) — surface stderr
        err = dryrun_result.stderr.strip()[:500] or "(no output)"
        add(
            "semantic_dryrun", False,
            f"ERROR: dry-run subprocess crashed: {err}",
        )
    else:
        try:
            dryrun_data = json.loads(dryrun_stdout)
        except json.JSONDecodeError:
            add(
                "semantic_dryrun", False,
                f"ERROR: dry-run produced non-JSON output: {dryrun_stdout[:300]}",
            )
        else:
            if dryrun_data.get("ok"):
                detail = (
                    f"estimator={dryrun_data.get('estimator')} "
                    f"fit+predict OK on "
                    f"({dryrun_data.get('n_features')}f, "
                    f"{dryrun_data.get('n_classes')}cls) "
                    f"pred_shape={dryrun_data.get('pred_shape')}"
                )
                add("semantic_dryrun", True, detail)
            else:
                stage = dryrun_data.get("stage", "unknown")
                error = dryrun_data.get("error", "(no error message)")
                add(
                    "semantic_dryrun", False,
                    f"ERROR [{stage}]: {error}",
                )
 
    overall = all(c["passed"] for c in checks)
    return {
        "overall_passed": overall,
        "checks": checks,
        "reason": "" if overall else "ERROR: one or more checks failed",
    }
 
 
def run_smoke_harness_for_profile(
    code_dir: Path,
    profile: DatasetProfile,
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Wrapper that derives n_features + n_classes from the real dataset profile.
 
    Use this instead of run_smoke_harness() when you have a DatasetProfile
    available — it makes the dry-run more realistic and catches shape mismatches
    that a generic 16-feature test would miss (e.g. SVC on 784-feature MNIST
    with wrong kernel settings).
    """
    n_features = oracle.get("n_features", 16)
    n_classes = profile.n_classes or 2
    return run_smoke_harness(
        code_dir=code_dir,
        n_features=n_features,
        n_classes=n_classes,
    )
 
 
# ===========================================================================
# Stage 4: Subprocess training
# ===========================================================================
def run_training_subprocess(
    code_dir: Path,
    prepared_dir: Path,
    output_dir: Path,
    log_dir: Path,
    max_time_seconds: int,
) -> dict[str, Any]:
    """Run `python code/train.py --data-dir ... --output-dir ... --max-time-seconds N`.
 
    Captures stdout/stderr to log files. Returns dict with metrics + status.
    Soft-success: if process times out but `best.pkl` exists, treat as success.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "train_stdout.log"
    stderr_log = log_dir / "train_stderr.log"
 
    cmd = [
        sys.executable,
        str(code_dir / "train.py"),
        "--data-dir", str(prepared_dir),
        "--output-dir", str(output_dir),
        "--max-time-seconds", str(max_time_seconds),
    ]
 
    t0 = time.time()
    timed_out = False
    return_code: int | None = None
    try:
        with open(stdout_log, "w", encoding="utf-8") as out_f, \
             open(stderr_log, "w", encoding="utf-8") as err_f:
            result = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                timeout=max_time_seconds + 30,  # small grace
                cwd=str(code_dir),
            )
            return_code = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = time.time() - t0
 
    best_pkl = output_dir / "best.pkl"
    metrics_json = output_dir / "metrics.json"
 
    metrics: dict[str, Any] = {}
    if metrics_json.exists():
        try:
            metrics = json.loads(metrics_json.read_text())
        except Exception:  # noqa: BLE001
            metrics = {"parse_error": "metrics.json present but unparseable"}
 
    def _tail(p: Path, n: int = 50) -> str:
        if not p.exists():
            return ""
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-n:])
        except Exception:  # noqa: BLE001
            return ""
 
    soft_success = best_pkl.exists()
    success = (return_code == 0 and not timed_out) or soft_success
    return {
        "success": success,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_s": duration,
        "best_pkl": str(best_pkl) if best_pkl.exists() else None,
        "metrics": metrics,
        "stdout_tail": _tail(stdout_log),
        "stderr_tail": _tail(stderr_log),
    }