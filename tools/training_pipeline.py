"""Agentic Trainer pipeline.

The TrainingAgent drives this in stages:

  1. Oracle baseline (sklearn LogReg) — the "must beat by 5 points" gate.
  2. **Generate design.md + model.py** — Nemotron writes both from the
     upstream Profiler/Researcher/Preparer context. AutoForge writes the
     templated train.py (modality-specific, reads prep_config.json at
     runtime to apply normalization or feature_scaling automatically).
  3. **Smoke harness** — verify generated code: py_compile, import,
     `build_model()` instantiates, `train.py --help` works. Failures feed
     back to the LLM as retry context.
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
    For classification: sklearn LogReg, scored by accuracy.
    For regression:     sklearn LinReg, scored by R².
    """
    t0 = time.time()
    is_regression = _is_regression_task(profile)
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

    if is_regression:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        model = LinearRegression()
        model.fit(X_train, y_train)
        acc = float(r2_score(y_test, model.predict(X_test)))
        oracle_model_name = "sklearn.linear_model.LinearRegression"
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, model.predict(X_test)))
        oracle_model_name = "sklearn.linear_model.LogisticRegression"
    wall = time.time() - t0

    oracle = {
        "model": oracle_model_name,
        "test_accuracy": acc,
        "wall_clock_s": wall,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]) if X_train.ndim > 1 else 1,
    }
    (run_dir / "oracle.json").write_text(json.dumps(oracle, indent=2))
    return oracle


# ===========================================================================
# Stage 2a: Generate design.md (LLM call 1, before HITL gate).
# Stage 2b: Generate model.py (LLM call 2, after HITL approves the design).
# Split into two calls so the human reviews the design BEFORE the code
# locks in. Each call is small (single artifact) so it's fast on the 9B.
# ===========================================================================
def _build_design_user_prompt(
    spec: StrategySpec,
    profile: DatasetProfile,
    envelope: TrainingEnvelope,
    oracle: dict[str, Any],
    prep: PreparationReport | None,
    prep_config: dict[str, Any] | None,
) -> str:
    arch = spec.candidate_architectures[0] if spec.candidate_architectures else None
    prepared_path = (
        prep.prepared_dataset_path if (prep and prep.prepared_dataset_path)
        else profile.dataset_path
    )
    lines = [
        "## Inputs",
        f"- Modality: tabular CSV (sklearn-only)",
        f"- Task: `{profile.task_type.value}`",
        f"- Target column: `{profile.target_column}`",
        f"- Prepared data dir: `{prepared_path}`",
        f"- Samples: {profile.n_rows:,} × {profile.n_cols} columns",
        f"- Wall-clock cap: {envelope.max_train_minutes * 60:.0f}s",
        "",
        "## Target",
        f"- `{spec.success_metric}` ≥ {spec.success_threshold:.3f}",
        "",
        "## Researcher's committed architecture (already picked at HITL)",
    ]
    if arch is not None:
        lines.append(f"- Name: `{arch.name}` ({arch.family} / `{arch.library}`)")
        lines.append(f"- Rationale: {arch.rationale}")
        lines.append(f"- Hyperparameter space: `{arch.hyperparameter_space}`")
    if prep is not None and prep.operations:
        lines.append("")
        lines.append("## Preparer's applied operations")
        for op in prep.operations:
            lines.append(f"- `{op}`")
    if prep_config:
        lines.append("")
        lines.append("## Preparer's prep_config (applied by train.py at runtime)")
        lines.append("```json")
        lines.append(json.dumps(prep_config, indent=2, default=str))
        lines.append("```")
    lines.append("")
    lines.append(
        "Write design.md now — Markdown prose, ~150-300 words total, "
        "all 7 required headers in order. Concrete values in Hyperparameters."
    )
    return "\n".join(lines)


def generate_design_md(
    llm: NemotronClient,
    spec: StrategySpec,
    profile: DatasetProfile,
    envelope: TrainingEnvelope,
    oracle: dict[str, Any],
    prep: PreparationReport | None,
    prep_config: dict[str, Any] | None = None,
    on_thinking=None,
    previous_feedback: dict[str, Any] | None = None,
) -> str:
    """LLM call 1 → design.md (Markdown). Fires before the HITL gate."""
    is_regression = _is_regression_task(profile)
    if is_regression:
        task_block = (
            "TASK TYPE: regression. Pick from sklearn regressors: "
            "LinearRegression, Ridge, Lasso, MLPRegressor, "
            "RandomForestRegressor, GradientBoostingRegressor, SVR, "
            "KNeighborsRegressor, DecisionTreeRegressor.\n"
            "- DO NOT use classifier classes. DO NOT use class_weight (regressors "
            "don't take it). DO NOT pick LightGBM/XGBoost/PyTorch.\n"
            "- For gradient boosting use `GradientBoostingRegressor`. For "
            "fast non-linear use `RandomForestRegressor`. For a strong "
            "linear baseline use `Ridge`."
        )
    else:
        task_block = (
            "TASK TYPE: classification. Pick from sklearn classifiers: "
            "MLPClassifier, LogisticRegression, RandomForestClassifier, "
            "GradientBoostingClassifier, SVC, KNeighborsClassifier, "
            "DecisionTreeClassifier.\n"
            "- DO NOT use regressor classes. DO NOT pick LightGBM/XGBoost/PyTorch.\n"
            "- **For binary classification: ALWAYS include "
            "`class_weight='balanced'`** as a hyperparameter (supported by "
            "LogisticRegression, SVC, RandomForestClassifier, "
            "DecisionTreeClassifier). Imbalanced datasets without class_weight "
            "collapse to predicting the majority class."
        )
    system = (
        "You are the Trainer. Write design.md for an sklearn model on a "
        "tabular CSV dataset. Output PLAIN MARKDOWN PROSE — not JSON, not code.\n\n"
        "Note: AutoForge will run an Optuna HP search (~10 trials) around your "
        "chosen architecture, so your hyperparameter picks are sensible "
        "*starting points*, not final values. Concentrate on picking the "
        "RIGHT sklearn class.\n\n"
        "Required sections (level-2 headers, in order, using these EXACT lines):\n"
        "  ## Architecture commitment\n"
        "  ## Hyperparameters (final)\n"
        "  ## Wall-clock budget\n"
        "  ## Success criteria\n"
        "  ## Risks & anti-patterns\n"
        "  ## Code structure\n"
        "  ## Verification plan\n\n"
        "Constraints:\n"
        f"- {task_block}\n"
        "- Hyperparameters: one bullet per HP with EXACT concrete value + "
        "one-line rationale. NO ranges, no `tbd`, no `~`.\n"
        "- Total length ~150-300 words.\n"
        "- DO NOT write JSON. DO NOT include code blocks for the architecture.\n"
        "- Architecture commitment names the concrete sklearn class.\n"
        "- Code structure section just says model.py exports `build_model()` "
        "and AutoForge handles train.py."
    )
    user = _build_design_user_prompt(
        spec=spec, profile=profile, envelope=envelope,
        oracle=oracle, prep=prep, prep_config=prep_config,
    )
    if previous_feedback:
        fail_mode = previous_feedback.get("failure_mode") or "unknown"
        gap = previous_feedback.get("accuracy_gap")
        suggestions = previous_feedback.get("suggestions") or []
        feedback_block = [
            "",
            "## PREVIOUS ATTEMPT FEEDBACK (Evaluator says try a different design)",
            f"- failure_mode: `{fail_mode}`",
        ]
        if gap is not None:
            feedback_block.append(f"- accuracy_gap from target: {gap}")
        if suggestions:
            feedback_block.append("- suggestions:")
            for s in suggestions:
                feedback_block.append(f"  - {s}")
        feedback_block.append(
            "\nAct on this feedback: pick a stronger model, increase capacity, "
            "or change hyperparameters to address the failure mode. DO NOT "
            "emit the same architecture + hyperparameters as before."
        )
        user = user + "\n" + "\n".join(feedback_block)
    md = llm.think_and_answer(
        system=system,
        user=user,
        on_thinking=on_thinking,
        max_tokens=2000,
        temperature=0.3,
        no_think=True,
    )
    # Strip code fences if the LLM wrapped its output
    md = md.strip()
    if md.startswith("```"):
        md = md.split("\n", 1)[1] if "\n" in md else md
        if md.endswith("```"):
            md = md.rsplit("```", 1)[0]
        md = md.strip()
    return md


_SKLEARN_CLASS_MAP: dict[str, str] = {
    # Classifiers (binary / multiclass)
    "MLPClassifier":               "sklearn.neural_network",
    "LogisticRegression":          "sklearn.linear_model",
    "RandomForestClassifier":      "sklearn.ensemble",
    "GradientBoostingClassifier":  "sklearn.ensemble",
    "SVC":                         "sklearn.svm",
    "KNeighborsClassifier":        "sklearn.neighbors",
    "DecisionTreeClassifier":      "sklearn.tree",
    # Regressors
    "LinearRegression":            "sklearn.linear_model",
    "Ridge":                       "sklearn.linear_model",
    "Lasso":                       "sklearn.linear_model",
    "MLPRegressor":                "sklearn.neural_network",
    "RandomForestRegressor":       "sklearn.ensemble",
    "GradientBoostingRegressor":   "sklearn.ensemble",
    "SVR":                         "sklearn.svm",
    "KNeighborsRegressor":         "sklearn.neighbors",
    "DecisionTreeRegressor":       "sklearn.tree",
}

_CLASSIFIER_NAMES = frozenset({
    "MLPClassifier", "LogisticRegression", "RandomForestClassifier",
    "GradientBoostingClassifier", "SVC", "KNeighborsClassifier",
    "DecisionTreeClassifier",
})

_REGRESSOR_NAMES = frozenset({
    "LinearRegression", "Ridge", "Lasso", "MLPRegressor",
    "RandomForestRegressor", "GradientBoostingRegressor", "SVR",
    "KNeighborsRegressor", "DecisionTreeRegressor",
})


def _is_regression_task(profile: DatasetProfile | None) -> bool:
    if profile is None:
        return False
    return profile.task_type.value == "regression"


class _ModelChoice(BaseModel):
    """LLM picks ONE sklearn class + its concrete hyperparameter dict."""
    model_config = ConfigDict(extra="forbid")
    sklearn_class: str = Field(
        description=(
            "Exact sklearn class name. Must be one of: "
            "MLPClassifier, LogisticRegression, RandomForestClassifier, "
            "GradientBoostingClassifier, SVC, KNeighborsClassifier, "
            "DecisionTreeClassifier."
        ),
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Concrete keyword arguments to pass to the class. Each value is "
            "a literal Python primitive: int, float, str, bool, null, or a "
            "list/tuple of those. Example for MLPClassifier: "
            '{"hidden_layer_sizes": [128, 64], "alpha": 0.001, '
            '"learning_rate_init": 0.001, "max_iter": 200, "random_state": 42}'
        ),
    )


def _hp_value_to_literal(value: Any) -> str:
    """Render a hyperparameter value as a Python literal for the source.

    Lists with all-numeric entries are emitted as tuples (sklearn uses
    tuples for `hidden_layer_sizes` etc.). None → None. Strings get repr().
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_hp_value_to_literal(v) for v in value) + (
            "," if len(value) == 1 else ""
        ) + ")"
    return repr(value)


# If the LLM picks a non-sklearn gradient-boosting class, coerce to
# sklearn's GradientBoostingClassifier. Keeps the pipeline alive.
_FALLBACK_CLASS_FOR_NON_SKLEARN: dict[str, str] = {
    # Classification → GradientBoostingClassifier
    "XGBClassifier":       "GradientBoostingClassifier",
    "XGBoostClassifier":   "GradientBoostingClassifier",
    "LGBMClassifier":      "GradientBoostingClassifier",
    "LightGBMClassifier":  "GradientBoostingClassifier",
    "CatBoostClassifier":  "GradientBoostingClassifier",
    # Regression → GradientBoostingRegressor
    "XGBRegressor":        "GradientBoostingRegressor",
    "XGBoostRegressor":    "GradientBoostingRegressor",
    "LGBMRegressor":       "GradientBoostingRegressor",
    "LightGBMRegressor":   "GradientBoostingRegressor",
    "CatBoostRegressor":   "GradientBoostingRegressor",
}


# Modules to scan when resolving an LLM-picked class name dynamically.
# Anything reachable via these modules counts as sklearn and is allowed.
_SKLEARN_MODULES_FOR_LOOKUP = (
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.neural_network",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.tree",
    "sklearn.naive_bayes",
    "sklearn.discriminant_analysis",
    "sklearn.gaussian_process",
)


def _resolve_sklearn_class(class_name: str) -> tuple[str, str] | None:
    """Find which sklearn submodule exports `class_name`, if any.

    Returns (module_path, class_name) on hit, None on miss.
    """
    import importlib
    for mod_path in _SKLEARN_MODULES_FOR_LOOKUP:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, class_name):
            cls = getattr(mod, class_name)
            # Sanity: must be a class
            if isinstance(cls, type):
                return mod_path, class_name
    return None


def _synthesize_model_py(choice: _ModelChoice) -> str:
    """Generate model.py source from the LLM's structured choice.

    AutoForge owns the file shape — the LLM only contributes the class
    name + hyperparameter dict. This makes the output reliable regardless
    of the LLM's prior about what "model.py" should look like.

    Defense in depth: non-sklearn class names (XGBoost / LightGBM / CatBoost)
    are coerced to a sklearn equivalent. Class lookup is dynamic — any
    class reachable via the standard sklearn modules works, not just a
    hardcoded allowlist. Hyperparameters are filtered to ones the target
    class actually accepts.
    """
    import importlib
    import inspect

    cls = choice.sklearn_class
    # Try dynamic sklearn lookup first.
    resolved = _resolve_sklearn_class(cls)
    if resolved is None and cls in _FALLBACK_CLASS_FOR_NON_SKLEARN:
        cls = _FALLBACK_CLASS_FOR_NON_SKLEARN[cls]
        resolved = _resolve_sklearn_class(cls)
    if resolved is None:
        # Last-resort hardcoded map (legacy paths).
        legacy_module = _SKLEARN_CLASS_MAP.get(cls)
        if legacy_module is not None:
            resolved = (legacy_module, cls)
    if resolved is None:
        raise ValueError(
            f"LLM picked unsupported class `{choice.sklearn_class}`. "
            f"Tried dynamic lookup across "
            f"{sorted(_SKLEARN_MODULES_FOR_LOOKUP)!r} — not found."
        )
    module, cls = resolved

    # Filter HPs to ones the actual class accepts. Drops xgboost-only
    # parameters like `gamma` that would crash sklearn.
    cls_obj = getattr(importlib.import_module(module), cls)
    try:
        accepted = set(inspect.signature(cls_obj).parameters.keys())
    except (TypeError, ValueError):
        accepted = set()
    raw_hp = dict(choice.hyperparameters or {})
    hp: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in raw_hp.items():
        if not accepted or k in accepted:
            hp[k] = v
        else:
            dropped.append(k)
    if "random_state" not in hp and (not accepted or "random_state" in accepted):
        hp["random_state"] = 42

    lines = [
        f"from {module} import {cls}",
        "",
        "",
        "def build_model(**overrides):",
        f"    params = {{",
    ]
    for k, v in hp.items():
        lines.append(f"        {k!r}: {_hp_value_to_literal(v)},")
    lines.append("    }")
    lines.append("    params.update(overrides)")
    lines.append(f"    return {cls}(**params)")
    lines.append("")
    if dropped or cls != choice.sklearn_class:
        notes = []
        if cls != choice.sklearn_class:
            notes.append(
                f"coerced `{choice.sklearn_class}` → `{cls}` (sklearn-only)"
            )
        if dropped:
            notes.append(f"dropped non-{cls} kwargs: {sorted(dropped)!r}")
        lines.insert(0, "# AutoForge note: " + "; ".join(notes))
    return "\n".join(lines)


def generate_model_py(
    llm: NemotronClient,
    design_md: str,
    spec: StrategySpec,
    profile: DatasetProfile,
    on_thinking=None,
) -> str:
    """LLM call 2 → model.py source code. AutoForge synthesizes the source
    from the LLM's structured (sklearn_class, hyperparameters) choice."""
    is_regression = _is_regression_task(profile)
    if is_regression:
        class_block = (
            "REGRESSION task: pick from LinearRegression, Ridge, Lasso, "
            "MLPRegressor, RandomForestRegressor, GradientBoostingRegressor, "
            "SVR, KNeighborsRegressor, DecisionTreeRegressor."
        )
    else:
        class_block = (
            "CLASSIFICATION task: pick from MLPClassifier, LogisticRegression, "
            "RandomForestClassifier, GradientBoostingClassifier, SVC, "
            "KNeighborsClassifier, DecisionTreeClassifier."
        )
    system = (
        "You pick the sklearn estimator class and its hyperparameters for "
        "model.py. You do NOT write Python code — you emit a tiny JSON "
        "object that AutoForge then formats into model.py source.\n\n"
        "Required output schema:\n"
        '  {"sklearn_class": "<class name>", "hyperparameters": {...}}\n\n'
        "Rules:\n"
        f"- {class_block}\n"
        "- DO NOT pick LightGBM, XGBoost, CatBoost, or any non-sklearn class. "
        "If gradient boosting is in the design, use the sklearn equivalent "
        "(GradientBoostingClassifier or GradientBoostingRegressor).\n"
        "- `hyperparameters` is a flat JSON object. Keys are sklearn constructor "
        "argument names. Values are literal numbers, strings, booleans, null, "
        "or lists/tuples of those.\n"
        "- Values MUST match the approved design.md exactly. If design.md says "
        "`alpha = 1e-3` you write `\"alpha\": 0.001`.\n"
        "- Include `random_state` only if you want a specific value; AutoForge "
        "defaults it to 42 when omitted."
    )
    user = (
        "## Approved design.md (use these hyperparameters EXACTLY):\n\n"
        + design_md
        + "\n\n"
        "Now emit the JSON object with `sklearn_class` and `hyperparameters`. "
        "No prose, no explanation — just the JSON."
    )
    choice: _ModelChoice = llm.think_and_answer_structured(
        system=system,
        user=user,
        schema=_ModelChoice,
        on_thinking=on_thinking,
        max_tokens=1500,
        temperature=0.1,
        no_think=True,
    )
    return _synthesize_model_py(choice)


# (legacy combined function removed — Trainer now calls generate_design_md
# and generate_model_py separately so the HITL gate fires between them.)
def _legacy_generate_design_and_code(*_args, **_kwargs):  # noqa: ARG001
    raise NotImplementedError(
        "generate_design_and_code was split into generate_design_md + "
        "generate_model_py — call those instead."
    )


# Stage 3: Smoke harness
# ===========================================================================
def run_smoke_harness(code_dir: Path) -> dict[str, Any]:
    """Verify generated code is at least syntactically + structurally sane.

    Returns a dict with `overall_passed: bool` and per-check details. ERROR-
    prefixed lines for grep. Mirrors agentic-pipeline's verify_report.json.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    # Check 1: py_compile each .py
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

    # Check 1b: model.py must be sklearn-only. Fail fast on torch/tensorflow
    # imports so the retry gets a clear, actionable error message.
    model_py = code_dir / "model.py"
    if model_py.exists():
        model_src = model_py.read_text(encoding="utf-8")
        forbidden = [
            ("import torch", "torch"),
            ("from torch", "torch"),
            ("import tensorflow", "tensorflow"),
            ("from tensorflow", "tensorflow"),
            ("import keras", "keras"),
            ("from keras", "keras"),
            ("import jax", "jax"),
            ("from jax", "jax"),
            ("import lightgbm", "lightgbm"),
            ("from lightgbm", "lightgbm"),
            ("import xgboost", "xgboost"),
            ("from xgboost", "xgboost"),
            ("import catboost", "catboost"),
            ("from catboost", "catboost"),
        ]
        hits = [name for token, name in forbidden if token in model_src]
        if hits:
            add(
                "sklearn_only", False,
                f"ERROR: model.py imports {sorted(set(hits))!r} — AutoForge "
                f"is sklearn-only. Use sklearn.neural_network.MLPClassifier, "
                f"sklearn.linear_model.LogisticRegression, "
                f"sklearn.ensemble.RandomForestClassifier, or another "
                f"sklearn estimator from sklearn.*.",
            )
        elif not model_src.strip():
            add(
                "sklearn_only", False,
                "ERROR: model.py is empty — the LLM returned no code. "
                "(Likely a streaming hiccup or stripped fence. Re-run.)",
            )
        else:
            add("sklearn_only", True)

    # If syntax failed, don't bother trying to import
    syntax_ok = all(c["passed"] for c in checks if c["name"].startswith("syntax_"))
    if not syntax_ok:
        return {
            "overall_passed": False,
            "checks": checks,
            "reason": "ERROR: syntax check(s) failed; skipping import + build",
        }

    # Check 2: import + has build_model
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

    # Check 3: train.py --help. Runs train.py's imports + argparse setup
    # without doing any real training. Catches "ImportError: torch", missing
    # required argparse args, NameError at module scope, etc.
    train_help = subprocess.run(
        [sys.executable, str(code_dir / "train.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(code_dir),
    )
    if train_help.returncode != 0:
        err = train_help.stderr.strip()[:500] or train_help.stdout.strip()[:500]
        add("train_py_help", False, f"ERROR: train.py --help failed: {err}")
    else:
        # Verify the three required flags actually appear in usage
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

    # Check 4: design.md is real Markdown with ≥5 level-2 headers, not a
    # JSON dump of the input prompt context. We don't require exact header
    # text (the LLM uses semantic variants — "Model Selection" instead of
    # "Architecture commitment" etc.) — just sanity-check structure.
    design_path = code_dir / "design.md"
    if design_path.exists():
        design_text = design_path.read_text(encoding="utf-8").strip()
        starts_like_json = design_text.startswith("{") or design_text.startswith("[")
        # Accept ## (level-2) or ### (level-3) headers — the LLM sometimes
        # uses the deeper level when retrying with feedback.
        h2_headers = [
            ln for ln in design_text.splitlines()
            if ln.lstrip().startswith("## ") or ln.lstrip().startswith("### ")
        ]
        if starts_like_json:
            add(
                "design_md_format", False,
                "ERROR: design.md starts with a JSON brace — must be plain "
                "Markdown prose. Re-read the system prompt's required "
                "section structure and write English text, not JSON.",
            )
        elif len(h2_headers) < 5:
            add(
                "design_md_format", False,
                f"ERROR: design.md has only {len(h2_headers)} level-2 "
                f"headers — must have at least 5 sections covering "
                f"architecture, hyperparameters, budget, success criteria, "
                f"and risks.",
            )
        else:
            add(
                "design_md_format", True,
                f"Markdown structure OK ({len(h2_headers)} sections)",
            )

    overall = all(c["passed"] for c in checks)
    return {
        "overall_passed": overall,
        "checks": checks,
        "reason": "" if overall else "ERROR: one or more checks failed",
    }


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

    # Last 50 lines of each log for the agent to see
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
