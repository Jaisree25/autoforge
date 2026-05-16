"""
Agentic Trainer pipeline (fixed + Optuna-ready version)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import importlib
import inspect

from pydantic import BaseModel, ConfigDict, Field

from agents._llm_client import NemotronClient
from contracts.schemas import (
    DatasetProfile,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
)
from tools import training_tools as tt

# ============================================================
# Utilities
# ============================================================

def _is_regression_task(profile: DatasetProfile | None) -> bool:
    return bool(profile and profile.task_type.value == "regression")


# ============================================================
# Oracle baseline
# ============================================================

def run_oracle(profile: DatasetProfile,
               prep: PreparationReport | None,
               run_dir: Path) -> dict[str, Any]:

    t0 = time.time()

    target = profile.target_column or "target"
    prepared_dir = Path(prep.prepared_dataset_path) if prep else None

    X_train, y_train, X_test, y_test = tt.load_csv_split_or_full(
        prepared_dir=prepared_dir,
        fallback_csv=Path(profile.dataset_path),
        target_column=target,
    )

    if X_test is None:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    if _is_regression_task(profile):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = float(r2_score(y_test, model.predict(X_test)))

        name = "sklearn.linear_model.LinearRegression"
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        score = float(accuracy_score(y_test, model.predict(X_test)))

        name = "sklearn.linear_model.LogisticRegression"

    oracle = {
        "model": name,
        "score": score,
        "wall_s": time.time() - t0,
    }

    (run_dir / "oracle.json").write_text(json.dumps(oracle, indent=2))
    return oracle


# ============================================================
# Design prompt builder
# ============================================================

def _build_design_user_prompt(spec, profile, envelope, oracle, prep, prep_config):
    prepared_path = prep.prepared_dataset_path if prep else profile.dataset_path

    return f"""
## Inputs
- Task: {profile.task_type.value}
- Dataset: {prepared_path}
- Target: {profile.target_column}
- Samples: {profile.n_rows} x {profile.n_cols}

## Oracle score
{oracle}

## Instruction
Write sklearn design.
"""


# ============================================================
# Model selection schema
# ============================================================

class _ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sklearn_class: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# sklearn resolution
# ============================================================

_SKLEARN_MODULES = (
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.neural_network",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.tree",
)


def _resolve_sklearn_class(name: str):
    from sklearn.base import BaseEstimator

    for mod_name in _SKLEARN_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

        obj = getattr(mod, name, None)
        if isinstance(obj, type) and issubclass(obj, BaseEstimator):
            return mod_name, name

    obj = getattr(importlib.import_module("sklearn"), name, None)
    if isinstance(obj, type) and issubclass(obj, BaseEstimator):
        return "sklearn", name

    return None


# ============================================================
# model.py synthesis
# ============================================================
_BLOCKED_CLASSES = frozenset({"Pipeline", "FeatureUnion", "ColumnTransformer"})

def _synthesize_model_py(choice: _ModelChoice) -> str:
    original = choice.sklearn_class

    if original in _BLOCKED_CLASSES:
        raise ValueError(
            f"LLM chose '{original}' which requires composite construction. "
            "Only flat estimator classes are supported."
        )

    resolved = _resolve_sklearn_class(original)
    if resolved is None:
       raise ValueError(f"Unsupported sklearn class: {original}")

    module, cls = resolved
    cls_obj = getattr(importlib.import_module(module), cls)

    try:
        valid_params = set(inspect.signature(cls_obj).parameters)
    except Exception:
        valid_params = set()

    hp = {}
    for k, v in choice.hyperparameters.items():
        if not valid_params or k in valid_params:
            hp[k] = v

    if not valid_params or "random_state" in valid_params:
        hp.setdefault("random_state", 42)

    lines = [
        f"from {module} import {cls}",
        "",
        "def build_model(**overrides):",
        "    params = {",
    ]

    for k, v in hp.items():
        lines.append(f"        {k!r}: {repr(v)},")
    lines += [
        "    }",
        "    params.update(overrides)",
        f"    return {cls}(**params)",
    ]

    return "\n".join(lines)


# ============================================================
# LLM model generator
# ============================================================
# --- design stage (ADD THIS) ---
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
        "- DO NOT pick Pipeline, FeatureUnion, or ColumnTransformer — "
        "only flat estimator classes are supported.\n"
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


# ============================================================
# Smoke harness (FIXED)
# ============================================================

def run_smoke_harness(code_dir: Path) -> dict[str, Any]:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "passed": ok, "detail": detail})

    # syntax
    for f in ("model.py", "train.py"):
        p = code_dir / f
        if not p.exists():
            add(f"exists_{f}", False, "missing")
            continue

        r = subprocess.run(
            [sys.executable, "-m", "py_compile", str(p)],
            capture_output=True,
            text=True,
        )

        add(f"syntax_{f}", r.returncode == 0, r.stderr[:200])

    return {
        "overall_passed": all(c["passed"] for c in checks),
        "ok": all(c["passed"] for c in checks),
        "checks": checks,
    }

    # import test
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, r'{code_dir}');"
         "from model import build_model; m = build_model();"
         "print(hasattr(m,'fit') and hasattr(m,'predict'))"],
        capture_output=True,
        text=True,
    )

    add("sklearn_interface", "True" in r.stdout, r.stderr[:200])

    return {
        "ok": all(c["passed"] for c in checks),
        "checks": checks,
    }


# ============================================================
# Training subprocess
# ============================================================

def run_training_subprocess(code_dir: Path,
                            prepared_dir: Path,
                            output_dir: Path,
                            log_dir: Path,
                            max_time_seconds: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(code_dir / "train.py"),
        "--data-dir", str(prepared_dir),
        "--output-dir", str(output_dir),
        "--max-time-seconds", str(max_time_seconds),
    ]

    t0 = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max_time_seconds + 30,
            cwd=str(code_dir),
        )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        timed_out = True

    best = output_dir / "best.pkl"

    return {
        "success": best.exists() and rc == 0,
        "return_code": rc,
        "duration": time.time() - t0,
        "has_model": best.exists(),
        "timed_out": timed_out,
    }