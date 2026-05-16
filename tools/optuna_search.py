"""Sklearn-class-aware Optuna search spaces + runner.

The Trainer's templated train.py imports `run_optuna_search` from here. The
search spaces are tuned to AutoForge's sklearn-only constraint: per-class
ranges that are wide enough to find a good fit but narrow enough that 10-15
trials cover meaningful ground.

The agent keeps autonomy over the **architecture** (which sklearn class to
pick); AutoForge handles the **hyperparameter search**. The architecture
choice is the decision-rich part; HP grids are mechanical.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna


# Per-class search definitions. Each entry maps an HP name → spec dict that
# Optuna's trial.suggest_* can interpret.
_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "MLPClassifier": {
        "hidden_layer_sizes": {"choices": [(64,), (128,), (64, 32), (128, 64)]},
        "alpha": {"low": 1e-5, "high": 1e-1, "log": True},
        "learning_rate_init": {"low": 1e-4, "high": 1e-2, "log": True},
        "max_iter": {"choices": [200, 300, 500]},
    },
    "MLPRegressor": {
        "hidden_layer_sizes": {"choices": [(64,), (128,), (64, 32), (128, 64)]},
        "alpha": {"low": 1e-5, "high": 1e-1, "log": True},
        "learning_rate_init": {"low": 1e-4, "high": 1e-2, "log": True},
        "max_iter": {"choices": [200, 300, 500]},
    },
    "LogisticRegression": {
        "C": {"low": 1e-3, "high": 1e2, "log": True},
        "class_weight": {"choices": [None, "balanced"]},
        "max_iter": {"choices": [500, 1000]},
    },
    "LinearRegression": {
        # LinearRegression has no real HPs to tune; we skip search.
    },
    "Ridge": {
        "alpha": {"low": 1e-3, "high": 1e2, "log": True},
    },
    "Lasso": {
        "alpha": {"low": 1e-4, "high": 1e1, "log": True},
        "max_iter": {"choices": [1000, 5000]},
    },
    "RandomForestClassifier": {
        "n_estimators": {"choices": [100, 200, 300]},
        "max_depth": {"choices": [None, 5, 10, 20]},
        "min_samples_split": {"low": 2, "high": 10},
        "class_weight": {"choices": [None, "balanced"]},
    },
    "RandomForestRegressor": {
        "n_estimators": {"choices": [100, 200, 300]},
        "max_depth": {"choices": [None, 5, 10, 20]},
        "min_samples_split": {"low": 2, "high": 10},
    },
    "GradientBoostingClassifier": {
        "n_estimators": {"choices": [100, 200, 300]},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"choices": [3, 4, 5, 7]},
    },
    "GradientBoostingRegressor": {
        "n_estimators": {"choices": [100, 200, 300]},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"choices": [3, 4, 5, 7]},
    },
    "HistGradientBoostingClassifier": {
        "max_iter": {"choices": [100, 200, 300]},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"choices": [None, 5, 10]},
    },
    "HistGradientBoostingRegressor": {
        "max_iter": {"choices": [100, 200, 300]},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"choices": [None, 5, 10]},
    },
    "SVC": {
        "C": {"low": 1e-2, "high": 1e2, "log": True},
        "kernel": {"choices": ["rbf", "linear"]},
        "class_weight": {"choices": [None, "balanced"]},
    },
    "SVR": {
        "C": {"low": 1e-2, "high": 1e2, "log": True},
        "kernel": {"choices": ["rbf", "linear"]},
    },
    "LinearSVC": {
        "C": {"low": 1e-2, "high": 1e2, "log": True},
        "class_weight": {"choices": [None, "balanced"]},
        "max_iter": {"choices": [1000, 5000]},
    },
    "KNeighborsClassifier": {
        "n_neighbors": {"low": 3, "high": 25},
        "weights": {"choices": ["uniform", "distance"]},
    },
    "KNeighborsRegressor": {
        "n_neighbors": {"low": 3, "high": 25},
        "weights": {"choices": ["uniform", "distance"]},
    },
    "DecisionTreeClassifier": {
        "max_depth": {"choices": [None, 5, 10, 20]},
        "min_samples_split": {"low": 2, "high": 10},
        "class_weight": {"choices": [None, "balanced"]},
    },
    "DecisionTreeRegressor": {
        "max_depth": {"choices": [None, 5, 10, 20]},
        "min_samples_split": {"low": 2, "high": 10},
    },
}


def get_search_space(sklearn_class: str) -> dict[str, dict[str, Any]]:
    """Return the predefined Optuna search space for an sklearn class, or
    an empty dict if the class isn't recognized (no search → just fit
    the base build_model() once)."""
    return _SEARCH_SPACES.get(sklearn_class, {})


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Translate a search-space entry into an Optuna trial.suggest_* call."""
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    if "low" in spec and "high" in spec:
        if isinstance(spec["low"], int) and isinstance(spec["high"], int):
            return trial.suggest_int(name, spec["low"], spec["high"])
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=bool(spec.get("log", False)),
        )
    raise ValueError(f"Unrecognized search-space spec for {name!r}: {spec!r}")


def run_optuna_search(
    build_model_fn: Callable[..., Any],
    sklearn_class: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 10,
    direction: str = "maximize",
    timeout: int | None = None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    search_space = get_search_space(sklearn_class)
    if not search_space:
        model = build_model_fn()
        model.fit(X_train, y_train)
        score = float(model.score(X_val, y_val))
        return model, {}, [{"params": {}, "score": score, "state": "complete"}]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {n: _suggest(trial, n, spec) for n, spec in search_space.items()}
        try:
            model = build_model_fn(**params)
            model.fit(X_train, y_train)
            return float(model.score(X_val, y_val))
        except Exception:
            return -1e9 if direction == "maximize" else 1e9

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_estimator = build_model_fn(**best_params)
    best_estimator.fit(X_train, y_train)

    trials = [
        {
            "params": dict(t.params),
            "score": float(t.value) if t.value is not None else None,
            "state": t.state.name.lower(),
        }
        for t in study.trials
    ]
    return best_estimator, best_params, trials