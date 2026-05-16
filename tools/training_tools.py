"""Helpers used by the Trainer / Evaluator / Optimizer agents.

Kept deliberately small and sklearn-centric so the demo runs in ~30s on CPU.
No PyTorch dependency — MNIST + tabular both fit sklearn well (MLPClassifier
hits ~96% on MNIST in <10s). When the agentic-pipeline pattern lands, this
file becomes the "smoke harness" the Trainer's generated PyTorch code is
checked against.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ===========================================================================
# Dataset loading
# ===========================================================================
def load_image_folder(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a class-folder image dataset → (X, y, class_names).

    X: (n_samples, n_pixels) float32, normalized to [0, 1].
    y: (n_samples,) int64 class indices.
    class_names: sorted unique class names.
    """
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls in classes:
        cls_dir = root / cls
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            with Image.open(img_path) as img:
                img = img.convert("L")  # grayscale (works for MNIST + general)
                arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
            images.append(arr)
            labels.append(cls_to_idx[cls])

    if not images:
        raise ValueError(f"No images found under {root}")

    return np.stack(images), np.array(labels, dtype=np.int64), classes


_ID_TOKENS = ("_id", "customer_id", "user_id", "id_", "uuid")


def _is_id_column(name: str) -> bool:
    """Name-based ID detection. Catches `id`, `customer_id`, `PassengerId`."""
    n = name.lower()
    if n == "id":
        return True
    if any(t in n for t in _ID_TOKENS):
        return True
    if n.endswith("id") and len(n) > 3:
        return True
    return False


def _is_high_card_drop(df, col: str) -> bool:
    """High-cardinality columns AutoForge auto-drops."""
    try:
        n = len(df)
        if n == 0:
            return False
        nunique = df[col].nunique(dropna=True)
        if nunique / n > 0.9:
            return True
        if df[col].dtype == object and nunique > 50:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def load_csv_split_or_full(
    prepared_dir: Path | None,
    fallback_csv: Path,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load (X_train, y_train, X_test, y_test) — test arrays may be None.

    Drops ID-like columns, auto-one-hots leftover non-numeric features, and
    encodes string target labels to int — same logic as the Trainer's
    `autoforge_helpers.load_csv_split`, so the Oracle baseline and the
    LLM-generated trainer see identical numeric input.
    """
    import pandas as pd

    train_csv = prepared_dir / "train.csv" if prepared_dir else None
    test_csv = prepared_dir / "test.csv" if prepared_dir else None

    if train_csv and train_csv.exists() and test_csv and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    else:
        df = pd.read_csv(fallback_csv)
        train_df, test_df = df, None

    # Drop ID-like + high-cardinality / near-unique columns from features.
    drop_cols = [
        c for c in train_df.columns
        if c != target_column
        and (_is_id_column(c) or _is_high_card_drop(train_df, c))
    ]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        if test_df is not None:
            test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # Auto-one-hot non-numeric feature columns (consistent across splits).
    feature_cols = [c for c in train_df.columns if c != target_column]

    # Fill remaining missing values before to_numpy().
    for c in feature_cols:
        train_missing = train_df[c].isna().any()
        test_missing = test_df is not None and test_df[c].isna().any()
        if train_missing or test_missing:
            if pd.api.types.is_numeric_dtype(train_df[c]):
                med = train_df[c].median()
                train_df[c] = train_df[c].fillna(med)
                if test_df is not None:
                    test_df[c] = test_df[c].fillna(med)
            else:
                mode = train_df[c].mode()
                fill = mode.iloc[0] if len(mode) else "missing"
                train_df[c] = train_df[c].fillna(fill)
                if test_df is not None:
                    test_df[c] = test_df[c].fillna(fill)
    obj_cols = [c for c in feature_cols if train_df[c].dtype == object or (
        test_df is not None and test_df[c].dtype == object
    )]
    if obj_cols:
        if test_df is not None:
            combined = pd.concat(
                [train_df.assign(__split="train"), test_df.assign(__split="test")],
                ignore_index=True,
            )
            combined = pd.get_dummies(combined, columns=obj_cols, drop_first=False)
            train_df = combined[combined["__split"] == "train"].drop(columns="__split")
            test_df = combined[combined["__split"] == "test"].drop(columns="__split")
        else:
            train_df = pd.get_dummies(train_df, columns=obj_cols, drop_first=False)
        feature_cols = [c for c in train_df.columns if c != target_column]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_column].to_numpy()

    if test_df is not None:
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df[target_column].to_numpy()
    else:
        X_test, y_test = None, None

    # Encode non-numeric target labels (consistent map across splits).
    if y_train.dtype == object or (y_test is not None and y_test.dtype == object):
        classes_set = set(y_train.tolist())
        if y_test is not None:
            classes_set |= set(y_test.tolist())
        classes = sorted(classes_set)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        y_train = np.array([cls_to_idx[v] for v in y_train], dtype=np.int64)
        if y_test is not None:
            y_test = np.array([cls_to_idx[v] for v in y_test], dtype=np.int64)

    return X_train, y_train, X_test, y_test


# ===========================================================================
# Model selection
# ===========================================================================
def select_classifier_class(library: str, arch_name: str):
    """Map (library, architecture name) → sklearn classifier class.

    Best-effort routing. Always returns a callable; falls back to
    `MLPClassifier` if nothing else fits.
    """
    lib = (library or "").lower()
    name = (arch_name or "").lower()

    # Explicit library names
    if "xgboost" in lib:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier
        except ImportError:
            pass
    if "lightgbm" in lib:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier
        except ImportError:
            pass

    # By architecture name keywords
    if any(k in name for k in ["mlp", "neural", "mlp_classifier", "lenet", "cnn"]):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier
    if any(k in name for k in ["random_forest", "rf"]):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier
    if any(k in name for k in ["logistic", "logreg"]):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    if "svm" in name or "svc" in name:
        from sklearn.svm import SVC
        return SVC

    # Default — MLP works for both image and tabular
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier


# ===========================================================================
# Optuna-driven HPO
# ===========================================================================
def run_hpo(
    classifier_class,
    hyperparam_space: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 5,
    on_trial_complete=None,
) -> dict[str, Any]:
    """Run Optuna with `n_trials`. Returns dict with best params/score/all trials."""
    import optuna
    from sklearn.metrics import accuracy_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for key, values in hyperparam_space.items():
            if isinstance(values, list) and values:
                # Categorical sample
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(values, dict) and "low" in values and "high" in values:
                # Range
                if isinstance(values["low"], int) and isinstance(values["high"], int):
                    params[key] = trial.suggest_int(key, values["low"], values["high"])
                else:
                    params[key] = trial.suggest_float(key, values["low"], values["high"])
            else:
                # Treat scalar as fixed
                params[key] = values

        params = _coerce_sklearn_params(classifier_class, params)

        t0 = time.time()
        try:
            model = classifier_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = float(accuracy_score(y_val, preds))
        except Exception as exc:  # noqa: BLE001
            score = 0.0
            params["_error"] = f"{type(exc).__name__}: {exc}"
        duration = time.time() - t0

        if on_trial_complete is not None:
            on_trial_complete(trial.number, score, params, duration)

        # Store as user attrs (Optuna keeps these)
        trial.set_user_attr("duration_s", duration)
        trial.set_user_attr("params_used", params)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_data: list[dict[str, Any]] = []
    for t in study.trials:
        trials_data.append({
            "trial_id": t.number,
            "score": t.value if t.value is not None else 0.0,
            "params": t.user_attrs.get("params_used", dict(t.params)),
            "duration_s": float(t.user_attrs.get("duration_s", 0.0)),
            "status": "completed" if t.state.name == "COMPLETE" else t.state.name.lower(),
        })

    return {
        "best_score": study.best_value if study.best_trial else 0.0,
        "best_params": study.best_params if study.best_trial else {},
        "n_trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "total_trials": len(study.trials),
        "trials": trials_data,
    }


def _coerce_sklearn_params(cls, params: dict[str, Any]) -> dict[str, Any]:
    """Convert hyperparameter values the LLM might have picked into the
    shapes sklearn expects (e.g. hidden_layer_sizes as a tuple)."""
    name = cls.__name__
    out = dict(params)
    if name == "MLPClassifier":
        # MLPClassifier expects hidden_layer_sizes as tuple
        if "hidden_layer_sizes" in out and isinstance(out["hidden_layer_sizes"], list):
            out["hidden_layer_sizes"] = tuple(out["hidden_layer_sizes"])
        # Sensible defaults for fast demo
        out.setdefault("max_iter", 30)
        out.setdefault("random_state", 42)
    if name in {"RandomForestClassifier", "XGBClassifier"}:
        out.setdefault("random_state", 42)
        if name == "RandomForestClassifier":
            out.setdefault("n_jobs", -1)
    return out


# ===========================================================================
# Final-model training (using best params)
# ===========================================================================
def fit_final_model(
    classifier_class,
    best_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit a fresh model with the winning hyperparameters on full train set."""
    params = _coerce_sklearn_params(classifier_class, best_params)
    model = classifier_class(**params)
    model.fit(X_train, y_train)
    return model


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = "accuracy",
) -> dict[str, Any]:
    """Compute headline metric + latency stats + throughput on test set.

    Returns a dict that can be plugged straight into BenchmarkReport.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    # --- latency: time 100 single-sample predictions ---
    n_latency = min(100, len(X_test))
    latencies = []
    for i in range(n_latency):
        t0 = time.perf_counter()
        _ = model.predict(X_test[i : i + 1])
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    # --- bulk predictions for throughput + accuracy ---
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    bulk_time = max(time.perf_counter() - t0, 1e-9)
    throughput_qps = float(len(X_test) / bulk_time)

    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "macro"

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
    prec = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_test, y_pred, average=avg, zero_division=0))

    auc: float | None = None
    if n_classes == 2 and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
        except Exception:  # noqa: BLE001
            auc = None

    headline: float
    if metric == "f1":
        headline = f1
    elif metric == "auc" and auc is not None:
        headline = auc
    elif metric == "precision":
        headline = prec
    elif metric == "recall":
        headline = rec
    else:
        headline = acc

    return {
        "headline_metric": metric,
        "headline_value": headline,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "throughput_qps": throughput_qps,
        "n_test_samples": int(len(X_test)),
    }


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = "r2",
) -> dict[str, Any]:
    """Same shape as evaluate_classifier but for regression metrics.

    Returns RMSE / MAE / R². `headline_metric` honors the requested metric;
    if it's not a regression metric, falls back to R².
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    n_latency = min(100, len(X_test))
    latencies = []
    for i in range(n_latency):
        t0 = time.perf_counter()
        _ = model.predict(X_test[i : i + 1])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    bulk_time = max(time.perf_counter() - t0, 1e-9)
    throughput_qps = float(len(X_test) / bulk_time)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    m = (metric or "r2").lower()
    if m == "rmse":
        headline = rmse
    elif m in ("mae", "mean_absolute_error"):
        headline = mae
    elif m == "mse":
        headline = mse
    else:
        m = "r2"
        headline = r2

    # `accuracy` field reused as the "headline_in_accuracy_position" so
    # BenchmarkReport.accuracy_value gets the headline regardless of task.
    return {
        "headline_metric": m,
        "headline_value": headline,
        "accuracy": r2,    # R² is the closest to "accuracy" for regression
        "f1": rmse,        # repurposed: surface RMSE in the f1 slot
        "precision": mae,  # repurposed: surface MAE in the precision slot
        "recall": mse,     # repurposed: surface MSE in the recall slot
        "auc": None,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "throughput_qps": throughput_qps,
        "n_test_samples": int(len(X_test)),
    }


# ===========================================================================
# Serialization
# ===========================================================================
def save_model(model, path: Path, compress: int = 3) -> dict[str, Any]:
    """Pickle a sklearn-style model with joblib. Returns size info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
    size_mb = path.stat().st_size / 1024.0 / 1024.0
    return {"path": str(path), "size_mb": size_mb, "compress": compress}


def load_model(path: Path):
    return joblib.load(path)
