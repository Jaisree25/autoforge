"""Helpers the Trainer's generated train.py imports.

The TrainingAgent copies this file into each attempt's directory as
`autoforge_helpers.py`. The templated train.py imports from it so the
LLM-authored model.py only needs to worry about the architecture +
hyperparameters — the data loading, joblib dumping, and metric extraction
boilerplate lives here.

AutoForge narrowed to sklearn-tabular only; this module is CSV-only.

Self-contained — no imports from other AutoForge modules — so dropping it
into an attempt dir Just Works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


_ID_TOKENS = ("_id", "customer_id", "user_id", "id_", "uuid")


def _is_id_column(name: str) -> bool:
    """Name-based ID detection. Catches `id`, `customer_id`, `PassengerId`."""
    n = name.lower()
    if n == "id":
        return True
    if any(t in n for t in _ID_TOKENS):
        return True
    # CamelCase / no-underscore IDs: `PassengerId`, `UserId`, `RecordId`,
    # `TransactionId`. Require length > 3 to skip "id", "lid", "kid".
    if n.endswith("id") and len(n) > 3:
        return True
    return False


def _is_high_card_drop(df, col: str) -> bool:
    """High-cardinality columns AutoForge auto-drops: free-text /
    near-unique identifiers / huge categorical fields that would explode
    one-hot encoding."""
    try:
        n = len(df)
        if n == 0:
            return False
        nunique = df[col].nunique(dropna=True)
        # Near-unique: looks like an ID even if the name isn't suggestive.
        if nunique / n > 0.9:
            return True
        # Many unique object values → likely free text / high-cardinality cat.
        if df[col].dtype == object and nunique > 50:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def load_csv_split(data_dir, target_column):
    """Load `<data_dir>/train.csv` + `<data_dir>/test.csv` → 4-tuple.

    Drops ID-like columns and auto-encodes non-numeric features (one-hot)
    so any LLM-generated sklearn estimator gets clean numeric input.
    Non-numeric target labels are also encoded to int.

    Returns:
        X_train, y_train, X_test, y_test
    """
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Drop ID-like columns + high-cardinality / near-unique columns from
    # features. These leak (ID) or explode one-hot encoding (free text).
    drop_cols = [
        c for c in train_df.columns
        if c != target_column
        and (_is_id_column(c) or _is_high_card_drop(train_df, c))
    ]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    feature_cols = [c for c in train_df.columns if c != target_column]

    # Fill any remaining missing numeric values with column median so the
    # model fit doesn't blow up. Object columns get filled with the mode
    # before one-hot encoding handles them as a "missing" level.
    for c in feature_cols:
        if train_df[c].isna().any() or test_df[c].isna().any():
            if pd.api.types.is_numeric_dtype(train_df[c]):
                med = train_df[c].median()
                train_df[c] = train_df[c].fillna(med)
                test_df[c] = test_df[c].fillna(med)
            else:
                mode = train_df[c].mode()
                fill = mode.iloc[0] if len(mode) else "missing"
                train_df[c] = train_df[c].fillna(fill)
                test_df[c] = test_df[c].fillna(fill)

    # Auto-one-hot any leftover non-numeric feature columns. Use the union
    # of train + test categories to keep column counts aligned.
    obj_cols = [
        c for c in feature_cols
        if train_df[c].dtype == object or test_df[c].dtype == object
    ]
    if obj_cols:
        combined = pd.concat(
            [train_df.assign(__split="train"), test_df.assign(__split="test")],
            ignore_index=True,
        )
        combined = pd.get_dummies(combined, columns=obj_cols, drop_first=False)
        train_df = combined[combined["__split"] == "train"].drop(columns="__split")
        test_df = combined[combined["__split"] == "test"].drop(columns="__split")
        feature_cols = [c for c in train_df.columns if c != target_column]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()

    # Encode non-numeric target labels to int (consistent map across splits).
    if y_train.dtype == object or y_test.dtype == object:
        classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        y_train = np.array([cls_to_idx[v] for v in y_train], dtype=np.int64)
        y_test = np.array([cls_to_idx[v] for v in y_test], dtype=np.int64)

    return X_train, y_train, X_test, y_test


def _introspect_fitted(model) -> dict[str, Any]:
    info: dict[str, Any] = {"estimator_class": type(model).__name__}

    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        try:
            iters = (
                list(n_iter) if hasattr(n_iter, "__iter__") else [int(n_iter)]
            )
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

    def _safe(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return repr(v)

    try:
        info["effective_params"] = {
            k: _safe(v) for k, v in model.get_params(deep=False).items()
        }
    except Exception:
        pass

    return info


def save_outputs(
    model,
    output_dir,
    val_accuracy: float,
    train_seconds: float,
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, float]:
    """Save `best.pkl` + `metrics.json` into `output_dir`. Returns the
    headline dict the calling train.py should print as JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_process = _introspect_fitted(model)
    if n_train is not None:
        training_process["n_train"] = int(n_train)
    if n_test is not None:
        training_process["n_test"] = int(n_test)

    joblib.dump(model, output_dir / "best.pkl")
    metrics = {
        "val_accuracy": float(val_accuracy),
        "train_seconds": float(train_seconds),
        "training_process": training_process,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics))

    return {
        "val_accuracy": float(val_accuracy),
        "train_seconds": float(train_seconds),
    }
