"""Operations the Preparer can apply.

Two flavors of operations:

  - **Data-modifying** (return a path to the new prepared dataset):
      resize_images_dir, split_image_dir,
      impute_missing_csv, encode_categoricals_csv, split_train_test_csv

  - **Config-only** (return a config dict the Trainer applies at runtime):
      record_normalization, record_augmentation, record_feature_scaling

The Preparer agent's `_dispatch_op` looks at the operation name and calls
the matching function here. Operation names are stable identifiers the LLM
picks from the system-prompt allowlist.

For MNIST-style demos the typical chain is:
  split_image_dir → record_normalization → record_augmentation
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ===========================================================================
# Image operations
# ===========================================================================
def resize_images_dir(
    source_dir: Path,
    target_h: int,
    target_w: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Resize every image under `source_dir` → `output_dir`.

    Preserves class-folder structure (e.g. `source_dir/cat/*.png` →
    `output_dir/cat/*.png` resized).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in source_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        relative = img_path.relative_to(source_dir)
        dest = output_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as img:
            resized = img.resize((target_w, target_h), Image.LANCZOS)
            resized.save(dest, format=img.format or "PNG")
        count += 1
    return {
        "output_dir": str(output_dir),
        "resized_count": count,
        "target_h": target_h,
        "target_w": target_w,
    }


def split_image_dir(
    source_dir: Path,
    test_size: float,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Split a class-folder image dataset into `output_dir/{train,test}/<class>/`.

    Stratified per class — each class loses the same fraction to test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    train_count = 0
    test_count = 0

    for cls_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        images = sorted(
            p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        if not images:
            continue

        n_test = max(1, int(len(images) * test_size))
        indices = rng.permutation(len(images))
        test_indices = set(indices[:n_test].tolist())

        cls_train = train_dir / cls_dir.name
        cls_test = test_dir / cls_dir.name
        cls_train.mkdir(exist_ok=True)
        cls_test.mkdir(exist_ok=True)

        for i, img_path in enumerate(images):
            dest_dir = cls_test if i in test_indices else cls_train
            shutil.copy2(img_path, dest_dir / img_path.name)
            if i in test_indices:
                test_count += 1
            else:
                train_count += 1

    return {
        "output_dir": str(output_dir),
        "train_count": train_count,
        "test_count": test_count,
        "test_size": test_size,
    }


# ===========================================================================
# CSV operations
# ===========================================================================
def impute_missing_csv(
    source_path: Path,
    strategy: str,
    columns: list[str] | None,
    output_path: Path,
) -> dict[str, Any]:
    """Impute missing values. Strategies: 'median', 'mean', 'mode', 'drop'."""
    df = pd.read_csv(source_path)
    if not columns:
        # Default: every column with any missingness
        columns = df.columns[df.isna().any()].tolist()

    imputed: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif strategy == "drop":
            df = df.dropna(subset=[col])
        else:
            continue
        imputed.append(col)

    df.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "strategy": strategy,
        "imputed_columns": imputed,
        "rows_after": len(df),
    }


def encode_categoricals_csv(
    source_path: Path,
    method: str,
    columns: list[str],
    output_path: Path,
) -> dict[str, Any]:
    """One-hot or label-encode categorical columns."""
    df = pd.read_csv(source_path)
    encoded: list[str] = []
    if method == "onehot":
        valid_cols = [c for c in columns if c in df.columns]
        df = pd.get_dummies(df, columns=valid_cols, drop_first=False)
        encoded = valid_cols
    elif method == "label":
        for col in columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
                encoded.append(col)
    df.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "method": method,
        "encoded_columns": encoded,
        "cols_after": len(df.columns),
    }


def split_train_test_csv(
    source_path: Path,
    test_size: float,
    stratify_by: str | None,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Split CSV into `output_dir/train.csv` + `output_dir/test.csv`."""
    from sklearn.model_selection import train_test_split as sk_split

    df = pd.read_csv(source_path)
    stratify = None
    if stratify_by and stratify_by in df.columns:
        stratify = df[stratify_by]

    train_df, test_df = sk_split(
        df, test_size=test_size, stratify=stratify, random_state=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_count": len(train_df),
        "test_count": len(test_df),
    }


# ===========================================================================
# Config-only operations — return the config the Trainer applies at runtime
# ===========================================================================
def record_normalization(
    mean: list[float],
    std: list[float],
) -> dict[str, Any]:
    """Record normalization (mean / std per channel) for the Trainer."""
    return {"normalization": {"mean": list(mean), "std": list(std)}}


def record_augmentation(
    transforms: list[str],
) -> dict[str, Any]:
    """Record augmentation transforms for the Trainer. Each item is a short
    identifier the Trainer maps to a torchvision/albumentations transform.

    Supported identifiers:
      'rotation' / 'rotation:<degrees>'   — random rotation by ±degrees
      'translate' / 'translate:<frac>'    — random translate
      'hflip'                             — random horizontal flip
      'vflip'                             — random vertical flip
      'crop' / 'crop:<size>'              — random crop
      'brightness' / 'brightness:<frac>'  — random brightness jitter
      'normalize'                         — apply recorded normalization
    """
    return {"augmentation": {"transforms": list(transforms)}}


def record_feature_scaling(
    method: str,
    columns: list[str],
) -> dict[str, Any]:
    """Record CSV feature scaling for the Trainer.

    method ∈ {'standard', 'minmax', 'robust'}.
    columns: which columns to scale.
    """
    return {"feature_scaling": {"method": method, "columns": list(columns)}}
