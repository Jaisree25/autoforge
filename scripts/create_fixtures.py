"""Create deterministic test datasets for Profiler / Preparer / Trainer.

  data/fixtures/churn_sample.csv          — 1000-row synthetic binary churn
  data/fixtures/sample_images/<class>/*.png — 2-class image dataset (6 imgs)

Both are small and committed so the demo + tests work offline. Run once:

    python scripts/create_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from rich.console import Console

from config import PROJECT_ROOT, configure_logging

configure_logging()
console = Console()

FIXTURES = PROJECT_ROOT / "data" / "fixtures"


# ---------------------------------------------------------------------------
# 1. Synthetic churn-like CSV
# ---------------------------------------------------------------------------
def make_churn_csv() -> Path:
    rng = np.random.default_rng(42)
    n = 1000
    age = rng.integers(18, 80, size=n)
    tenure = rng.integers(0, 72, size=n)
    monthly_charges = rng.normal(70, 20, size=n).clip(20, 200).round(2)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n, p=[0.55, 0.25, 0.20]
    )
    num_support_calls = rng.poisson(2, size=n)
    has_internet = rng.random(n) < 0.85

    # Inject some missingness in monthly_charges
    miss = rng.random(n) < 0.04
    monthly_charges_with_nans = monthly_charges.astype(object)
    monthly_charges_with_nans[miss] = np.nan

    # Churn correlated with high support calls + month-to-month + low tenure
    churn_logit = (
        -2.0
        + 0.4 * num_support_calls
        + 1.2 * (contract_type == "month-to-month").astype(int)
        - 0.04 * tenure
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges_with_nans,
        "contract_type": contract_type,
        "num_support_calls": num_support_calls,
        "has_internet": has_internet,
        "churn": churn,
    })

    out = FIXTURES / "churn_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# 2. Two-class image dataset (cat/dog) — generated, not real photos
# ---------------------------------------------------------------------------
def make_image_dataset() -> Path:
    """Generate 3 images per class in solid colors so PIL can profile shape,
    format, channels. Not actually trainable — just enough for Profiler smoke."""
    rng = np.random.default_rng(7)
    root = FIXTURES / "sample_images"

    for klass, base_color in [("cat", (200, 80, 80)), ("dog", (80, 80, 200))]:
        cls_dir = root / klass
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            # Slight variation in resolution + noise per image
            w = 224 + (i * 16)
            h = 224 + ((i + 1) * 8)
            arr = np.full((h, w, 3), base_color, dtype=np.uint8)
            # Add noise so they're not pure solid blocks
            noise = rng.integers(-30, 30, size=arr.shape, dtype=np.int16)
            arr = (arr.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"{klass}_{i:02d}.png", format="PNG")

    return root


# ---------------------------------------------------------------------------
def main() -> None:
    csv_path = make_churn_csv()
    img_root = make_image_dataset()
    console.print(f"[green]CSV[/]     {csv_path.relative_to(PROJECT_ROOT)}")
    console.print(f"[green]Images[/]  {img_root.relative_to(PROJECT_ROOT)}")
    # Quick stats
    df = pd.read_csv(csv_path)
    console.print(f"  → {len(df)} rows × {len(df.columns)} cols, "
                  f"churn rate {df['churn'].mean():.2%}")
    img_files = list(img_root.rglob("*.png"))
    console.print(f"  → {len(img_files)} images across {len(list(img_root.iterdir()))} classes")


if __name__ == "__main__":
    main()
