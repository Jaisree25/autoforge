"""Build a small MNIST fixture for the demo.

Pulls MNIST from OpenML via sklearn (cached after first run; ~12MB
download), subsamples N images per class, and writes them out as PNGs
under `data/fixtures/mnist/<class>/<class>_<idx>.png`.

We use a subsample so:
  - the Profiler runs in reasonable time
  - the dashboard chat feed stays readable
  - the Trainer (when real) finishes in <2 min on CPU

500 images (50 per class × 10 classes) is the sweet spot.

    python scripts/create_mnist_fixture.py
    python scripts/create_mnist_fixture.py --n-per-class 100   # bigger sample
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import typer
from PIL import Image
from rich.console import Console

from config import PROJECT_ROOT, configure_logging

configure_logging()
console = Console()

OUTPUT_DIR = PROJECT_ROOT / "data" / "fixtures" / "mnist"


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    n_per_class: int = typer.Option(
        50, "--n-per-class", "-n",
        help="Number of images to sample per class (10 classes total).",
    ),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    console.rule("[bold cyan]MNIST fixture")
    console.print(f"target: [cyan]{OUTPUT_DIR.relative_to(PROJECT_ROOT)}[/]")
    console.print(f"per class: {n_per_class}  |  total: {n_per_class * 10}")
    console.print()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.png")):
        existing = len(list(OUTPUT_DIR.rglob("*.png")))
        console.print(
            f"[yellow]Fixture already exists ({existing} images).[/] "
            "Overwrite? (Ctrl-C to cancel; pressing Enter to continue)"
        )
        # In non-interactive mode (subprocess), just proceed.
        try:
            input("Enter to overwrite, Ctrl-C to abort: ")
        except (EOFError, KeyboardInterrupt):
            console.print("[red]Aborted.[/]")
            return

    console.print("Fetching MNIST from OpenML (cached after first download)…")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        parser="auto",
        as_frame=False,
        cache=True,
    )
    X, y = mnist.data, mnist.target  # X: (70000, 784), y: 70000 string labels
    console.print(f"[green]OK[/] · loaded {X.shape[0]} samples, "
                  f"{X.shape[1]} features per sample")

    rng = np.random.default_rng(seed)
    total_saved = 0

    for cls in [str(i) for i in range(10)]:
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        chosen = rng.choice(
            cls_indices,
            size=min(n_per_class, len(cls_indices)),
            replace=False,
        )

        cls_dir = OUTPUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Clear old images so subsequent runs replace cleanly
        for old in cls_dir.glob("*.png"):
            old.unlink()

        for i, idx in enumerate(chosen):
            arr = X[idx].reshape(28, 28).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")  # grayscale
            img.save(cls_dir / f"{cls}_{i:03d}.png", format="PNG")
            total_saved += 1

        console.print(f"  class [cyan]{cls}[/]: {len(chosen)} images")

    console.print()
    console.print(
        f"[bold green]Wrote {total_saved} images[/] across "
        f"{len(list(OUTPUT_DIR.iterdir()))} classes."
    )
    console.print(
        f"  open the dashboard, launch a run with dataset: "
        f"[cyan]{OUTPUT_DIR.relative_to(PROJECT_ROOT)}[/]"
    )


if __name__ == "__main__":
    app()
