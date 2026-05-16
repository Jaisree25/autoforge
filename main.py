"""Convenience entry point — delegates to scripts.run_pipeline.

So `python main.py --dataset ... --objective ...` does the same thing as
`python scripts/run_pipeline.py --dataset ... --objective ...`.
"""
from __future__ import annotations

from scripts.run_pipeline import app


if __name__ == "__main__":
    app()
