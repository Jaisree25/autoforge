"""CLI subprocess smoke tests.

The unit tests construct `Coordinator` / `MemoryStore` directly, so they
never exercise the actual entry-point scripts. That leaves a real coverage
gap: typer arg parsing, sys.path manipulation, env loading via python-dotenv,
import resolution from the script-as-main entry point.

Each test here invokes a script as a subprocess with a fast-exit invocation
(`--help`, or `init_db.py` which runs in <1s) and asserts exit code + a
substring of stdout/stderr. This is the exact class of bug we hit at Phase 6
when `streamlit run dashboard/app.py` blew up on `from config import ...`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(
    args: list[str],
    env_overrides: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Subprocess helper. Uses the same Python interpreter pytest is running
    under (so it inherits the autoforge env)."""
    env = {**os.environ, **(env_overrides or {})}
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        # The scripts reconfigure stdout to utf-8 in config.py. Default
        # subprocess decoding on Windows is cp1252, which would fail on
        # box-drawing chars; force utf-8 to match.
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def test_init_db_runs_clean(tmp_path):
    """init_db.py should set up the DB + dump JSON schemas + exit 0."""
    db = tmp_path / "test.db"
    result = _run(
        ["scripts/init_db.py"],
        env_overrides={"AUTOFORGE_DB_PATH": str(db)},
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "SQLite store ready" in result.stdout
    assert "schema.json" in result.stdout
    assert db.exists()


def test_run_pipeline_help_lists_options():
    """`run_pipeline.py --help`: typer wiring + sys.path + imports all healthy."""
    result = _run(["scripts/run_pipeline.py", "--help"])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    out = result.stdout
    assert "--dataset" in out
    assert "--objective" in out
    assert "--auto-approve" in out


def test_main_help_delegates_to_run_pipeline():
    """`main.py --help` should expose the same CLI as run_pipeline.py."""
    result = _run(["main.py", "--help"])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    out = result.stdout
    assert "--dataset" in out
    assert "--objective" in out
    assert "--auto-approve" in out


def test_test_nemotron_without_api_key_exits_one_with_hint():
    """Without NVIDIA_API_KEY, the script should exit 1 with a clear hint
    rather than blowing up inside the OpenAI client."""
    # python-dotenv with override=False respects pre-set env vars, so setting
    # NVIDIA_API_KEY to empty string overrides any value a future .env may set.
    result = _run(
        ["scripts/test_nemotron.py"],
        env_overrides={"NVIDIA_API_KEY": ""},
    )
    assert result.returncode == 1, (
        f"expected exit 1 (no API key)\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "NVIDIA_API_KEY" in combined
    assert ".env" in combined  # tells the user where to put it
