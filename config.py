"""Centralized configuration constants for AutoForge.

All paths, model names, and tunable constants live here. Modules should import
from `config` rather than reading environment variables directly or hard-coding
strings.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows consoles default to cp1252 which can't encode the box-drawing chars
# Rich uses (e.g. console.rule's ─). Reconfigure stdout/stderr to utf-8 before
# anything else imports Console. No-op on non-Windows / older Pythons.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
ARTIFACTS_DIR: Path = DATA_DIR / "artifacts"
MEMORY_DIR: Path = PROJECT_ROOT / "memory"

# Load .env from project root (silently no-ops if absent)
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Nemotron / NVIDIA NIM
# ---------------------------------------------------------------------------
NVIDIA_API_KEY: str | None = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL: str = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
COORDINATOR_MODEL: str = os.getenv(
    "COORDINATOR_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5"
)
WORKER_MODEL: str = os.getenv(
    "WORKER_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2"
)

# ---------------------------------------------------------------------------
# Strategy Agent — web search
# ---------------------------------------------------------------------------
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# HITL — Slack
# ---------------------------------------------------------------------------
SLACK_BOT_TOKEN: str | None = os.getenv("SLACK_BOT_TOKEN")
# Main channel — hosts approval requests, Coordinator lifecycle, final results.
SLACK_CHANNEL_ID: str | None = os.getenv("SLACK_CHANNEL_ID")
# Optional per-agent channels (env vars use DISPLAY names: Profiler /
# Researcher / Preparer / Trainer / Evaluator / Optimizer). If a given var
# is unset, that agent's STARTED/COMPLETED/ERROR events fall back to
# SLACK_CHANNEL_ID. Channel IDs (Cxxxxxx), not display names.
SLACK_CHANNEL_PROFILER:  str | None = os.getenv("SLACK_CHANNEL_PROFILER")
SLACK_CHANNEL_RESEARCHER: str | None = os.getenv("SLACK_CHANNEL_RESEARCHER")
SLACK_CHANNEL_PREPARER:  str | None = os.getenv("SLACK_CHANNEL_PREPARER")
SLACK_CHANNEL_TRAINER:   str | None = os.getenv("SLACK_CHANNEL_TRAINER")
SLACK_CHANNEL_EVALUATOR: str | None = os.getenv("SLACK_CHANNEL_EVALUATOR")
SLACK_CHANNEL_OPTIMIZER: str | None = os.getenv("SLACK_CHANNEL_OPTIMIZER")


def slack_channel_map() -> dict[str, str]:
    """Return a dict of {AgentName.value → channel_id} for agents that have
    a dedicated channel configured. The env var names use DISPLAY names
    (Profiler/Researcher/Preparer/...), the dict keys are internal AgentName
    values (profiler/strategy/dataset/training/benchmark/hardware) — the
    code routes by internal name."""
    mapping: dict[str, str] = {}
    for agent_value, channel in (
        ("profiler",  SLACK_CHANNEL_PROFILER),
        ("strategy",  SLACK_CHANNEL_RESEARCHER),
        ("dataset",   SLACK_CHANNEL_PREPARER),
        ("training",  SLACK_CHANNEL_TRAINER),
        ("benchmark", SLACK_CHANNEL_EVALUATOR),
        ("hardware",  SLACK_CHANNEL_OPTIMIZER),
    ):
        if channel:
            mapping[agent_value] = channel
    return mapping

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
AUTOFORGE_DB_PATH: Path = Path(
    os.getenv("AUTOFORGE_DB_PATH", str(DATA_DIR / "autoforge.db"))
).resolve()
MEMORY_SCHEMA_PATH: Path = MEMORY_DIR / "schema.sql"

# ---------------------------------------------------------------------------
# HITL polling / timeouts (seconds)
# ---------------------------------------------------------------------------
APPROVAL_POLL_INTERVAL: float = 1.0
APPROVAL_TIMEOUT: float = 600.0  # 10 minutes — generous for live demo
DASHBOARD_REFRESH_INTERVAL_MS: int = 1500

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging() -> None:
    """Configure loguru sinks for the entire process.

    Call once at the top of each entry point (scripts, dashboard, main). Safe to
    call multiple times — replaces the existing sink each call.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,
        enqueue=False,
    )


# Ensure data directories exist (cheap, idempotent)
for _d in (DATA_DIR, UPLOADS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
