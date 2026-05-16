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
# HITL — Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

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
# Agent execution
# ---------------------------------------------------------------------------
STUB_AGENT_SLEEP: float = 0.5  # simulated thinking time for skeleton stubs
AGENT_EVENT_FLUSH: bool = True  # flush events to DB immediately for live trace

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
