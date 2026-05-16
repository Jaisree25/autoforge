"""Long-lived Slack bot runner — intended to run inside the NemoClaw sandbox.

The host pipeline (`HITLCoordinatorService` in outbox mode) never touches
Slack directly anymore. This runner does, from inside the OpenShell sandbox,
talking to the same SQLite DB through three loops:

  1. **Reply poll** — `SlackApprovalBot._poll_loop` (reused as-is). Reads
     channel history, parses ``CONFIRM`` / ``REJECT`` / digit replies,
     resolves the approval queue.
  2. **Unposted-approval poll** (new) — finds pending approvals the bot
     hasn't yet pushed (`store.list_unposted_pending_approvals`),
     posts them, stamps `slack_posted`.
  3. **Outbox drain** (new) — agents' STARTED/COMPLETED/ERROR pings get
     written to `slack_outbox` by `HITLCoordinatorService.notify`; this
     loop drains them.

Token flow inside the sandbox: `SLACK_BOT_TOKEN` is set to a placeholder
(``openshell:resolve:env:SLACK_BOT_TOKEN``); the L7 proxy substitutes the
real value at egress to slack.com. The runner reads `os.getenv` either way
— same code works on host for local dev.

Usage:
    # Inside the NemoClaw sandbox:
    python -m hitl.slack_bot_runner

    # On host for local dev (real tokens in .env):
    AUTOFORGE_SLACK_IN_PROCESS=  python -m hitl.slack_bot_runner

The pipeline process should NOT also run an in-process bot
(`AUTOFORGE_SLACK_IN_PROCESS=1`) at the same time — duplicate posts.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
from types import FrameType
from typing import Any

from loguru import logger

from config import (
    AUTOFORGE_DB_PATH,
    SLACK_BOT_TOKEN,
    SLACK_CHANNEL_ID,
    configure_logging,
    slack_channel_map,
)
from hitl.approval_queue import ApprovalQueue
from hitl.slack_bot import SlackApprovalBot
from memory.store import MemoryStore


# Two-second cadence: snappy for HITL gates, gentle on SQLite. Override
# with AUTOFORGE_SLACK_BRIDGE_POLL_SEC for testing.
_DEFAULT_POLL_SEC = 2.0
_POLL_INTERVAL = float(
    os.getenv("AUTOFORGE_SLACK_BRIDGE_POLL_SEC", str(_DEFAULT_POLL_SEC))
)


def _unposted_approval_loop(
    bot: SlackApprovalBot,
    store: MemoryStore,
    stop_event: threading.Event,
    interval: float = _POLL_INTERVAL,
) -> None:
    """Push pending approvals that the bot hasn't yet sent to Slack.

    Idempotent on bot restarts: `mark_approval_posted` uses INSERT OR IGNORE.
    """
    logger.info("Unposted-approval poller started (interval={}s)", interval)
    while not stop_event.is_set():
        try:
            for req in store.list_unposted_pending_approvals():
                try:
                    bot.send_approval_request(req)
                    store.mark_approval_posted(req.request_id)
                    logger.info(
                        "Posted approval {} ({}) to Slack",
                        req.request_id, req.title,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed posting approval {} — will retry next tick",
                        req.request_id,
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Unposted-approval poll iteration crashed")
        stop_event.wait(interval)
    logger.info("Unposted-approval poller stopped")


def _outbox_loop(
    bot: SlackApprovalBot,
    store: MemoryStore,
    stop_event: threading.Event,
    interval: float = _POLL_INTERVAL,
) -> None:
    """Drain `slack_outbox` — agents' STARTED/COMPLETED/ERROR pings."""
    logger.info("Outbox poller started (interval={}s)", interval)
    while not stop_event.is_set():
        try:
            for row in store.list_unsent_notifications(limit=20):
                try:
                    bot.notify(
                        run_id=row["run_id"],
                        message=row["message"],
                        agent=row["agent"],
                    )
                    store.mark_notification_sent(row["id"])
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed posting outbox row {} — will retry", row["id"],
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Outbox poll iteration crashed")
        stop_event.wait(interval)
    logger.info("Outbox poller stopped")


def main() -> int:
    configure_logging()

    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        logger.error(
            "SLACK_BOT_TOKEN / SLACK_CHANNEL_ID missing. Inside the sandbox "
            "these should resolve to gateway-provided placeholders "
            "(openshell:resolve:env:*). On host they're the real tokens. "
            "Aborting."
        )
        return 2

    channel_map = slack_channel_map()
    logger.info(
        "Slack bridge starting — db={} main_channel={} "
        "per_agent_channels={}",
        AUTOFORGE_DB_PATH, SLACK_CHANNEL_ID, list(channel_map.keys()),
    )

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    # init_schema is idempotent — safe even if the host pipeline created it first.
    store.init_schema()

    queue = ApprovalQueue(store)
    bot = SlackApprovalBot(
        token=SLACK_BOT_TOKEN,
        channel=SLACK_CHANNEL_ID,
        queue=queue,
        channel_map=channel_map,
    )

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal {} — shutting down", signum)
        stop_event.set()
        bot.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    bot.start()  # the existing reply-parsing loop
    threads = [
        threading.Thread(
            target=_unposted_approval_loop,
            args=(bot, store, stop_event),
            name="unposted-approvals",
            daemon=True,
        ),
        threading.Thread(
            target=_outbox_loop,
            args=(bot, store, stop_event),
            name="slack-outbox",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    stop_event.wait()
    for t in threads:
        t.join(timeout=5.0)
    logger.info("Slack bridge stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
