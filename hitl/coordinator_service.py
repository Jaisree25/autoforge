"""Bridge between the coordinator and live HITL surfaces (dashboard + Slack).

The coordinator only knows the `HITLService` Protocol declared in
`agents/coordinator.py`. This module's `HITLCoordinatorService` satisfies it
structurally — no inheritance needed; the duck-typed match is enough.

Two Slack delivery modes
------------------------
1. **outbox** (default): the Slack bot runs in a separate process (the
   NemoClaw sandbox). `request_and_wait` only persists the approval —
   the sandboxed bot polls `list_unposted_pending_approvals()` and posts.
   `notify()` enqueues into `slack_outbox`; the bot drains it.

2. **in-process** (set `AUTOFORGE_SLACK_IN_PROCESS=1`): for local dev
   without spinning up the sandbox. A `SlackApprovalBot` instance runs
   inside the pipeline process and is pushed to synchronously. Marks
   approvals as posted so a sandbox bot running in parallel (during a
   migration window) won't duplicate posts.

Responder convention (audit trail):
  - ``"dashboard"``      — clicked Approve/Reject in Streamlit
  - ``"slack:<id>"``     — replied CONFIRM/REJECT in Slack (user id)
  - ``"auto"``           — `AutoApproveHITLService` skeleton placeholder
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger

from config import APPROVAL_TIMEOUT
from contracts.messages import ApprovalRequest, ApprovalResponse
from contracts.schemas import AgentName
from memory.store import MemoryStore

from hitl.approval_queue import ApprovalQueue
from hitl.slack_bot import SlackApprovalBot


def _in_process_mode() -> bool:
    return os.getenv("AUTOFORGE_SLACK_IN_PROCESS", "").lower() in (
        "1", "true", "yes", "on",
    )


class HITLCoordinatorService:
    """Production HITL service: persists requests, fans out to channels, blocks.

    Pass `slack=None` for the outbox path (sandbox bot drains). Pass an
    in-process `SlackApprovalBot` for direct synchronous delivery.
    """

    def __init__(
        self,
        store: MemoryStore,
        queue: ApprovalQueue,
        slack: SlackApprovalBot | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.slack = slack

    def request_and_wait(
        self,
        request: ApprovalRequest,
        timeout: float = APPROVAL_TIMEOUT,
    ) -> ApprovalResponse:
        """Persist the request, fan out, then block.

        Raises `TimeoutError` if no decision lands within `timeout` seconds.
        """
        self.queue.request_approval(request)

        if self.slack is not None:
            # In-process delivery: post immediately, then stamp as posted so a
            # parallel sandbox bot won't re-post the same gate.
            try:
                self.slack.send_approval_request(request)
                self.store.mark_approval_posted(request.request_id)
            except Exception:  # noqa: BLE001 — never let a bot hiccup kill a run
                logger.exception("In-process Slack send failed (continuing)")
        # else: outbox path — sandbox bot picks up via its poll loop.

        response = self.queue.wait_for_approval(request.request_id, timeout=timeout)
        if response is None:
            raise TimeoutError(
                f"Approval request {request.request_id!r} timed out after {timeout}s"
            )
        return response

    def notify(
        self,
        run_id: str,
        message: str,
        payload: dict[str, Any] | None = None,
        agent: AgentName | str | None = None,
    ) -> None:
        """Send a notification through whichever Slack delivery mode is active.

        Out-of-process: persists to `slack_outbox`; sandbox bot drains.
        In-process: calls the bot's `notify()` directly.
        """
        if self.slack is not None:
            try:
                self.slack.notify(run_id, message, payload=payload, agent=agent)
            except Exception:  # noqa: BLE001
                logger.exception("In-process Slack notify failed")
            return

        # Outbox path. Coerce string agent name → AgentName enum.
        agent_enum: AgentName | None
        if isinstance(agent, str):
            try:
                agent_enum = AgentName(agent)
            except ValueError:
                agent_enum = None  # unknown agent name → main channel
        else:
            agent_enum = agent
        try:
            self.store.enqueue_slack_notification(run_id, message, agent=agent_enum)
        except Exception:  # noqa: BLE001
            logger.exception("Slack outbox enqueue failed")


def build_hitl_service(store: MemoryStore) -> HITLCoordinatorService:
    """Convenience factory.

    Default: **outbox mode** — the sandboxed `hitl/slack_bot_runner.py` drains
    `slack_outbox` and `list_unposted_pending_approvals()`. The host pipeline
    needs no Slack tokens.

    Override with `AUTOFORGE_SLACK_IN_PROCESS=1` (plus `SLACK_BOT_TOKEN` +
    `SLACK_CHANNEL_ID`) to start a `SlackApprovalBot` inside the pipeline
    process — useful for local dev when the NemoClaw sandbox isn't running.
    """
    from config import SLACK_BOT_TOKEN, SLACK_CHANNEL_ID, slack_channel_map

    queue = ApprovalQueue(store)
    slack: SlackApprovalBot | None = None

    if _in_process_mode():
        if SLACK_BOT_TOKEN and SLACK_CHANNEL_ID:
            channel_map = slack_channel_map()
            slack = SlackApprovalBot(
                token=SLACK_BOT_TOKEN,
                channel=SLACK_CHANNEL_ID,
                queue=queue,
                channel_map=channel_map,
            )
            slack.start()
            logger.info(
                "HITL Slack mode: in-process bot ({} per-agent channels)",
                len(channel_map),
            )
        else:
            logger.warning(
                "AUTOFORGE_SLACK_IN_PROCESS=1 but SLACK_BOT_TOKEN / "
                "SLACK_CHANNEL_ID missing — falling back to outbox mode",
            )
    else:
        logger.info(
            "HITL Slack mode: outbox (sandbox bot drains slack_outbox / "
            "list_unposted_pending_approvals)",
        )

    return HITLCoordinatorService(store=store, queue=queue, slack=slack)
