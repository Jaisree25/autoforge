"""Auto-approve HITL service — Phase 4 placeholder.

The coordinator depends on a HITL service with a `request_and_wait()` +
`notify()` interface. The real implementation (Phase 5) bridges to Streamlit
and Telegram. Until then, `AutoApproveHITLService` records the approval
request in `MemoryStore` and immediately resolves it with `approved`.

This keeps the end-to-end pipeline runnable for the skeleton smoke test and
for any future "headless" runs (CI, benchmarking) that shouldn't block on a
human.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from memory.store import MemoryStore


class AutoApproveHITLService:
    """Duck-typed interface: `request_and_wait`, `notify`.

    For headless/CI runs. Persists the request, broadcasts to Slack (if
    wired) so the demo Slack feed still shows approval activity, then
    immediately resolves with APPROVED.
    """

    def __init__(
        self,
        store: MemoryStore,
        responder: str = "auto",
        slack: Any = None,
    ) -> None:
        self.store = store
        self.responder = responder
        # When set, agent lifecycle broadcasts from the Coordinator path
        # still reach Slack even in auto-approve runs. The Coordinator pulls
        # this via `getattr(hitl, "slack", None)`.
        self.slack = slack

    def request_and_wait(
        self,
        request: ApprovalRequest,
        timeout: float = 600.0,  # noqa: ARG002 — unused in auto mode
    ) -> ApprovalResponse:
        """Persist the request, push to Slack if wired, immediately auto-approve."""
        self.store.create_approval_request(request)
        if self.slack is not None:
            try:
                self.slack.send_approval_request(request)
            except Exception:  # noqa: BLE001
                logger.exception("Slack send failed (auto-approve continuing)")
        response = ApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVED,
            responder=self.responder,
            comment="auto-approved (headless / CI mode)",
        )
        self.store.respond_to_approval(response)
        logger.info(
            "Auto-approved [{}] '{}' for run {}",
            request.request_id, request.title, request.run_id,
        )
        if self.slack is not None:
            try:
                self.slack.notify(
                    request.run_id,
                    f":robot_face: auto-approved gate: *{request.title}*",
                )
            except Exception:  # noqa: BLE001
                logger.exception("Slack auto-approve notify failed")
        return response

    def notify(
        self,
        run_id: str,
        message: str,
        payload: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        logger.info("(notify) run={}: {}", run_id, message)
        if self.slack is not None:
            try:
                self.slack.notify(run_id, message)
            except Exception:  # noqa: BLE001
                logger.exception("Slack notify failed (auto-approve continuing)")
