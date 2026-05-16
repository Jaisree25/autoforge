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
    """Duck-typed interface: `request_and_wait(request, timeout=...)`, `notify(run_id, message, payload=None)`.

    The Phase 5 implementation will keep the same shape so the coordinator
    doesn't change.
    """

    def __init__(self, store: MemoryStore, responder: str = "auto") -> None:
        self.store = store
        self.responder = responder

    def request_and_wait(
        self,
        request: ApprovalRequest,
        timeout: float = 600.0,  # noqa: ARG002 — unused in auto mode
    ) -> ApprovalResponse:
        """Persist the request and immediately auto-approve it."""
        self.store.create_approval_request(request)
        response = ApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVED,
            responder=self.responder,
            comment="auto-approved (no HITL service wired in yet)",
        )
        self.store.respond_to_approval(response)
        logger.info(
            "Auto-approved [{}] '{}' for run {}",
            request.request_id, request.title, request.run_id,
        )
        return response

    def notify(
        self,
        run_id: str,
        message: str,
        payload: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        logger.info("(notify) run={}: {}", run_id, message)
