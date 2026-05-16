"""Bridge between the coordinator and live HITL surfaces (dashboard + Telegram).

The coordinator only knows the `HITLService` Protocol declared in
`agents/coordinator.py`. This module's `HITLCoordinatorService` satisfies it
structurally — no inheritance needed; the duck-typed match is enough.

Responder convention (audit trail):
  - ``"dashboard"``      — clicked Approve/Reject in Streamlit
  - ``"telegram:<id>"``  — clicked the inline button in Telegram (user id)
  - ``"auto"``           — `AutoApproveHITLService` skeleton placeholder
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from config import APPROVAL_TIMEOUT
from contracts.messages import ApprovalRequest, ApprovalResponse
from memory.store import MemoryStore

from hitl.approval_queue import ApprovalQueue
from hitl.telegram_bot import TelegramApprovalBot


class HITLCoordinatorService:
    """Production HITL service: persists requests, fans out to channels, blocks.

    Pass `telegram=None` for dashboard-only HITL (useful in dev / for the smoke
    test when Telegram secrets aren't configured).
    """

    def __init__(
        self,
        store: MemoryStore,
        queue: ApprovalQueue,
        telegram: TelegramApprovalBot | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.telegram = telegram

    def request_and_wait(
        self,
        request: ApprovalRequest,
        timeout: float = APPROVAL_TIMEOUT,
    ) -> ApprovalResponse:
        """Persist the request, push it to all configured channels, then block.

        Raises `TimeoutError` if no decision lands within `timeout` seconds.
        """
        self.queue.request_approval(request)
        if self.telegram is not None:
            try:
                self.telegram.send_approval_request(request)
            except Exception:  # noqa: BLE001 — never let a bot hiccup kill a run
                logger.exception("Telegram send failed (continuing dashboard-only)")

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
    ) -> None:
        """Push a notification to Telegram if configured. Dashboard sees it via
        the event stream — no extra surface needed.
        """
        if self.telegram is not None:
            try:
                self.telegram.notify(run_id, message, payload)
            except Exception:  # noqa: BLE001
                logger.exception("Telegram notify failed")


def build_hitl_service(store: MemoryStore) -> HITLCoordinatorService:
    """Convenience factory.

    - If `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` are both set, the Telegram
      bot is started and wired in.
    - Otherwise, the service runs dashboard-only.
    """
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    queue = ApprovalQueue(store)
    telegram: TelegramApprovalBot | None = None
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram = TelegramApprovalBot(
            token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
            queue=queue,
        )
        telegram.start()
    else:
        logger.warning(
            "Telegram secrets not configured — HITL service is dashboard-only"
        )

    return HITLCoordinatorService(store=store, queue=queue, telegram=telegram)
