"""Approval queue with hybrid wakeup: threading.Event + SQLite polling.

Two ways an approval gets resolved:

  1. **In-process** (Telegram bot, auto-approver) — the resolver calls
     `queue.resolve(response)`. This persists the response AND sets a
     `threading.Event`, waking the waiter immediately.

  2. **Cross-process** (Streamlit dashboard) — the dashboard process has
     its own `MemoryStore` and writes the response directly via
     `store.respond_to_approval()`. The pipeline process's queue won't get a
     local Event signal, so we also poll the DB on a fixed interval.

The waiter's loop alternates: check DB → wait on Event (or sleep) →
repeat → time out. Either path completes within `poll_interval` of the
resolution.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

from config import APPROVAL_POLL_INTERVAL, APPROVAL_TIMEOUT
from contracts.messages import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from memory.store import MemoryStore


class ApprovalQueue:
    """Single-process bookkeeping over the persisted approval table."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Request side — called by the coordinator
    # ------------------------------------------------------------------
    def request_approval(self, request: ApprovalRequest) -> ApprovalRequest:
        """Persist the request and register an in-process wakeup event."""
        self.store.create_approval_request(request)
        with self._lock:
            self._events[request.request_id] = threading.Event()
        logger.info("ApprovalQueue: enqueued {}", request.request_id)
        return request

    # ------------------------------------------------------------------
    # Resolve side — called by Telegram bot, auto-approver, or tests
    # ------------------------------------------------------------------
    def resolve(self, response: ApprovalResponse) -> ApprovalRequest:
        """Persist the response and signal any in-process waiter.

        Cross-process resolvers (the dashboard) should call
        `store.respond_to_approval()` directly instead; the waiter's DB poll
        will pick it up.
        """
        updated = self.store.respond_to_approval(response)
        with self._lock:
            event = self._events.pop(response.request_id, None)
        if event is not None:
            event.set()
        return updated

    # ------------------------------------------------------------------
    # Waiter — called by the coordinator
    # ------------------------------------------------------------------
    def wait_for_approval(
        self,
        request_id: str,
        timeout: float = APPROVAL_TIMEOUT,
        poll_interval: float = APPROVAL_POLL_INTERVAL,
    ) -> ApprovalResponse | None:
        """Block until the request is resolved or `timeout` elapses.

        Returns the resolved `ApprovalResponse`, or `None` on timeout.
        Raises `LookupError` if the request doesn't exist.
        """
        with self._lock:
            event = self._events.get(request_id)

        deadline = time.monotonic() + timeout
        while True:
            req = self.store.get_approval_request(request_id)
            if req is None:
                raise LookupError(f"No such approval request: {request_id!r}")

            if req.status is ApprovalStatus.RESOLVED:
                # Clean up the local event if still present (cross-process path).
                with self._lock:
                    self._events.pop(request_id, None)
                return _request_to_response(req)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None  # timed out

            wait = min(poll_interval, remaining)
            if event is not None:
                # event.wait returns True if signaled, False on timeout.
                if event.wait(timeout=wait):
                    # Loop will re-read the DB to surface the resolved request.
                    continue
            else:
                time.sleep(wait)


def _request_to_response(req: ApprovalRequest) -> ApprovalResponse:
    """Build an `ApprovalResponse` from a resolved persisted request."""
    if req.decision is None or req.responded_at is None:
        # Should be unreachable — status=RESOLVED implies decision is set.
        raise RuntimeError(
            f"Approval {req.request_id} marked RESOLVED but missing decision/responded_at"
        )
    return ApprovalResponse(
        request_id=req.request_id,
        decision=req.decision,
        response_payload=req.response_payload,
        responder=req.responder,
        comment=req.comment,
        responded_at=req.responded_at,
    )
