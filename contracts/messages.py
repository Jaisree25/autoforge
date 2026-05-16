"""Inter-process message types: agent events and HITL approvals.

`AgentEvent` is what flows through the live trace surfaced by the dashboard.
`ApprovalRequest` / `ApprovalResponse` are the contract for Human-in-the-Loop
gates (the Coordinator pauses until a response lands).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import AgentName


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Agent events — live trace surfaced by the dashboard
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    STARTED = "started"
    THINKING = "thinking"  # streamed reasoning chunk from an LLM call
    TOOL_CALL = "tool_call"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    COMPLETED = "completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RECEIVED = "approval_received"


class AgentEvent(_Base):
    """One observable step in an agent's execution.

    Streamed by agents via `BaseAgent.emit_event()`, persisted by
    `MemoryStore.log_event()`, rendered live by the dashboard.
    """

    event_id: str = Field(default_factory=_new_id)
    run_id: str
    agent: AgentName
    event_type: EventType
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# HITL — approval requests and responses
# ---------------------------------------------------------------------------
class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"  # human approved with modifications (see response.payload)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"


class ApprovalRequest(_Base):
    """Asks the human to approve, reject, or edit an agent's proposed output.

    `payload` is the JSON-serialized form of the artifact under review
    (e.g. a `StrategySpec`). The dashboard renders it as editable JSON; the
    Telegram bot offers Approve / Reject inline buttons (no edit on Telegram).
    """

    request_id: str = Field(default_factory=_new_id)
    run_id: str
    agent: AgentName
    title: str
    description: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: ApprovalDecision | None = None
    response_payload: dict[str, Any] | None = None
    responder: str | None = None  # "dashboard" | "telegram:<user_id>" | "auto"
    responded_at: datetime | None = None
    comment: str = ""


class ApprovalResponse(_Base):
    """Human reply to an `ApprovalRequest`."""

    request_id: str
    decision: ApprovalDecision
    response_payload: dict[str, Any] | None = None
    responder: str | None = None
    comment: str = ""
    responded_at: datetime = Field(default_factory=_utcnow)
