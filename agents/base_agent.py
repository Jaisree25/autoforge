"""Shared base class for all AutoForge agents.

Each agent inherits, sets a class-level `name`, and implements `run()`. The
base class provides:

  - `emit_event()`     — dual-sink log: SQLite (live trace) + loguru (terminal).
  - Stored `store` + `run_id` so subclasses don't manage state plumbing.
  - A `_lifecycle()` helper that emits STARTED at entry and COMPLETED at exit.

The `MemoryStore` is injected by the constructor — never imported globally.
That keeps tests independent (each test fixture passes its own tmp store) and
lets future code run multiple pipelines side-by-side with their own stores.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, ClassVar, Iterator

from loguru import logger
from pydantic import BaseModel

from contracts.messages import AgentEvent, EventType
from contracts.schemas import AgentName
from memory.store import MemoryStore


# EventType → loguru level. THINKING is high-volume so it lands at DEBUG;
# everything else gets surfaced at INFO or above.
_EVENT_LEVEL: dict[EventType, str] = {
    EventType.STARTED: "INFO",
    EventType.THINKING: "DEBUG",
    EventType.TOOL_CALL: "INFO",
    EventType.INFO: "INFO",
    EventType.WARNING: "WARNING",
    EventType.ERROR: "ERROR",
    EventType.COMPLETED: "INFO",
    EventType.APPROVAL_REQUESTED: "INFO",
    EventType.APPROVAL_RECEIVED: "INFO",
}


class BaseAgent(ABC):
    """Abstract base for all five AutoForge agents.

    Subclasses set the class-level `name` to one of `AgentName.*` and implement
    `run()` with whatever typed signature they need. The coordinator calls
    `run()` and persists the returned Pydantic model via `MemoryStore`.
    """

    name: ClassVar[AgentName]

    # Event types that get broadcast to Slack when `self.slack` is wired.
    # Keep this tight to avoid flooding the channel — TOOL_CALL and THINKING
    # would push 30+ messages per pipeline run.
    _SLACK_BROADCAST_TYPES: ClassVar[set] = {
        EventType.STARTED,
        EventType.COMPLETED,
        EventType.ERROR,
    }

    def __init__(self, store: MemoryStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        # Slack handle is set by the Coordinator after construction if the
        # HITL service has Slack wired. Default None = no Slack broadcast.
        self.slack: Any = None

    # ------------------------------------------------------------------
    # Live trace
    # ------------------------------------------------------------------
    def emit_event(
        self,
        event_type: EventType,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to SQLite (dashboard) + loguru (terminal) + Slack (if wired)."""
        event = AgentEvent(
            run_id=self.run_id,
            agent=self.name,
            event_type=event_type,
            message=message,
            payload=payload or {},
        )
        self.store.log_event(event)
        level = _EVENT_LEVEL.get(event_type, "INFO")
        logger.opt(depth=1).log(
            level, "[{}] {}: {}", self.name.value, event_type.value, message
        )
        # Broadcast lifecycle / failure events to Slack if wired. Routes to
        # the agent's dedicated channel (per the channel_map) if configured;
        # otherwise to the main coordinator channel. Wrapped in try/except so
        # a flaky bot never kills a pipeline run.
        if self.slack is not None and event_type in self._SLACK_BROADCAST_TYPES:
            try:
                emoji = {
                    EventType.STARTED: ":arrow_forward:",
                    EventType.COMPLETED: ":white_check_mark:",
                    EventType.ERROR: ":x:",
                }.get(event_type, ":mega:")
                self.slack.notify(
                    self.run_id,
                    f"{emoji} *{self.name.value}* {event_type.value}: {message}",
                    agent=self.name.value,
                )
            except Exception:  # noqa: BLE001
                logger.opt(depth=1).warning(
                    "Slack broadcast failed for {} {}",
                    self.name.value, event_type.value,
                )

    @contextmanager
    def _lifecycle(self, summary: str = "") -> Iterator[None]:
        """STARTED at entry, COMPLETED at exit, ERROR on exception."""
        self.emit_event(EventType.STARTED, message=summary)
        try:
            yield
        except Exception as exc:  # noqa: BLE001 — re-raised below
            self.emit_event(
                EventType.ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            self.emit_event(EventType.COMPLETED, message=summary)

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Execute the agent's work. Return a Pydantic model from `contracts.schemas`.

        Subclasses must wrap their work in `with self._lifecycle(...)` so
        STARTED/COMPLETED/ERROR events appear in the live trace.
        """
        raise NotImplementedError
