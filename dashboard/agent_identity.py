"""Single source of truth for agent visual identity.

Every UI module imports from here. Never hard-code an agent name, color, or
icon elsewhere — the whole point of this registry is that swapping a color
or display name happens in exactly one place.
"""
from __future__ import annotations

from typing import Any

from contracts.messages import EventType
from contracts.schemas import AgentName


AGENT_IDENTITIES: dict[str, dict[str, Any]] = {
    AgentName.PROFILER.value: {
        "display_name": "Profiler",
        "role": "Observe & Detect",
        "color": "#06B6D4",
        "color_name": "cyan",
        "icon": "🔍",
        "model": "Nemotron-Nano-9B",
    },
    AgentName.STRATEGY.value: {
        "display_name": "Researcher",
        "role": "Strategy & Literature",
        "color": "#A855F7",
        "color_name": "purple",
        "icon": "📚",
        "model": "Nemotron-Super-49B",
    },
    AgentName.DATASET.value: {
        "display_name": "Data Preparer",
        "role": "Clean & Augment",
        "color": "#4F8EF7",
        "color_name": "blue",
        "icon": "🧹",
        "model": "Nemotron-Nano-9B",
    },
    AgentName.TRAINING.value: {
        "display_name": "Trainer",
        "role": "HPO & Tuning",
        "color": "#10B981",
        "color_name": "green",
        "icon": "🧠",
        "model": "Nemotron-Nano-9B",
    },
    AgentName.BENCHMARK.value: {
        "display_name": "Evaluator",
        "role": "Accuracy & Performance",
        "color": "#EF4444",
        "color_name": "red",
        "icon": "📐",
        "model": "Nemotron-Nano-9B",
    },
    AgentName.HARDWARE.value: {
        "display_name": "Optimizer",
        "role": "Quantize & Export",
        "color": "#F59E0B",
        "color_name": "amber",
        "icon": "⚙️",
        "model": "Nemotron-Nano-9B",
    },
    AgentName.COORDINATOR.value: {
        "display_name": "Director",
        "role": "Project Coordinator",
        "color": "#6366F1",
        "color_name": "indigo",
        "icon": "🎯",
        "model": "Nemotron-Super-49B",
    },
}

# Display order for cards / graph nodes — left-to-right pipeline flow.
AGENT_DISPLAY_ORDER: list[str] = [
    AgentName.PROFILER.value,
    AgentName.STRATEGY.value,
    AgentName.DATASET.value,
    AgentName.TRAINING.value,
    AgentName.BENCHMARK.value,
    AgentName.HARDWARE.value,
]


def get_identity(agent: AgentName | str) -> dict[str, Any]:
    """Look up the visual identity for an agent.

    Accepts either an `AgentName` enum or the underlying string value.
    Falls back to a neutral identity if the agent isn't in the registry.
    """
    key = agent.value if isinstance(agent, AgentName) else str(agent)
    return AGENT_IDENTITIES.get(key, {
        "display_name": key,
        "role": "Unknown",
        "color": "#888888",
        "color_name": "gray",
        "icon": "🤖",
        "model": "?",
    })


# ---------------------------------------------------------------------------
# Agent status derivation — used by roster cards + network graph highlighting.
# ---------------------------------------------------------------------------
class AgentStatus:
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


_STATUS_FOR_EVENT: dict[EventType, str] = {
    EventType.STARTED: AgentStatus.WORKING,
    EventType.THINKING: AgentStatus.WORKING,
    EventType.TOOL_CALL: AgentStatus.WORKING,
    EventType.INFO: AgentStatus.WORKING,
    EventType.APPROVAL_REQUESTED: AgentStatus.WAITING,
    EventType.APPROVAL_RECEIVED: AgentStatus.WORKING,
    EventType.WARNING: AgentStatus.WORKING,
    EventType.ERROR: AgentStatus.ERROR,
    EventType.COMPLETED: AgentStatus.DONE,
}


def agent_status_from_events(
    events: list[Any],
    agent: AgentName | str,
) -> str:
    """Walk the events for this agent and report the most recent status.

    `events` is assumed sorted oldest-first (the way `MemoryStore.get_events`
    returns them). The status reflects the most recent event type.
    """
    agent_value = agent.value if isinstance(agent, AgentName) else str(agent)
    latest_status = AgentStatus.IDLE
    for ev in events:
        ev_agent = ev.agent.value if isinstance(ev.agent, AgentName) else str(ev.agent)
        if ev_agent != agent_value:
            continue
        latest_status = _STATUS_FOR_EVENT.get(ev.event_type, AgentStatus.WORKING)
    return latest_status


def latest_event_for_agent(
    events: list[Any],
    agent: AgentName | str,
) -> Any | None:
    """Most recent event for the given agent (or None)."""
    agent_value = agent.value if isinstance(agent, AgentName) else str(agent)
    latest = None
    for ev in events:
        ev_agent = ev.agent.value if isinstance(ev.agent, AgentName) else str(ev.agent)
        if ev_agent == agent_value:
            latest = ev
    return latest
