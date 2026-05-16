"""Agent roster — compact one-card-per-agent stack.

Each card is emitted as a SINGLE `st.markdown` block so the animation class
actually wraps the content. Card layout is intentionally dense (2 lines per
agent) so the whole roster stays short and doesn't push the timeline/chat
below the fold.
"""
from __future__ import annotations

import html as _html

import streamlit as st

from contracts.messages import AgentEvent

from dashboard.agent_identity import (
    AGENT_DISPLAY_ORDER,
    AgentStatus,
    agent_status_from_events,
    get_identity,
    latest_event_for_agent,
)


_STATUS_BADGE: dict[str, tuple[str, str]] = {
    AgentStatus.IDLE:    ("idle",    "status-idle"),
    AgentStatus.WORKING: ("working", "status-working"),
    AgentStatus.WAITING: ("waiting", "status-waiting"),
    AgentStatus.DONE:    ("done",    "status-done"),
    AgentStatus.ERROR:   ("error",   "status-error"),
}


def render_agent_roster(events: list[AgentEvent]) -> None:
    """Vertical stack of compact cards in pipeline order."""
    focused = st.session_state.get("focused_agent")

    for agent_value in AGENT_DISPLAY_ORDER:
        identity = get_identity(agent_value)
        status = agent_status_from_events(events, agent_value)
        latest = latest_event_for_agent(events, agent_value)
        latest_msg = (latest.message[:50] if latest and latest.message else "—")
        badge_label, badge_class = _STATUS_BADGE[status]
        color_name = identity["color_name"]

        # Per-color animation class — no CSS variables. Each color has its
        # own keyframe defined in styles.py so the pulse actually fires.
        animation_class = ""
        if status == AgentStatus.WORKING:
            animation_class = f"agent-card-working-{color_name}"
        elif status == AgentStatus.WAITING:
            animation_class = f"agent-card-waiting-{color_name}"

        # Whole card as ONE markdown block. Two lines: icon+name+badge, then
        # latest event message truncated.
        card_html = (
            f'<div class="agent-card {animation_class}">'
            f'  <div style="display:flex; align-items:center; gap:8px;">'
            f'    <span style="font-size:1.2rem;">{identity["icon"]}</span>'
            f'    <span style="color:{identity["color"]}; font-weight:700; '
            f'font-size:0.92rem;">{_html.escape(identity["display_name"])}</span>'
            f'    <span class="status-badge {badge_class}" '
            f'style="margin-left:auto;">{badge_label}</span>'
            f'  </div>'
            f'  <div style="color:#aaa; font-size:0.74rem; '
            f'margin-top:2px; white-space:nowrap; overflow:hidden; '
            f'text-overflow:ellipsis;">{_html.escape(latest_msg)}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Compact click-to-focus button (small).
        is_focused = focused == agent_value
        if st.button(
            "✕ Close" if is_focused else "Open →",
            key=f"focus::{agent_value}",
            use_container_width=True,
            type="primary" if is_focused else "secondary",
        ):
            if is_focused:
                st.session_state.pop("focused_agent", None)
            else:
                st.session_state["focused_agent"] = agent_value
            st.rerun()
