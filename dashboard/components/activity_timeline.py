"""Activity timeline — Plotly Gantt chart of agent work over time.

One row per agent, one bar per (started → completed) run. Bars colored using
each agent's accent color. In-progress work extends to "now". When an agent
runs multiple times (e.g. Hardware Specialist runs twice), each run gets its
own bar on the same row.

Hover over a bar to see the underlying STARTED-event message.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from contracts.messages import AgentEvent, EventType

from dashboard.agent_identity import AGENT_DISPLAY_ORDER, get_identity


def render_activity_timeline(events: list[AgentEvent]) -> None:
    st.subheader("Activity timeline")
    if not events:
        st.caption("Timeline will populate as agents run.")
        return

    bars = _build_bars(events)
    if not bars:
        st.caption("Waiting for the first agent to start.")
        return

    df = pd.DataFrame(bars)
    color_map = {
        get_identity(av)["display_name"]: get_identity(av)["color"]
        for av in AGENT_DISPLAY_ORDER
    }

    # Pin the y-axis category order to the pipeline order (top-down).
    y_order = [get_identity(av)["display_name"] for av in AGENT_DISPLAY_ORDER]

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="agent",
        color="agent",
        color_discrete_map=color_map,
        category_orders={"agent": y_order},
        hover_data={"message": True, "agent": False},
    )
    fig.update_yaxes(autorange="reversed")  # first agent on top
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _build_bars(events: list[AgentEvent]) -> list[dict]:
    """Pair STARTED with COMPLETED events per agent. In-progress = extend to now."""
    now = datetime.now(timezone.utc)
    bars: list[dict] = []
    for agent_value in AGENT_DISPLAY_ORDER:
        identity = get_identity(agent_value)
        starts = [
            e for e in events
            if e.agent.value == agent_value and e.event_type == EventType.STARTED
        ]
        completes = [
            e for e in events
            if e.agent.value == agent_value and e.event_type == EventType.COMPLETED
        ]
        for i, start_ev in enumerate(starts):
            end_dt = completes[i].created_at if i < len(completes) else now
            bars.append({
                "agent": identity["display_name"],
                "start": start_ev.created_at,
                "end": end_dt,
                "message": (start_ev.message or "")[:80],
            })
    return bars
