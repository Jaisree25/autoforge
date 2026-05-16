"""AutoForge dashboard — multi-agent team visualization.

Layout:

  ┌──────────────────────────────────────────────────────────────┐
  │ Sidebar: run picker + run metadata                            │
  ├──────────────┬───────────────────────────┬───────────────────┤
  │ AGENTS       │ PIPELINE FLOW              │ APPROVALS         │
  │ (panel)      │ (panel, animated)          │ (panel)           │
  ├──────────────┴───────────────────────────┴───────────────────┤
  │ FOCUSED AGENT DETAILS (when a card is clicked)               │
  ├───────────────────────────────────────────────────────────────┤
  │ ACTIVITY TIMELINE                                            │
  ├───────────────────────────────────────────────────────────────┤
  │ AGENT CHAT FEED                                              │
  └───────────────────────────────────────────────────────────────┘

Each top-row tile is a bordered container with a colored header strip — they
should read as three distinct panels, not a single whiteboard.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts the script dir on sys.path, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import (
    AUTOFORGE_DB_PATH,
    DASHBOARD_REFRESH_INTERVAL_MS,
    configure_logging,
)
from contracts.schemas import AgentName, PipelineStatus
from memory.store import MemoryStore

from dashboard.agent_identity import get_identity
from dashboard.styles import PULSE_CSS

from dashboard.components.activity_timeline import render_activity_timeline
from dashboard.components.agent_detail import render_agent_detail
from dashboard.components.agent_roster import render_agent_roster
from dashboard.components.approval_panel import render_approval_panel
from dashboard.components.chat_feed import render_chat_feed
from dashboard.components.network_graph import render_network_graph
from dashboard.components.run_launcher import render_run_launcher
from dashboard.components.run_picker import render_run_picker

configure_logging()

st.set_page_config(
    page_title="AutoForge",
    layout="wide",
    page_icon="🔨",
    initial_sidebar_state="expanded",
)
st.markdown(PULSE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached store
# ---------------------------------------------------------------------------
@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore(db_path=AUTOFORGE_DB_PATH)


store = get_store()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
render_run_launcher()  # at the top so it's the first thing users see
run = render_run_picker(store)
if run is None:
    st.title("🔨 AutoForge")
    st.warning("No pipeline runs yet — use the **Start a new run** panel in the sidebar.")
    st.caption(
        "_Or populate a synthetic run for UI iteration:_ "
        "`python scripts/fake_events.py`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Auto-refresh (suspended during edit so text-area state survives reruns)
# ---------------------------------------------------------------------------
edit_active = bool(st.session_state.get("edit_request_id"))
if not edit_active:
    st_autorefresh(
        interval=DASHBOARD_REFRESH_INTERVAL_MS,
        key=f"autorefresh::{run.run_id}",
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_color = {
    PipelineStatus.PENDING: "gray",
    PipelineStatus.RUNNING: "blue",
    PipelineStatus.AWAITING_APPROVAL: "orange",
    PipelineStatus.COMPLETED: "green",
    PipelineStatus.FAILED: "red",
    PipelineStatus.CANCELLED: "gray",
}.get(run.status, "gray")
st.markdown(
    f"## 🔨 AutoForge &nbsp;·&nbsp; "
    f"Run `{run.run_id[:8]}` &nbsp;·&nbsp; "
    f":{status_color}[**{run.status.value}**]"
)
st.caption(run.objective)


# Pull events once per render — every component reads the same snapshot.
events = store.get_events(run.run_id, limit=500)


# ---------------------------------------------------------------------------
# Top row: three distinct panels (roster | flow | approvals)
#
# Each panel is `st.container(border=True)` with a colored-header strip on top
# so they read as separate tiles, not one big whiteboard.
# ---------------------------------------------------------------------------
def _panel_header(title: str, subtitle: str, accent_class: str) -> None:
    st.markdown(
        f'<div class="{accent_class}">'
        f'<div class="panel-header">'
        f'<span class="panel-header-title">{title}</span>'
        f'<span class="panel-header-sub">· {subtitle}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


col_roster, col_graph, col_approval = st.columns([2, 5, 2], gap="medium")

focused = st.session_state.get("focused_agent")

with col_roster:
    with st.container(border=True):
        _panel_header(
            "Agents",
            "click a card to drill in",
            "panel-accent-blue",
        )
        render_agent_roster(events)

with col_graph:
    with st.container(border=True):
        if focused:
            # Detail view REPLACES the pipeline flow until "Back" is clicked.
            identity = get_identity(focused)
            _panel_header(
                identity["display_name"],
                f"detail view · {identity['role']}",
                "panel-accent-purple",
            )
            render_agent_detail(run, events, focused)
        else:
            _panel_header(
                "Pipeline flow",
                "live handoffs between agents",
                "panel-accent-purple",
            )
            render_network_graph(events)

with col_approval:
    with st.container(border=True):
        _panel_header(
            "Approvals",
            "HITL handoff gates",
            "panel-accent-amber",
        )
        render_approval_panel(store, run.run_id)


# ---------------------------------------------------------------------------
# Activity timeline (full width)
# ---------------------------------------------------------------------------
render_activity_timeline(events)


# ---------------------------------------------------------------------------
# Chat feed (full width, last)
# ---------------------------------------------------------------------------
render_chat_feed(events)


# ---------------------------------------------------------------------------
# Per-agent outputs — collapsed JSON dump at the very bottom
# (Mostly subsumed by the focused-agent detail panel above; kept as a fallback
#  way to see every output side-by-side without clicking through.)
# ---------------------------------------------------------------------------
with st.expander("All agent outputs (JSON)", expanded=False):
    _output_cards = [
        ("Dataset Profile", "dataset_profile", AgentName.DATASET),
        ("Strategy Spec", "strategy_spec", AgentName.STRATEGY),
        ("Training Envelope", "training_envelope", AgentName.HARDWARE),
        ("Training Result", "training_result", AgentName.TRAINING),
        ("Benchmark Report", "benchmark_report", AgentName.BENCHMARK),
        ("Deployment Artifact", "deployment_artifact", AgentName.HARDWARE),
    ]
    for label, attr, agent_name in _output_cards:
        identity = get_identity(agent_name)
        output = getattr(run, attr)
        with st.container(border=True):
            head = f"{identity['icon']} **{label}** &nbsp;·&nbsp; _{identity['display_name']}_"
            if output is None:
                st.markdown(head + " &nbsp;·&nbsp; :gray[pending]")
            else:
                st.markdown(head)
                st.json(output.model_dump(mode="json"), expanded=False)
