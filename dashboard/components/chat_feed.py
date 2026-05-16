"""Agent communications — Slack-style chat feed.

Each `AgentEvent` is rendered as one `st.chat_message` with the agent's
display name + icon avatar + accent color. Event types translate to natural-
sounding messages so the feed reads like a real team channel:

  Data Manager  · 18:42:01
  ▶ Starting: profile test.csv

  Data Manager  · 18:42:02
  🔧 Calling tool: ydata_profiling.ProfileReport (stub)

  Data Manager  · 18:42:02
  ✅ Done: profile test.csv

  Director      · 18:42:03
  🔔 Handoff: Data Manager → Researcher
  _Profiled 10,000 rows × 12 columns. Target `churn` · task `binary_classification`._

Newest is at the bottom (chat convention). We cap to the last ~120 events
so very long runs don't paint a 10-minute-tall page.
"""
from __future__ import annotations

import html as _html

import streamlit as st

from contracts.messages import AgentEvent, EventType

from dashboard.agent_identity import get_identity


_RECENT_CAP = 120


def render_chat_feed(events: list[AgentEvent]) -> None:
    st.subheader("Agent communications")
    if not events:
        st.info("No events yet — agents haven't started.")
        return

    recent = events[-_RECENT_CAP:]
    if len(events) > _RECENT_CAP:
        st.caption(
            f"_Showing last {_RECENT_CAP} of {len(events)} events._"
        )

    # Fixed-height scrollable region so the page doesn't grow with the run.
    # Streamlit handles the scrollbar natively. New events land at the bottom;
    # user can scroll up to see history.
    with st.container(height=560, border=True):
        for ev in recent:
            identity = get_identity(ev.agent)
            with st.chat_message(name=identity["display_name"], avatar=identity["icon"]):
                ts = ev.created_at.strftime("%H:%M:%S")
                st.markdown(
                    f"<div style='line-height:1.2; margin-bottom:2px;'>"
                    f"<span style='color:{identity['color']}; font-weight:700;'>"
                    f"{_html.escape(identity['display_name'])}</span>"
                    f"<span style='color:#777; font-size:0.78rem;'> · "
                    f"{_html.escape(identity['role'])} · {ts}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(_format_event(ev), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Event → natural language
# ---------------------------------------------------------------------------
def _format_event(ev: AgentEvent) -> str:
    et = ev.event_type
    msg = ev.message or ""

    if et == EventType.STARTED:
        return f"▶ _Starting:_ {msg}"

    if et == EventType.THINKING:
        return f"💭 _{msg}_"

    if et == EventType.TOOL_CALL:
        return f"🔧 Calling tool: `{msg}`"

    if et == EventType.INFO:
        return msg or "_no message_"

    if et == EventType.WARNING:
        return f"⚠ **Warning:** {msg}"

    if et == EventType.ERROR:
        return f"❌ **Error:** {msg}"

    if et == EventType.COMPLETED:
        return f"✅ _Done:_ {msg}"

    if et == EventType.APPROVAL_REQUESTED:
        summary = ev.payload.get("summary", "")
        next_agent = ev.payload.get("next_agent", "")
        head = f"🔔 **{msg}**" if msg else "🔔 **Asking for approval**"
        body = ""
        if summary:
            body += f"\n\n> {summary}"
        if next_agent:
            body += f"\n\n_Next: {next_agent}_"
        return head + body

    if et == EventType.APPROVAL_RECEIVED:
        decision = ev.payload.get("decision", "")
        if decision:
            return f"📩 _Got_ **{decision}** _— proceeding._" + (
                f"\n\n_{msg}_" if msg else ""
            )
        return f"📩 _Got approval._ {msg}"

    return msg or "_(no message)_"
