"""Animated pipeline flow — rendered as a self-contained HTML iframe.

We use `streamlit.components.v1.html` (not `st.markdown`) because Streamlit's
markdown sanitizer is unreliable for `<style>` blocks with `@keyframes`
across versions. An iframe with embedded CSS guarantees the animations fire.

Iframe content can't reference page-level CSS variables, so each agent color
gets its own concrete keyframe inside this file's CSS. Same pattern as
`dashboard.styles` for the roster cards.
"""
from __future__ import annotations

import streamlit.components.v1 as components

from contracts.messages import AgentEvent

from dashboard.agent_identity import (
    AGENT_DISPLAY_ORDER,
    AgentStatus,
    agent_status_from_events,
    get_identity,
)


_MARKER: dict[str, str] = {
    AgentStatus.IDLE:    "○",
    AgentStatus.WORKING: "⚡",
    AgentStatus.WAITING: "⏸",
    AgentStatus.DONE:    "✓",
    AgentStatus.ERROR:   "✗",
}

# Match the registry in agent_identity.py
_COLORS: dict[str, str] = {
    "cyan":   "#06B6D4",
    "blue":   "#4F8EF7",
    "purple": "#A855F7",
    "amber":  "#F59E0B",
    "green":  "#10B981",
    "red":    "#EF4444",
    "indigo": "#6366F1",
    "gray":   "#888888",
}


def _per_color_css() -> str:
    """One @keyframes + one .node-working-* class per agent color."""
    blocks: list[str] = []
    for name, hex_ in _COLORS.items():
        blocks.append(
            f"@keyframes pulse-{name} {{"
            f"  0%   {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"  50%  {{ box-shadow: 0 0 26px 6px {hex_}; }}"
            f"  100% {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"}}"
            f".node-working-{name} {{"
            f"  transform: scale(1.10);"
            f"  border-color: {hex_};"
            f"  animation: pulse-{name} 1.3s ease-in-out infinite;"
            f"}}"
            f".node-waiting-{name} {{"
            f"  border-color: {hex_};"
            f"  animation: pulse-{name} 1.6s ease-in-out infinite;"
            f"}}"
        )
    return "\n".join(blocks)


_PAGE_CSS = f"""
html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    color: #ddd;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
/* Horizontal scroll if the strip doesn't fit; vertical clip is fine. */
body {{ overflow-x: auto; overflow-y: hidden; }}

{_per_color_css()}

@keyframes flow {{
    from {{ background-position: 100% 0; }}
    to   {{ background-position: -100% 0; }}
}}

.flow-row {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding: 26px 12px 18px 12px;
    gap: 0;
    width: max-content;          /* lets row grow past viewport → scrollbar */
    min-width: 100%;
}}

.node {{
    flex: 0 0 auto;
    width: 108px;
    min-height: 112px;
    text-align: center;
    padding: 10px 5px 8px 5px;
    border-radius: 14px;
    border: 2px solid #3a3a3a;
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    box-sizing: border-box;
    transition: transform 0.25s ease, border-color 0.25s ease;
    overflow: hidden;
}}
.node-idle {{ opacity: 0.5; }}
.node-done {{ border-color: #10B981; }}

.node .icon   {{ font-size: 1.7rem; line-height: 1; }}
.node .name   {{
    font-size: 0.80rem; font-weight: 700; margin-top: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    padding: 0 2px;
}}
.node .role   {{
    font-size: 0.62rem; color: #999; margin-top: 2px; line-height: 1.15;
    /* Allow up to 2 lines, then clip. Long roles still fit. */
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    padding: 0 2px;
}}
.node .marker {{ font-size: 1.0rem; margin-top: 4px; color: #ccc; }}

.edge {{
    flex: 1 1 auto;
    height: 6px;
    margin: 0 5px;
    background: #3a3a3a;
    border-radius: 3px;
    min-width: 18px;
    max-width: 60px;
}}
.edge-done {{
    background: #10B981;
}}
.edge-active {{
    background: linear-gradient(90deg,
        rgba(79,142,247,0)    0%,
        rgba(79,142,247,1.0)  45%,
        rgba(79,142,247,1.0)  55%,
        rgba(79,142,247,0)    100%);
    background-size: 200% 100%;
    animation: flow 1.0s linear infinite;
}}
"""


def _node_class(status: str, color_name: str) -> str:
    base = "node"
    if status == AgentStatus.WORKING:
        return f"{base} node-working-{color_name}"
    if status == AgentStatus.WAITING:
        return f"{base} node-waiting-{color_name}"
    if status == AgentStatus.DONE:
        return f"{base} node-done"
    if status == AgentStatus.ERROR:
        return f"{base} node-working-red"  # error = pulse-red border
    return f"{base} node-idle"


def _node_html(identity: dict, status: str) -> str:
    cls = _node_class(status, identity["color_name"])
    color = identity["color"]
    return (
        f'<div class="{cls}">'
        f'  <div class="icon">{identity["icon"]}</div>'
        f'  <div class="name" style="color:{color};">{identity["display_name"]}</div>'
        f'  <div class="role">{identity["role"]}</div>'
        f'  <div class="marker">{_MARKER[status]}</div>'
        f'</div>'
    )


def _edge_html(prev_status: str, next_status: str) -> str:
    cls = "edge"
    if prev_status == AgentStatus.DONE and next_status in (
        AgentStatus.WORKING, AgentStatus.WAITING
    ):
        cls += " edge-active"
    elif prev_status == AgentStatus.DONE and next_status == AgentStatus.DONE:
        cls += " edge-done"
    return f'<div class="{cls}"></div>'


def render_network_graph(events: list[AgentEvent]) -> None:
    statuses = [
        (av, agent_status_from_events(events, av))
        for av in AGENT_DISPLAY_ORDER
    ]
    inner: list[str] = []
    for i, (agent_value, status) in enumerate(statuses):
        identity = get_identity(agent_value)
        inner.append(_node_html(identity, status))
        if i < len(statuses) - 1:
            next_status = statuses[i + 1][1]
            inner.append(_edge_html(status, next_status))

    html_doc = (
        "<!DOCTYPE html><html><head>"
        f"<style>{_PAGE_CSS}</style>"
        "</head><body>"
        f'<div class="flow-row">{"".join(inner)}</div>'
        "</body></html>"
    )
    # Height needs headroom for the glow halo so it isn't clipped at the
    # iframe edge. `scrolling=True` permits horizontal scroll if the strip
    # is wider than the column.
    components.html(html_doc, height=200, scrolling=True)
