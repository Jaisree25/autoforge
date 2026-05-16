"""Custom CSS injected once at app load.

CSS variables inside `@keyframes` (e.g. `box-shadow: ... var(--c, #fff)`) are
stripped by Streamlit's HTML sanitizer in some configurations, so every pulse
animation here uses a hardcoded color. We generate one keyframe per agent
color and one matching class per color/state combination. This is what makes
the agent-card and pipeline-node pulses actually fire.
"""

# Single source of truth — keep in sync with agent_identity.AGENT_IDENTITIES.
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


def _per_color_keyframes() -> str:
    """Emit `@keyframes pulse-<name>` + `@keyframes pulse-strong-<name>` per color."""
    blocks: list[str] = []
    for name, hex_ in _COLORS.items():
        blocks.append(
            f"@keyframes pulse-{name} {{"
            f"  0%   {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"  50%  {{ box-shadow: 0 0 22px 4px {hex_}; }}"
            f"  100% {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"}}"
            f"@keyframes pulse-strong-{name} {{"
            f"  0%   {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"  50%  {{ box-shadow: 0 0 32px 6px {hex_}; }}"
            f"  100% {{ box-shadow: 0 0 0px 0px {hex_}; }}"
            f"}}"
            f".agent-card-working-{name} {{"
            f"  animation: pulse-{name} 1.4s ease-in-out infinite;"
            f"  border-color: {hex_} !important;"
            f"}}"
            f".agent-card-waiting-{name} {{"
            f"  animation: pulse-strong-{name} 1.6s ease-in-out infinite;"
            f"  border-color: {hex_} !important;"
            f"}}"
        )
    return "\n".join(blocks)


PULSE_CSS = f"""
<style>
/* ===========================================================
   Keyframes — per-agent-color so we never rely on CSS variables
   inside @keyframes (Streamlit's sanitizer strips them)
   =========================================================== */
{_per_color_keyframes()}

@keyframes thinking-dots {{
    0%, 20%   {{ content: ''; }}
    40%       {{ content: '.'; }}
    60%       {{ content: '..'; }}
    80%, 100% {{ content: '...'; }}
}}

/* The orange approval-pulse already worked; kept as-is for continuity */
@keyframes pulse-glow-strong {{
    0%   {{ box-shadow: 0 0 0px 0px #F59E0B; }}
    50%  {{ box-shadow: 0 0 26px 5px #F59E0B; }}
    100% {{ box-shadow: 0 0 0px 0px #F59E0B; }}
}}

/* ===========================================================
   Status pills (tiny badges)
   =========================================================== */
.status-badge {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.status-idle    {{ background: #2a2a2a; color: #b0b0b0; }}
.status-working {{ background: rgba(79,142,247,0.18); color: #82B1FF; }}
.status-waiting {{ background: rgba(245,158,11,0.22); color: #FBBF24; }}
.status-done    {{ background: rgba(16,185,129,0.20); color: #6EE7B7; }}
.status-error   {{ background: rgba(239,68,68,0.22);  color: #FCA5A5; }}

/* ===========================================================
   Agent roster cards — base styling; per-color animation classes
   above add the pulse + border color.
   =========================================================== */
.agent-card {{
    border-radius: 10px;
    padding: 8px 10px;
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 4px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
}}

.thinking-dots::after {{
    content: '';
    animation: thinking-dots 1.4s steps(4, end) infinite;
}}

/* ===========================================================
   Approval panel pulse — orange "look at me"
   =========================================================== */
.approval-pulse {{
    animation: pulse-glow-strong 1.6s ease-in-out infinite;
    border-radius: 12px;
}}

/* ===========================================================
   Panel header — colored strip above each top-row tile
   =========================================================== */
.panel-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    margin: -8px -8px 8px -8px;
    border-radius: 8px 8px 0 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border-bottom: 1px solid rgba(255,255,255,0.05);
}}
.panel-header-title {{
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}}
.panel-header-sub {{
    color: #888;
    font-size: 0.78rem;
}}
.panel-accent-blue   {{ border-top: 3px solid #4F8EF7; }}
.panel-accent-purple {{ border-top: 3px solid #A855F7; }}
.panel-accent-amber  {{ border-top: 3px solid #F59E0B; }}
.panel-accent-green  {{ border-top: 3px solid #10B981; }}
.panel-accent-red    {{ border-top: 3px solid #EF4444; }}
</style>
"""
