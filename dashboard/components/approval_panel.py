"""Approval panel — pending HITL handoff gates.

Each approval request represents a handoff from one agent to the next.
Default payload shape:

    {
        "kind":         "default" | "candidate_pick",
        "summary":      "<one-sentence what just happened>",
        "next_agent":   "<display name of who's up next>",
        "agent_output": <full Pydantic dump for inspection / editing>,
    }

When `kind == "candidate_pick"`, the payload also carries:

    {
        "candidates": [{index, name, family, library, rationale, ...}, ...],
        "default_index": 0,
    }

…and the panel renders a radio picker instead of edit-JSON. The chosen
index comes back in `ApprovalResponse.response_payload["selected_index"]`,
and the Coordinator trims `candidate_architectures` to just that one.
"""
from __future__ import annotations

import html as _html
import json

import streamlit as st

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from memory.store import MemoryStore

from dashboard.agent_identity import get_identity


def render_approval_panel(store: MemoryStore, run_id: str) -> None:
    pending = store.list_pending_approvals(run_id)
    if not pending:
        st.success("No pending approvals — pipeline running autonomously.")
        return

    for req in pending:
        if req.payload.get("kind") == "candidate_pick":
            _render_candidate_pick(store, req)
        else:
            _render_default(store, req)


# ---------------------------------------------------------------------------
# Candidate-pick gate (Researcher → Preparer)
# ---------------------------------------------------------------------------
def _render_candidate_pick(store: MemoryStore, req: ApprovalRequest) -> None:
    identity = get_identity(req.agent)
    summary = req.payload.get("summary", "")
    next_agent = req.payload.get("next_agent", "next agent")
    candidates: list[dict] = req.payload.get("candidates", []) or []
    default_index: int = int(req.payload.get("default_index", 0) or 0)

    # Orange-pulse header — same look as other gates so it stands out
    st.markdown(
        f'<div class="approval-pulse" style="'
        f'border: 1px solid rgba(245,158,11,0.6); '
        f'border-radius: 12px; '
        f'padding: 12px 14px; '
        f'background: rgba(245,158,11,0.06);">'
        f"<div style='display:flex; align-items:center; gap:8px;'>"
        f"<span style='font-size:1.3rem;'>{identity['icon']}</span>"
        f"<span style='color:{identity['color']}; font-weight:700;'>"
        f"{_html.escape(identity['display_name'])}</span>"
        f"<span style='color:#bbb; font-size:0.85rem;'>"
        f"&nbsp;→&nbsp;</span>"
        f"<span style='color:#FBBF24; font-weight:700;'>"
        f"{_html.escape(next_agent)}</span>"
        f"</div>"
        f"<div style='color:#e5e5e5; font-size:0.9rem; margin-top:8px; "
        f"line-height:1.35;'>{_html.escape(summary)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not candidates:
        st.error("Researcher returned no candidates.")
        if st.button("Reject (no choice possible)", key=f"reject::{req.request_id}",
                     use_container_width=True):
            store.respond_to_approval(ApprovalResponse(
                request_id=req.request_id,
                decision=ApprovalDecision.REJECTED,
                responder="dashboard",
                comment="no candidates to choose from",
            ))
            st.rerun()
        return

    # Radio picker — one option per candidate
    radio_key = f"candidate_radio::{req.request_id}"
    options = list(range(len(candidates)))
    default_idx = max(0, min(default_index, len(candidates) - 1))

    def _fmt(i: int) -> str:
        c = candidates[i]
        return f"{c.get('name', '?')}  ·  {c.get('family', '?')} / {c.get('library', '?')}"

    chosen = st.radio(
        "Choose the candidate the Trainer should commit to:",
        options=options,
        format_func=_fmt,
        index=default_idx,
        key=radio_key,
    )

    # Rationale + hyperparam preview for the currently-selected candidate
    sel = candidates[chosen]
    rationale = sel.get("rationale", "") or "_no rationale_"
    hp = sel.get("hyperparameter_space") or {}
    with st.container(border=True):
        st.markdown(f"**Rationale:** {rationale}")
        if hp:
            st.markdown("**Hyperparameter space:**")
            st.json(hp, expanded=False)

    with st.expander("Inspect full StrategySpec"):
        st.json(req.payload.get("agent_output", {}), expanded=False)

    c1, c2 = st.columns([1, 1])
    if c1.button("✅ Approve selection", key=f"approve::{req.request_id}",
                 type="primary", use_container_width=True):
        store.respond_to_approval(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.APPROVED,
            response_payload={"selected_index": int(chosen)},
            responder="dashboard",
            comment=f"picked candidate #{int(chosen) + 1}: {sel.get('name', '?')}",
        ))
        st.rerun()
    if c2.button("❌ Reject all", key=f"reject::{req.request_id}",
                 use_container_width=True):
        store.respond_to_approval(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.REJECTED,
            responder="dashboard",
            comment="rejected via dashboard",
        ))
        st.rerun()


# ---------------------------------------------------------------------------
# Default gate (every other handoff)
# ---------------------------------------------------------------------------
def _render_default(store: MemoryStore, req: ApprovalRequest) -> None:
    identity = get_identity(req.agent)
    edit_mode = st.session_state.get("edit_request_id") == req.request_id
    edit_key = f"edit_text::{req.request_id}"

    summary = req.payload.get("summary", req.description or "")
    next_agent = req.payload.get("next_agent", "next agent")
    agent_output_payload = req.payload.get("agent_output", req.payload)

    pulse_class = "" if edit_mode else "approval-pulse"
    st.markdown(
        f'<div class="{pulse_class}" style="'
        f'border: 1px solid rgba(245,158,11,0.6); '
        f'border-radius: 12px; '
        f'padding: 12px 14px; '
        f'background: rgba(245,158,11,0.06);">'
        f"<div style='display:flex; align-items:center; gap:8px;'>"
        f"<span style='font-size:1.3rem;'>{identity['icon']}</span>"
        f"<span style='color:{identity['color']}; font-weight:700;'>"
        f"{_html.escape(identity['display_name'])}</span>"
        f"<span style='color:#bbb; font-size:0.85rem;'>"
        f"&nbsp;→&nbsp;</span>"
        f"<span style='color:#FBBF24; font-weight:700;'>"
        f"{_html.escape(next_agent)}</span>"
        f"</div>"
        f"<div style='color:#e5e5e5; font-size:0.9rem; margin-top:8px; "
        f"line-height:1.35;'>{summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if edit_mode:
        if edit_key not in st.session_state:
            st.session_state[edit_key] = json.dumps(agent_output_payload, indent=2)
        st.text_area("Edit agent output (JSON)", key=edit_key, height=320)
        c1, c2 = st.columns([1, 1])
        if c1.button("Submit edit", key=f"submit::{req.request_id}",
                     type="primary", use_container_width=True):
            try:
                parsed = json.loads(st.session_state[edit_key])
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                store.respond_to_approval(ApprovalResponse(
                    request_id=req.request_id,
                    decision=ApprovalDecision.EDITED,
                    response_payload={"agent_output": parsed},
                    responder="dashboard",
                    comment="edited via dashboard",
                ))
                st.session_state.pop("edit_request_id", None)
                st.session_state.pop(edit_key, None)
                st.rerun()
        if c2.button("Cancel", key=f"cancel::{req.request_id}",
                     use_container_width=True):
            st.session_state.pop("edit_request_id", None)
            st.session_state.pop(edit_key, None)
            st.rerun()
    else:
        with st.expander("Inspect handoff payload"):
            st.json(agent_output_payload, expanded=False)

        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("✅ Approve", key=f"approve::{req.request_id}",
                     type="primary", use_container_width=True):
            store.respond_to_approval(ApprovalResponse(
                request_id=req.request_id,
                decision=ApprovalDecision.APPROVED,
                responder="dashboard",
                comment="approved via dashboard",
            ))
            st.rerun()
        if c2.button("📝 Edit", key=f"edit::{req.request_id}",
                     use_container_width=True):
            st.session_state["edit_request_id"] = req.request_id
            st.rerun()
        if c3.button("❌ Reject", key=f"reject::{req.request_id}",
                     use_container_width=True):
            store.respond_to_approval(ApprovalResponse(
                request_id=req.request_id,
                decision=ApprovalDecision.REJECTED,
                responder="dashboard",
                comment="rejected via dashboard",
            ))
            st.rerun()
        st.caption("You can also respond via Telegram (if configured).")
