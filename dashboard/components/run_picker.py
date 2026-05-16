"""Sidebar component: pick a run + show its status + delete it.

Two UX behaviors worth knowing:

  1. **Auto-jump to newest.** When the launcher fires, it sets
     `session_state["launcher_waiting_since"]` to the click timestamp. On
     each autorefresh tick, the picker checks whether any run was created
     after that timestamp. If yes, it forces selection to the newest run
     and clears the flag. Net effect: clicking Launch hops you to the new
     run as soon as the subprocess has registered it (~1-2 autorefresh ticks).

  2. **Delete this run.** A 🗑 button under the picker hard-deletes the
     selected run (events / outputs / approvals cascade). Cheap way to
     clean the dropdown without nuking the DB.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from contracts.schemas import PipelineRun
from memory.store import MemoryStore


def _fmt_label(r: dict) -> str:
    short = r["run_id"][:8]
    obj = r["objective"][:50]
    return f"{short} · {r['status']} · {obj}"


def render_run_picker(store: MemoryStore) -> PipelineRun | None:
    """Render the sidebar run picker. Returns the selected hydrated `PipelineRun`."""
    st.sidebar.title("🔨 AutoForge")
    st.sidebar.caption("Multi-agent autonomous ML pipeline")

    runs = store.list_runs(limit=50)
    if not runs:
        st.sidebar.info("No runs yet")
        return None

    options = {_fmt_label(r): r["run_id"] for r in runs}
    labels = list(options.keys())

    # --- Auto-jump to newest if the launcher fired recently ---------------
    wait_since: datetime | None = st.session_state.get("launcher_waiting_since")
    if wait_since is not None:
        for r in runs:
            try:
                created = datetime.fromisoformat(r["created_at"])
            except Exception:
                continue
            if created > wait_since:
                # New run found — force select it and clear the flag
                st.session_state["run_picker"] = _fmt_label(r)
                st.session_state.pop("launcher_waiting_since", None)
                break

    selected_label = st.sidebar.selectbox(
        "Run",
        options=labels,
        index=0,
        key="run_picker",
    )
    selected_run_id = options[selected_label]
    run = store.get_run(selected_run_id)
    if run is None:
        return None

    st.sidebar.divider()
    st.sidebar.markdown(f"**Status** &nbsp; `{run.status.value}`")
    st.sidebar.markdown(f"**Objective** &nbsp; {run.objective}")
    st.sidebar.markdown(f"**Dataset** &nbsp; `{run.dataset_path}`")
    st.sidebar.caption(f"Created: {run.created_at:%Y-%m-%d %H:%M:%S} UTC")
    st.sidebar.caption(f"Updated: {run.updated_at:%Y-%m-%d %H:%M:%S} UTC")
    if run.error:
        st.sidebar.error(run.error)

    # --- Per-run delete (with one-click confirm) --------------------------
    confirm_key = f"confirm_delete::{run.run_id}"
    if st.session_state.get(confirm_key):
        st.sidebar.warning("Delete this run?")
        c1, c2 = st.sidebar.columns([1, 1])
        if c1.button("Yes, delete", key=f"delete_yes::{run.run_id}",
                     type="primary", use_container_width=True):
            store.delete_run(run.run_id)
            # Clear all session state tied to this run
            st.session_state.pop(confirm_key, None)
            st.session_state.pop("run_picker", None)
            st.session_state.pop("focused_agent", None)
            st.rerun()
        if c2.button("Cancel", key=f"delete_no::{run.run_id}",
                     use_container_width=True):
            st.session_state.pop(confirm_key, None)
            st.rerun()
    else:
        if st.sidebar.button("🗑 Delete this run", key=f"delete::{run.run_id}",
                             use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()

    return run
