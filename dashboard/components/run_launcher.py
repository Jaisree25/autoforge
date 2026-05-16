"""Sidebar component: upload a dataset, type an objective, kick off a pipeline.

Launches `scripts/run_pipeline.py` as a detached subprocess so the dashboard
doesn't block. The subprocess writes events + outputs to the same SQLite store
the dashboard polls — the new run appears in the run picker on the next
autorefresh.

Notes:
  - For the skeleton, we accept CSVs only via `st.file_uploader`. Image dirs
    are out of scope until the Dataset Agent supports them.
  - We always pass `--auto-approve` is NOT used here; the user is expected to
    approve via the dashboard approval panel (matches the demo loop).
  - If `NVIDIA_API_KEY` isn't set, the run still works because every agent
    is currently a stub. The launcher does NOT validate env state.
"""
from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from config import PROJECT_ROOT, UPLOADS_DIR


def render_run_launcher() -> None:
    """Render the 'Start a new run' panel in the sidebar."""
    with st.sidebar.expander("▶ Start a new run", expanded=False):
        st.caption(
            "Use one of the built-in fixtures or upload your own CSV. "
            "The image fixture (`data/fixtures/mnist/`) is the recommended demo path."
        )
        preset = st.radio(
            "Dataset",
            options=["MNIST (images)", "Churn (CSV)", "Upload your own CSV"],
            index=0,
            key="launcher_preset",
            horizontal=False,
        )
        uploaded = None
        if preset == "Upload your own CSV":
            uploaded = st.file_uploader(
                "Pick a CSV file",
                type=["csv"],
                key="launcher_dataset",
                accept_multiple_files=False,
            )

        default_obj = (
            "classify handwritten digits (MNIST) with accuracy >= 0.95"
            if preset == "MNIST (images)"
            else "predict customer churn with F1 >= 0.85"
        )
        objective = st.text_input(
            "Objective",
            value=default_obj,
            key="launcher_objective",
            help="Plain-English goal. The Researcher will parse the metric + threshold.",
        )
        auto_approve = st.checkbox(
            "Auto-approve HITL gates",
            value=False,
            key="launcher_auto_approve",
            help="Skip the approval panel. Useful for quick smoke runs.",
        )
        start = st.button(
            "🚀 Launch",
            key="launcher_start",
            type="primary",
            use_container_width=True,
        )

        if start:
            if not objective.strip():
                st.error("Type an objective.")
                return

            # Resolve dataset path based on preset
            if preset == "MNIST (images)":
                target = PROJECT_ROOT / "data" / "fixtures" / "mnist"
                if not target.exists():
                    st.error(
                        f"MNIST fixture missing at {target}. "
                        "Run `python scripts/create_mnist_fixture.py` first."
                    )
                    return
            elif preset == "Churn (CSV)":
                target = PROJECT_ROOT / "data" / "fixtures" / "churn_sample.csv"
                if not target.exists():
                    st.error(
                        f"Churn fixture missing at {target}. "
                        "Run `python scripts/create_fixtures.py` first."
                    )
                    return
            else:
                # Upload path
                if uploaded is None:
                    st.error("Pick a CSV file first.")
                    return
                safe_name = f"{uuid.uuid4().hex[:8]}_{uploaded.name}"
                target = UPLOADS_DIR / safe_name
                target.write_bytes(uploaded.getvalue())

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                "--dataset", str(target),
                "--objective", objective.strip(),
            ]
            if auto_approve:
                cmd.append("--auto-approve")

            # Detach the subprocess so the dashboard isn't blocked.
            # Output goes to per-run log files under data/artifacts/runs/.
            log_dir = PROJECT_ROOT / "data" / "artifacts" / "runs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"launch_{uuid.uuid4().hex[:8]}.log"

            try:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        creationflags=(
                            subprocess.CREATE_NEW_PROCESS_GROUP
                            if sys.platform == "win32" else 0
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Launch failed: {exc}")
                return

            # Tell run_picker to jump to the new run when it appears.
            st.session_state["launcher_waiting_since"] = datetime.now(timezone.utc)

            st.success(
                f"Run launched against `{target.name}`. "
                f"Dashboard will jump to the new run automatically."
            )
            st.caption(f"Subprocess output: `{log_path.relative_to(PROJECT_ROOT)}`")
