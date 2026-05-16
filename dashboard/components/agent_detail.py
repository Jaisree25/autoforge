"""Per-agent detail view — rendered in place of the pipeline flow.

When a roster card's "View work" button is clicked, the center column
swaps from the pipeline flow to this component. Clicking "Close" pops
`focused_agent` and the pipeline reappears.

The center column is ~55% of page width, so the layout is stacked
vertically (no inner columns) and the recent activity log lives in an
expander at the bottom to save space.

Per-agent specialized content:
  - Data Manager        → columns table, class balance, warnings
  - Researcher          → candidate architectures + clickable arxiv citations
  - Hardware Specialist → envelope metrics + deployment artifact card
  - Trainer             → trial-score chart + best params + all trials table
  - Evaluator           → big PASS/FAIL + accuracy/latency metrics + Pareto
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from config import ARTIFACTS_DIR
from contracts.messages import AgentEvent, EventType
from contracts.schemas import (
    AgentName,
    BenchmarkReport,
    DatasetProfile,
    DeploymentArtifact,
    Modality,
    PipelineRun,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
    TrainingResult,
)

from dashboard.agent_identity import (
    AgentStatus,
    agent_status_from_events,
    get_identity,
)


_STATUS_BADGE: dict[str, tuple[str, str]] = {
    AgentStatus.IDLE:    ("idle",    "status-idle"),
    AgentStatus.WORKING: ("working", "status-working"),
    AgentStatus.WAITING: ("waiting", "status-waiting"),
    AgentStatus.DONE:    ("done",    "status-done"),
    AgentStatus.ERROR:   ("error",   "status-error"),
}

_EVENT_ICON: dict[EventType, str] = {
    EventType.STARTED: "▶",
    EventType.THINKING: "💭",
    EventType.TOOL_CALL: "🔧",
    EventType.INFO: "ℹ",
    EventType.WARNING: "⚠",
    EventType.ERROR: "✗",
    EventType.COMPLETED: "✓",
    EventType.APPROVAL_REQUESTED: "🔔",
    EventType.APPROVAL_RECEIVED: "📩",
}


def render_agent_detail(
    run: PipelineRun,
    events: list[AgentEvent],
    agent_value: str,
) -> None:
    """Render the focused-agent detail in place of the pipeline flow.

    Designed to fit the ~55%-wide center column. Vertically stacked; no inner
    columns (would be too cramped).
    """
    identity = get_identity(agent_value)
    status = agent_status_from_events(events, agent_value)
    agent_events = [e for e in events if e.agent.value == agent_value]
    badge_label, badge_class = _STATUS_BADGE[status]

    # --- Header bar with close button -------------------------------------
    head_col, close_col = st.columns([5, 1])
    with head_col:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px;'>"
            f"<span style='font-size:1.9rem;'>{identity['icon']}</span>"
            f"<div>"
            f"<div style='color:{identity['color']}; font-weight:800; "
            f"font-size:1.3rem; line-height:1.1;'>{identity['display_name']}</div>"
            f"<div style='color:#aaa; font-size:0.82rem;'>"
            f"{identity['role']} &nbsp;·&nbsp; via {identity['model']}</div>"
            f"</div>"
            f"<span class='status-badge {badge_class}' "
            f"style='margin-left:auto;'>{badge_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with close_col:
        if st.button("✕ Back", key="close_focus", use_container_width=True,
                     type="primary"):
            st.session_state.pop("focused_agent", None)
            st.rerun()

    st.divider()

    # --- Specialized view -------------------------------------------------
    _render_specialized(run, agent_value)

    # --- Recent activity for this agent (expander) ------------------------
    with st.expander(
        f"Recent activity · {len(agent_events)} events",
        expanded=False,
    ):
        if not agent_events:
            st.caption("No events yet.")
        else:
            with st.container(height=320, border=True):
                for ev in reversed(agent_events):
                    ts = ev.created_at.strftime("%H:%M:%S")
                    icon = _EVENT_ICON.get(ev.event_type, "•")
                    st.markdown(
                        f"`{ts}` {icon} _{ev.event_type.value}_ — "
                        f"{ev.message or ''}"
                    )


# ---------------------------------------------------------------------------
# Specialized per-agent renderers
# ---------------------------------------------------------------------------
def _render_specialized(run: PipelineRun, agent_value: str) -> None:
    if agent_value == AgentName.PROFILER.value:
        _profiler_view(run.dataset_profile, run.training_envelope)
    elif agent_value == AgentName.STRATEGY.value:
        _researcher_view(run.strategy_spec)
    elif agent_value == AgentName.DATASET.value:
        _preparer_view(run.preparation_report, run.dataset_profile)
    elif agent_value == AgentName.HARDWARE.value:
        _optimizer_view(run.deployment_artifact)
    elif agent_value == AgentName.TRAINING.value:
        _trainer_view(run.training_result, run.run_id)
    elif agent_value == AgentName.BENCHMARK.value:
        _evaluator_view(run.benchmark_report)
    else:
        st.caption("No specialized view for this agent.")


# ---------- Profiler ----------
def _profiler_view(
    profile: DatasetProfile | None,
    envelope: TrainingEnvelope | None,
) -> None:
    if profile is None and envelope is None:
        st.info("Profiler hasn't produced output yet.")
        return

    if profile is not None:
        modality = profile.modality
        st.markdown(
            f"**Dataset observation** &nbsp;·&nbsp; "
            f":blue[modality: `{modality.value}`]"
        )

        # Top metrics row differs by modality
        if modality == Modality.IMAGE:
            m1, m2, m3 = st.columns(3)
            m1.metric("Images", f"{profile.n_rows:,}")
            m2.metric("Classes", profile.n_classes if profile.n_classes is not None else "—")
            m3.metric("Channels", profile.image_channels or "—")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{profile.n_rows:,}")
            m2.metric("Columns", profile.n_cols)
            m3.metric("Target", profile.target_column or "—")

        st.markdown(f"**Task** &nbsp;·&nbsp; `{profile.task_type.value}`")
        if profile.profile_summary:
            st.caption(profile.profile_summary)
        if profile.class_balance:
            st.markdown("**Class balance**")
            st.bar_chart(profile.class_balance, horizontal=True, height=120)

        # Tabular-specific: columns table
        if modality != Modality.IMAGE and profile.columns:
            with st.expander(f"Columns ({len(profile.columns)})"):
                rows = [
                    {
                        "name": c.name,
                        "dtype": c.dtype,
                        "missing %": f"{c.missing_pct * 100:.2f}%",
                        "unique": c.unique_count if c.unique_count is not None else "—",
                    }
                    for c in profile.columns
                ]
                st.dataframe(rows, hide_index=True, use_container_width=True)

        # Image-specific: resolutions + formats
        if modality == Modality.IMAGE:
            if profile.image_resolutions:
                with st.expander(
                    f"Sample resolutions ({len(profile.image_resolutions)})"
                ):
                    res_rows = [
                        {"width": w, "height": h}
                        for w, h in profile.image_resolutions
                    ]
                    st.dataframe(res_rows, hide_index=True, use_container_width=True)
            if profile.image_formats:
                st.markdown(
                    "**Formats:** "
                    + ", ".join(f"`{f}`" for f in profile.image_formats)
                )

        if profile.warnings:
            st.markdown("**Warnings (for Preparer)**")
            for w in profile.warnings:
                st.warning(w)

    if envelope is not None:
        st.divider()
        st.markdown("**Hardware envelope**")
        m1, m2, m3 = st.columns(3)
        m1.metric("GPU", envelope.gpu_name or "—")
        m2.metric("VRAM", f"{envelope.gpu_memory_gb or 0:.0f} GB")
        m3.metric("Max trials", envelope.max_trials)
        m4, m5, m6 = st.columns(3)
        m4.metric("CPUs", envelope.cpu_count)
        m5.metric("RAM", f"{envelope.system_memory_gb:.0f} GB")
        m6.metric("Max min", f"{envelope.max_train_minutes:.1f}")
        st.markdown(
            "**Allowed libraries:** "
            + ", ".join(f"`{lib}`" for lib in envelope.allowed_libraries)
        )
        if envelope.notes:
            st.caption(envelope.notes)

    with st.expander("Raw outputs (JSON)"):
        if profile is not None:
            st.markdown("_DatasetProfile:_")
            st.json(profile.model_dump(mode="json"), expanded=False)
        if envelope is not None:
            st.markdown("_TrainingEnvelope:_")
            st.json(envelope.model_dump(mode="json"), expanded=False)


# ---------- Preparer (Data Preparer) ----------
def _preparer_view(
    report: PreparationReport | None,
    profile: DatasetProfile | None,
) -> None:
    if report is None:
        st.info("Data Preparer hasn't produced output yet.")
        return

    m1, m2 = st.columns(2)
    m1.metric("Operations applied", len(report.operations))
    m2.metric("Original rows", f"{profile.n_rows:,}" if profile else "—")

    st.markdown("**Original dataset**")
    st.code(report.original_dataset_path, language=None)
    if report.prepared_dataset_path:
        st.markdown("**Prepared dataset**")
        st.code(report.prepared_dataset_path, language=None)

    if report.operations:
        st.markdown(f"**Operations** &nbsp;·&nbsp; {len(report.operations)}")
        for op in report.operations:
            st.markdown(f"- `{op}`")

    if report.summary:
        st.caption(report.summary)
    if report.notes:
        st.info(report.notes)

    with st.expander("Raw output (JSON)"):
        st.json(report.model_dump(mode="json"), expanded=False)


# ---------- Optimizer (was Hardware post-pass) ----------
def _optimizer_view(artifact: DeploymentArtifact | None) -> None:
    if artifact is None:
        st.info("Optimizer hasn't produced output yet.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Format", artifact.format)
    m2.metric("Quantization", artifact.quantization or "—")
    m3.metric("Size", f"{artifact.size_mb:.1f} MB")

    st.markdown("**Artifact path**")
    st.code(artifact.artifact_path, language=None)

    if artifact.notes:
        st.caption(artifact.notes)

    with st.expander("Raw output (JSON)"):
        st.json(artifact.model_dump(mode="json"), expanded=False)


# ---------- Researcher ----------
def _researcher_view(spec: StrategySpec | None) -> None:
    if spec is None:
        st.info("Researcher hasn't produced output yet.")
        return

    m1, m2 = st.columns(2)
    m1.metric("Success metric", spec.success_metric.upper())
    m2.metric("Threshold", f"{spec.success_threshold:.2f}")
    st.markdown(f"**Objective** &nbsp;·&nbsp; _{spec.objective}_")

    if spec.research_summary:
        st.markdown("**Research summary**")
        st.write(spec.research_summary)

    if spec.candidate_architectures:
        st.markdown(f"**Candidate architectures** &nbsp;·&nbsp; {len(spec.candidate_architectures)}")
        for arch in spec.candidate_architectures:
            with st.container(border=True):
                st.markdown(
                    f"**`{arch.name}`** &nbsp;·&nbsp; "
                    f":gray[{arch.family}] / `{arch.library}`"
                )
                if arch.rationale:
                    st.caption(arch.rationale)
                if arch.hyperparameter_space:
                    with st.expander("Hyperparameter space"):
                        st.json(arch.hyperparameter_space, expanded=False)

    if spec.citations:
        st.markdown(f"**Citations** &nbsp;·&nbsp; {len(spec.citations)} papers")
        for c in spec.citations:
            with st.container(border=True):
                title_md = (
                    f"**[{c.title}]({c.url})**" if c.url else f"**{c.title}**"
                )
                st.markdown(f"📄 {title_md}")
                meta = f":gray[source: `{c.source}`]"
                if c.url:
                    meta += f" &nbsp;·&nbsp; [`{c.url}`]({c.url})"
                st.markdown(meta)
                if c.snippet:
                    st.markdown(f"> _{c.snippet}_")

    with st.expander("Raw output (JSON)"):
        st.json(spec.model_dump(mode="json"), expanded=False)


# ---------- Trainer ----------
def _trainer_view(result: TrainingResult | None, run_id: str) -> None:
    """Trainer detail view.

    Two modes:
      1. Final mode — `TrainingResult` is persisted (after subprocess training).
         Shows the metric, trial chart, etc.
      2. **In-progress mode** — Trainer is mid-run. Linear flow writes all
         artifacts into a flat `training/` directory:
             oracle.json                      — baseline accuracy
             training/{design.md, model.py, train.py, verify_report.json,
                       status.json, models/, logs/}
    """
    run_dir = ARTIFACTS_DIR / run_id

    # ---- In-progress: no TrainingResult yet, but artifacts may exist ----
    if result is None:
        artifacts_found = _render_trainer_in_progress(run_dir)
        if not artifacts_found:
            st.info("Trainer hasn't produced output yet.")
        return

    # ---- Final: TrainingResult is persisted ----
    # Focus: training PROCESS (iters, loss, time, HPs). The Evaluator owns
    # the accuracy/latency benchmark — don't headline accuracy here.
    # getattr-with-default handles old runs persisted before the
    # training_process field existed on TrainingResult.
    process = getattr(result, "training_process", None) or {}

    n_iter = process.get("n_iter")
    final_loss = process.get("final_loss")
    estimator = process.get("estimator_class") or result.library or "—"

    m1, m2, m3 = st.columns(3)
    m1.metric("Wall time", f"{result.training_time_seconds:.1f}s")
    m2.metric("Iterations", n_iter if n_iter is not None else "—")
    m3.metric(
        "Final loss",
        f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "—",
    )

    st.markdown(
        f"**Estimator** &nbsp;·&nbsp; `{estimator}` &nbsp;·&nbsp; "
        f"saved as `{result.best_model_id}.pkl`"
    )

    # --- Loss curve (the actual training progression) ---
    loss_curve = process.get("loss_curve") or []
    if loss_curve and len(loss_curve) >= 2:
        label = process.get("loss_curve_label", "loss")
        st.markdown(f"**Training {label} per iteration**")
        chart_data = {
            "iteration": list(range(1, len(loss_curve) + 1)),
            label: loss_curve,
        }
        st.line_chart(chart_data, x="iteration", y=label, height=220)

    # --- Effective hyperparameters (what sklearn actually ran with) ---
    effective = process.get("effective_params") or {}
    if effective:
        st.markdown("**Effective hyperparameters**")
        with st.container(border=True):
            shown = {k: v for k, v in effective.items() if not k.startswith("_")}
            rows = [{"param": k, "value": str(v)} for k, v in shown.items()]
            st.dataframe(rows, hide_index=True, use_container_width=True)
    elif result.best_params:
        # Fallback to best_params when introspection didn't fire
        st.markdown("**Hyperparameters**")
        with st.expander("View"):
            st.json(result.best_params, expanded=False)

    # --- Training data shape ---
    n_train = process.get("n_train")
    n_test = process.get("n_test")
    if n_train is not None or n_test is not None:
        st.caption(
            f"Trained on **{n_train}** samples, "
            f"validated on **{n_test}** samples. "
            f"(Accuracy + latency benchmark → Evaluator card.)"
        )

    if result.notes:
        st.caption(result.notes)

    # --- Generated code + design.md (the LLM artifacts the user asked for) ---
    # Same artifacts the in-progress view shows; surface them post-completion
    # so reviewers / judges can still inspect what Nemotron produced.
    _render_attempt_code_artifacts(run_dir)

    with st.expander("Raw output (JSON)"):
        st.json(result.model_dump(mode="json"), expanded=False)


def _latest_attempt_dir(run_dir: Path) -> Path | None:
    """Return the most-recent attempt-N subdir under run_dir/training, or None."""
    training_dir = run_dir / "training"
    if not training_dir.is_dir():
        return None
    attempts = [
        p for p in training_dir.iterdir()
        if p.is_dir() and p.name.startswith("attempt-")
        and p.name.split("-", 1)[1].isdigit()
    ]
    if not attempts:
        return None
    return max(attempts, key=lambda p: int(p.name.split("-", 1)[1]))


def _render_attempt_code_artifacts(run_dir: Path) -> None:
    """Render design.md + model.py + train.py expanders for the latest attempt.

    Used by both the in-progress trainer view and the final view so the LLM
    output stays inspectable across the entire run lifecycle.
    """
    attempt = _latest_attempt_dir(run_dir)
    if attempt is None:
        return

    design_path = attempt / "design.md"
    model_py_path = attempt / "model.py"
    train_py_path = attempt / "train.py"

    if not any(p.exists() for p in (design_path, model_py_path, train_py_path)):
        return

    st.divider()
    st.markdown(f"### LLM-generated artifacts &nbsp;·&nbsp; `{attempt.name}/`")

    if design_path.exists():
        with st.expander(
            f"📄 design.md ({design_path.stat().st_size:,} bytes)"
        ):
            try:
                st.markdown(design_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read design.md: {exc}")

    if model_py_path.exists():
        with st.expander(
            f"🐍 model.py ({model_py_path.stat().st_size:,} bytes)"
        ):
            try:
                st.code(
                    model_py_path.read_text(encoding="utf-8"),
                    language="python",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read model.py: {exc}")

    if train_py_path.exists():
        with st.expander(
            f"🐍 train.py ({train_py_path.stat().st_size:,} bytes)"
        ):
            try:
                st.code(
                    train_py_path.read_text(encoding="utf-8"),
                    language="python",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read train.py: {exc}")


# ---------------------------------------------------------------------------
# Trainer: in-progress artifact renderer (used while design.md gate is open)
# ---------------------------------------------------------------------------
def _render_trainer_in_progress(run_dir: Path) -> bool:
    """Render whatever Trainer artifacts are on disk. Returns True if anything
    was rendered (caller can suppress the empty-state placeholder)."""
    oracle_path = run_dir / "oracle.json"
    training_dir = run_dir / "training"

    # Find the latest attempt-N subdir (retry loop creates attempt-1, -2, ...).
    attempt_dirs = []
    if training_dir.is_dir():
        attempt_dirs = sorted(
            (p for p in training_dir.iterdir()
             if p.is_dir() and p.name.startswith("attempt-")),
            key=lambda p: int(p.name.split("-", 1)[1])
            if p.name.split("-", 1)[1].isdigit() else 0,
        )
    latest = attempt_dirs[-1] if attempt_dirs else training_dir
    design_path = latest / "design.md"
    verify_path = latest / "verify_report.json"
    model_py_path = latest / "model.py"
    train_py_path = latest / "train.py"
    status_path = latest / "status.json"

    any_found = oracle_path.exists() or training_dir.is_dir()
    if not any_found:
        return False

    status_info = {}
    if status_path.exists():
        try:
            status_info = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    status_label = status_info.get("status")
    header = "**Trainer is mid-flight.**"
    if status_label == "success":
        header = "**Trainer succeeded.**"
    elif status_label in ("smoke_failed", "training_failed"):
        header = f"**Trainer failed:** {status_label}"
    st.markdown(
        header + " Review artifacts below. If a design gate is open in the "
        "Approvals panel, approve it to continue."
    )

    # --- Oracle baseline ---
    if oracle_path.exists():
        try:
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            oracle = None
        if oracle:
            st.markdown("### Oracle baseline")
            o1, o2, o3 = st.columns(3)
            o1.metric(
                "Baseline accuracy",
                f"{float(oracle.get('test_accuracy', 0.0)):.3f}",
            )
            o2.metric("Wall time", f"{float(oracle.get('wall_clock_s', 0.0)):.1f}s")
            o3.metric(
                "Split",
                f"{int(oracle.get('n_train', 0))} / {int(oracle.get('n_test', 0))}",
            )
            st.caption(
                "Trained model must beat this by ≥0.05 to pass the oracle delta check."
            )

    # --- design.md (the star of the show) ---
    if design_path.exists():
        st.divider()
        st.markdown("### 📄 design.md (awaiting your approval)")
        try:
            design_text = design_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read design.md: {exc}")
        else:
            with st.container(height=520, border=True):
                st.markdown(design_text)
            st.caption(
                f"`{design_path}` &nbsp;·&nbsp; {len(design_text):,} chars"
            )

    # --- Smoke harness verdict ---
    if verify_path.exists():
        st.divider()
        st.markdown("### Smoke harness")
        try:
            verify = json.loads(verify_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            verify = None
        if verify:
            checks = verify.get("checks") or []
            passed = sum(1 for c in checks if c.get("passed"))
            total = len(checks)
            overall = verify.get("overall_passed", False)
            badge_color = "#10B981" if overall else "#EF4444"
            badge_text = "PASS" if overall else "FAIL"
            st.markdown(
                f"<div style='display:inline-block; padding:2px 10px; "
                f"border-radius:6px; background:{badge_color}; color:white; "
                f"font-weight:700;'>{badge_text}</div> &nbsp; "
                f"{passed}/{total} checks",
                unsafe_allow_html=True,
            )
            for c in checks:
                icon = "✓" if c.get("passed") else "✗"
                name = c.get("name", "?")
                detail = c.get("detail", "")
                if c.get("passed"):
                    st.markdown(f"- {icon} `{name}` {detail}")
                else:
                    st.markdown(f"- {icon} `{name}` — {detail[:200]}")

    # --- Generated code (collapsed by default) ---
    if model_py_path.exists() or train_py_path.exists():
        st.divider()
        st.markdown("### Generated code (model.py + train.py, by Nemotron)")
        if model_py_path.exists():
            with st.expander(
                f"`model.py` ({model_py_path.stat().st_size:,} bytes)"
            ):
                try:
                    st.code(
                        model_py_path.read_text(encoding="utf-8"),
                        language="python",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read model.py: {exc}")
        if train_py_path.exists():
            with st.expander(
                f"`train.py` ({train_py_path.stat().st_size:,} bytes)"
            ):
                try:
                    st.code(
                        train_py_path.read_text(encoding="utf-8"),
                        language="python",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read train.py: {exc}")

    # --- Final status block ---
    if status_info:
        st.divider()
        status_label = status_info.get("status", "unknown")
        reason = (status_info.get("reason") or "").strip()
        color = {
            "success": "#10B981",
            "smoke_failed": "#EF4444",
            "training_failed": "#EF4444",
        }.get(status_label, "#9CA3AF")
        st.markdown(
            f"<div style='display:inline-block; padding:2px 10px; "
            f"border-radius:6px; background:{color}; color:white; "
            f"font-weight:700;'>{status_label}</div>",
            unsafe_allow_html=True,
        )
        if reason:
            with st.expander("status.json reason", expanded=False):
                st.code(reason, language="text")

    return True


# ---------- Evaluator ----------
def _evaluator_view(report: BenchmarkReport | None) -> None:
    if report is None:
        st.info("Evaluator hasn't produced output yet.")
        return

    color = "#10B981" if report.passed_threshold else "#EF4444"
    label = "PASS" if report.passed_threshold else "FAIL"
    st.markdown(
        f"<div style='display:inline-block; padding:4px 14px; "
        f"border-radius:6px; background:{color}; color:white; "
        f"font-weight:800; letter-spacing:0.05em;'>{label}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(report.accuracy_metric.upper(), f"{report.accuracy_value:.3f}")
    m2.metric("Throughput", f"{report.throughput_qps:.0f} QPS")
    m3.metric("Memory", f"{report.memory_mb:.0f} MB")

    st.markdown("**Latency**")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("p50", f"{report.latency.p50_ms:.1f} ms")
    l2.metric("p95", f"{report.latency.p95_ms:.1f} ms")
    l3.metric("p99", f"{report.latency.p99_ms:.1f} ms")
    l4.metric("mean", f"{report.latency.mean_ms:.1f} ms")

    if report.pareto_frontier:
        st.markdown(f"**Pareto frontier** &nbsp;·&nbsp; {len(report.pareto_frontier)} points")
        rows = [
            {
                "config_id": p.config_id,
                "accuracy": f"{p.accuracy:.4f}",
                "latency_ms": f"{p.latency_ms:.2f}",
                "memory_mb": f"{p.memory_mb:.1f}",
            }
            for p in report.pareto_frontier
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

    if report.feedback_to_training:
        st.warning(f"**Feedback to Trainer:** {report.feedback_to_training}")

    if report.notes:
        st.caption(report.notes)

    with st.expander("Raw output (JSON)"):
        st.json(report.model_dump(mode="json"), expanded=False)
