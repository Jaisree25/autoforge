"""Populate the SQLite store with a synthetic run.

Lets you iterate on the dashboard UI without running the real pipeline.
Two modes:

    python scripts/fake_events.py             # paused at FIRST gate (Profiler → Researcher)
    python scripts/fake_events.py --completed # fully approved end-to-end

The completed mode emits all five approval gates as RESOLVED so the chat
feed and Gantt timeline have realistic data.

New (post-refactor) pipeline:
    Profiler → Researcher → Data Preparer → Trainer → Evaluator → Optimizer
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console

from config import AUTOFORGE_DB_PATH, configure_logging
from contracts.messages import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    EventType,
)
from contracts.schemas import (
    AgentName,
    BenchmarkReport,
    CandidateArchitecture,
    Citation,
    ColumnProfile,
    DatasetProfile,
    DeploymentArtifact,
    LatencyStats,
    ParetoPoint,
    PipelineStatus,
    PreparationReport,
    StrategySpec,
    TaskType,
    TrainingEnvelope,
    TrainingResult,
    TrialResult,
)
from memory.store import MemoryStore

configure_logging()
console = Console()


_DISPLAY = {
    AgentName.PROFILER:  "Profiler",
    AgentName.STRATEGY:  "Researcher",
    AgentName.DATASET:   "Data Preparer",
    AgentName.TRAINING:  "Trainer",
    AgentName.BENCHMARK: "Evaluator",
    AgentName.HARDWARE:  "Optimizer",
}


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def _dataset_profile(path: str) -> DatasetProfile:
    return DatasetProfile(
        dataset_path=path,
        n_rows=10_000, n_cols=12,
        columns=[
            ColumnProfile(name="customer_id", dtype="int64", unique_count=10_000),
            ColumnProfile(name="age", dtype="int64", missing_pct=0.012),
            ColumnProfile(name="tenure_months", dtype="int64"),
            ColumnProfile(name="monthly_charges", dtype="float64", missing_pct=0.004),
            ColumnProfile(name="contract_type", dtype="category", unique_count=3),
            ColumnProfile(name="churn", dtype="bool"),
        ],
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        class_balance={"0": 0.734, "1": 0.266},
        warnings=["`days_since_last_login` missing in 2.1% of rows — Preparer should impute."],
        profile_summary="10,000 rows × 12 columns. Target `churn` (73/27 imbalance).",
    )


def _training_envelope() -> TrainingEnvelope:
    return TrainingEnvelope(
        gpu_available=True, gpu_name="NVIDIA L40S", gpu_memory_gb=48.0,
        cpu_count=16, system_memory_gb=128.0,
        max_train_minutes=5.0, max_trials=20,
        batch_size_range=(32, 256),
        allowed_libraries=["xgboost", "sklearn"],
        notes="L40S has headroom; capping trials at 20 to keep iteration < 5 min.",
    )


def _strategy_spec(objective: str) -> StrategySpec:
    return StrategySpec(
        objective=objective,
        task_type=TaskType.BINARY_CLASSIFICATION,
        success_metric="f1",
        success_threshold=0.85,
        candidate_architectures=[
            CandidateArchitecture(
                name="xgboost-tuned", family="gradient_boost", library="xgboost",
                hyperparameter_space={
                    "max_depth": [3, 4, 5, 6, 8, 10],
                    "learning_rate": [0.01, 0.03, 0.1, 0.2],
                },
                rationale="Strong default for tabular imbalanced binary classification.",
            ),
            CandidateArchitecture(
                name="logreg-calibrated", family="linear", library="sklearn",
                hyperparameter_space={"C": [0.01, 0.1, 1.0, 10.0]},
                rationale="Cheap calibrated baseline for the latency Pareto frontier.",
            ),
        ],
        research_summary=(
            "Recent literature still favors gradient-boosted trees on tabular churn. "
            "Calibration matters when probabilities feed retention budgets."
        ),
        citations=[
            Citation(title="A Survey on Tabular Data Models",
                     url="https://arxiv.org/abs/2402.17944", source="arxiv"),
            Citation(title="Probabilistic Calibration for Imbalanced Binary Classification",
                     url="https://arxiv.org/abs/2310.07334", source="arxiv"),
        ],
    )


def _preparation_report(profile: DatasetProfile) -> PreparationReport:
    prepared = str(Path(profile.dataset_path).with_name(
        Path(profile.dataset_path).stem + "_prepared.csv"
    ))
    return PreparationReport(
        original_dataset_path=profile.dataset_path,
        prepared_dataset_path=prepared,
        operations=[
            "impute_missing(strategy='median')",
            "encode_categoricals(method='onehot', columns=['contract_type'])",
            "train_test_split(test_size=0.2, stratify_by='churn')",
        ],
        summary="Applied 3 operation(s). Wrote prepared dataset.",
        notes="Stub — operations planned, not executed.",
    )


def _training_result(spec: StrategySpec) -> TrainingResult:
    scores = [0.802, 0.811, 0.823, 0.828, 0.831, 0.834, 0.840, 0.843,
              0.847, 0.851, 0.855, 0.858, 0.860, 0.863, 0.866, 0.868,
              0.870, 0.871, 0.872, 0.874]
    trials = [
        TrialResult(trial_id=i,
                    params={"max_depth": 3 + i % 6, "learning_rate": 0.01 * (1 + i % 5)},
                    score=s, duration_seconds=1.4 + 0.1 * (i % 4), status="completed")
        for i, s in enumerate(scores)
    ]
    best = max(trials, key=lambda t: t.score)
    model_id = f"m_{uuid.uuid4().hex[:8]}"
    return TrainingResult(
        best_model_id=model_id, metric_name=spec.success_metric,
        best_score=best.score, best_params=best.params,
        trials_completed=len(trials), total_trials=len(trials),
        training_time_seconds=sum(t.duration_seconds for t in trials),
        artifact_path=f"data/artifacts/{model_id}.pkl",
        library=spec.candidate_architectures[0].library,
        all_trials=trials,
        notes=f"target {spec.success_threshold:.2f} — PASS",
    )


def _benchmark_report(tr: TrainingResult, spec: StrategySpec) -> BenchmarkReport:
    top = sorted(tr.all_trials, key=lambda t: t.score, reverse=True)[:3]
    return BenchmarkReport(
        model_id=tr.best_model_id,
        accuracy_metric=spec.success_metric, accuracy_value=tr.best_score,
        latency=LatencyStats(p50_ms=1.4, p95_ms=2.6, p99_ms=3.9, mean_ms=1.6),
        throughput_qps=620.0, memory_mb=14.0,
        passed_threshold=tr.best_score >= spec.success_threshold,
        pareto_frontier=[
            ParetoPoint(config_id=f"cfg_{t.trial_id}", accuracy=t.score,
                        latency_ms=1.4 + 0.6 * i, memory_mb=14.0 + 2.5 * i)
            for i, t in enumerate(top)
        ],
    )


def _deployment_artifact(tr: TrainingResult) -> DeploymentArtifact:
    return DeploymentArtifact(
        artifact_path=f"data/artifacts/{tr.best_model_id}.onnx",
        format="onnx", quantization="fp16", size_mb=3.7,
        notes="Exported to ONNX with fp16 weights.",
    )


# ---------------------------------------------------------------------------
# Summarizers — match coordinator._summarize()
# ---------------------------------------------------------------------------
def _sum_profile(p: DatasetProfile) -> str:
    return (
        f"Observed {p.n_rows:,} rows × {p.n_cols} columns. "
        f"Target `{p.target_column}` · task `{p.task_type.value}`."
    )
def _sum_env(e: TrainingEnvelope) -> str:
    return (
        f"Hardware envelope: {e.gpu_name} ({e.gpu_memory_gb:.0f}GB). "
        f"Max {e.max_trials} trials in ≤{e.max_train_minutes:.1f}min."
    )
def _sum_spec(s: StrategySpec) -> str:
    archs = ", ".join(f"`{a.name}`" for a in s.candidate_architectures)
    return (
        f"Recommended {len(s.candidate_architectures)} architecture(s): "
        f"{archs}. Target {s.success_metric.upper()} ≥ {s.success_threshold:.2f}."
    )
def _sum_prep(p: PreparationReport) -> str:
    return (
        f"Applied {len(p.operations)} operation(s). "
        f"Prepared dataset: `{p.prepared_dataset_path or '—'}`."
    )
def _sum_train(tr: TrainingResult) -> str:
    return (
        f"Best {tr.metric_name.upper()}: **{tr.best_score:.3f}** from "
        f"{tr.trials_completed}/{tr.total_trials} trials. "
        f"Library: `{tr.library}` · model `{tr.best_model_id}`."
    )
def _sum_bench(b: BenchmarkReport) -> str:
    verdict = "PASS" if b.passed_threshold else "FAIL"
    return (
        f"**{verdict}** · {b.accuracy_metric.upper()}={b.accuracy_value:.3f} · "
        f"p50={b.latency.p50_ms:.1f}ms · throughput={b.throughput_qps:.0f}QPS."
    )


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------
@dataclass
class _Timeline:
    run_id: str
    base_ts: datetime
    step: float = 0.5
    _i: int = 0
    events: list[AgentEvent] = field(default_factory=list)

    def _next_ts(self) -> datetime:
        ts = self.base_ts + timedelta(seconds=self._i * self.step)
        self._i += 1
        return ts

    def emit(
        self,
        agent: AgentName,
        et: EventType,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        ev = AgentEvent(
            run_id=self.run_id, agent=agent, event_type=et,
            message=message, payload=payload or {},
            created_at=self._next_ts(),
        )
        self.events.append(ev)
        return ev


def _run_one_agent(
    tl: _Timeline,
    agent: AgentName,
    summary_msg: str,
    tool_calls: list[str] = (),
    thinking: list[str] = (),
) -> None:
    tl.emit(agent, EventType.STARTED, summary_msg)
    for t in thinking:
        tl.emit(agent, EventType.THINKING, t)
    for tc in tool_calls:
        tl.emit(agent, EventType.TOOL_CALL, tc)
    tl.emit(agent, EventType.COMPLETED, summary_msg)


def _emit_gate(
    tl: _Timeline,
    store: MemoryStore,
    from_agent: AgentName,
    next_agent_display: str,
    summary: str,
    agent_output: Any,
    *,
    resolve: bool,
) -> None:
    request_id = str(uuid.uuid4())
    from_display = _DISPLAY[from_agent]

    tl.emit(
        AgentName.COORDINATOR,
        EventType.APPROVAL_REQUESTED,
        f"Handoff: {from_display} → {next_agent_display}",
        payload={
            "summary": summary,
            "next_agent": next_agent_display,
            "from_agent": from_agent.value,
            "request_id": request_id,
        },
    )

    req_created_at = tl.events[-1].created_at
    request = ApprovalRequest(
        request_id=request_id,
        run_id=tl.run_id,
        agent=from_agent,
        title=f"{from_display} → {next_agent_display}",
        description=summary,
        payload={
            "summary": summary,
            "next_agent": next_agent_display,
            "agent_output": agent_output.model_dump(mode="json"),
        },
        created_at=req_created_at,
    )
    store.create_approval_request(request)

    if resolve:
        store.respond_to_approval(ApprovalResponse(
            request_id=request_id,
            decision=ApprovalDecision.APPROVED,
            responder="dashboard",
            comment="auto-approved by fake_events",
        ))
        tl.emit(
            AgentName.COORDINATOR,
            EventType.APPROVAL_RECEIVED,
            "approved by dashboard",
            payload={"request_id": request_id, "decision": "approved"},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    completed: bool = typer.Option(
        False, "--completed", help="Generate a fully-approved run "
        "(default is a run paused at the FIRST gate, Profiler → Researcher)."
    ),
    objective: str = typer.Option(
        "predict customer churn with F1 >= 0.85", "--objective", "-o",
    ),
    dataset: str = typer.Option(
        "data/uploads/test.csv", "--dataset", "-d",
    ),
) -> None:
    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()

    run_id = str(uuid.uuid4())
    store.create_run(run_id, objective, dataset)

    base_ts = datetime.now(timezone.utc) - timedelta(minutes=3)
    tl = _Timeline(run_id=run_id, base_ts=base_ts)

    tl.emit(
        AgentName.COORDINATOR, EventType.STARTED,
        f"pipeline start: {objective}",
        payload={"dataset_path": dataset},
    )

    # === Stage 1: Profiler ===
    profile = _dataset_profile(dataset)
    envelope = _training_envelope()
    _run_one_agent(
        tl, AgentName.PROFILER, "observe dataset + hardware",
        thinking=[
            "detecting dataset modality (CSV / image / other)",
            f"parsing objective: '{objective}'",
        ],
        tool_calls=["pandas.read_csv (stub)", "pynvml.nvmlDeviceGetCount()"],
    )
    store.save_agent_output(run_id, AgentName.PROFILER, "dataset_profile", profile)
    store.save_agent_output(run_id, AgentName.PROFILER, "training_envelope", envelope)
    _emit_gate(
        tl, store, AgentName.PROFILER, _DISPLAY[AgentName.STRATEGY],
        _sum_profile(profile) + "\n\n_Envelope:_ " + _sum_env(envelope),
        profile,  # gate payload uses profile; envelope info is in the summary
        resolve=completed,
    )

    if not completed:
        store.update_run_status(run_id, PipelineStatus.AWAITING_APPROVAL)
        _persist_events(store, tl.events)
        _report(run_id, objective, tl.events, pending=True)
        return

    # === Stage 2: Researcher ===
    spec = _strategy_spec(objective)
    _run_one_agent(
        tl, AgentName.STRATEGY, "formalize objective + literature search",
        thinking=["parsing metric + threshold"],
        tool_calls=[
            "tavily.search('churn prediction tabular xgboost 2025')",
            "arxiv.search('imbalanced binary classification calibration')",
        ],
    )
    store.save_agent_output(run_id, AgentName.STRATEGY, "strategy_spec", spec)
    _emit_gate(
        tl, store, AgentName.STRATEGY, _DISPLAY[AgentName.DATASET],
        _sum_spec(spec), spec, resolve=True,
    )

    # === Stage 3: Data Preparer ===
    prep = _preparation_report(profile)
    _run_one_agent(
        tl, AgentName.DATASET, "clean + prepare dataset",
        thinking=[
            f"reviewing profile: {len(profile.warnings)} warning(s)",
            f"researcher recommends {spec.candidate_architectures[0].library}",
        ],
        tool_calls=prep.operations,
    )
    store.save_agent_output(run_id, AgentName.DATASET, "preparation_report", prep)
    _emit_gate(
        tl, store, AgentName.DATASET, _DISPLAY[AgentName.TRAINING],
        _sum_prep(prep), prep, resolve=True,
    )

    # === Stage 4: Trainer ===
    tl.emit(AgentName.TRAINING, EventType.STARTED, "HPO xgboost-tuned (20 trials)")
    for i in range(20):
        score = 0.802 + i * 0.0036
        tl.emit(
            AgentName.TRAINING, EventType.THINKING,
            f"trial {i + 1}/20: xgboost-tuned -> {score:.3f}",
            {"trial_id": i, "score": score},
        )
    tl.emit(AgentName.TRAINING, EventType.INFO, "best: 0.874 (xgboost-tuned, trial #19)")
    tl.emit(AgentName.TRAINING, EventType.COMPLETED, "HPO xgboost-tuned (20 trials)")
    tr = _training_result(spec)
    store.save_agent_output(run_id, AgentName.TRAINING, "training_result", tr)
    _emit_gate(
        tl, store, AgentName.TRAINING, _DISPLAY[AgentName.BENCHMARK],
        _sum_train(tr), tr, resolve=True,
    )

    # === Stage 5: Evaluator ===
    _run_one_agent(
        tl, AgentName.BENCHMARK, f"benchmark {tr.best_model_id}",
        tool_calls=[
            "sklearn.metrics.f1_score on held-out split",
            "latency harness (5000 inferences, batch=1)",
        ],
    )
    br = _benchmark_report(tr, spec)
    store.save_agent_output(run_id, AgentName.BENCHMARK, "benchmark_report", br)
    _emit_gate(
        tl, store, AgentName.BENCHMARK, _DISPLAY[AgentName.HARDWARE],
        _sum_bench(br), br, resolve=True,
    )

    # === Stage 6: Optimizer (no gate after final) ===
    _run_one_agent(
        tl, AgentName.HARDWARE, f"optimize {tr.best_model_id} for deployment",
        tool_calls=["tensorrt.export", "quantize fp16 → int8 calibration"],
    )
    da = _deployment_artifact(tr)
    store.save_agent_output(run_id, AgentName.HARDWARE, "deployment_artifact", da)

    tl.emit(
        AgentName.COORDINATOR, EventType.COMPLETED,
        "pipeline complete: f1=0.874, passed=True",
    )

    _persist_events(store, tl.events)
    store.update_run_status(run_id, PipelineStatus.COMPLETED)
    _report(run_id, objective, tl.events, pending=False)


def _persist_events(store: MemoryStore, events: list[AgentEvent]) -> None:
    for ev in events:
        store.log_event(ev)


def _report(
    run_id: str, objective: str, events: list[AgentEvent], pending: bool,
) -> None:
    state = "pending @ first gate" if pending else "fully completed"
    console.print(f"[green]Created {state} run[/] {run_id[:8]}")
    console.print(f"  objective: [cyan]{objective}[/]")
    console.print(f"  events:    {len(events)}")
    console.print(f"  open the dashboard:  [bold]streamlit run dashboard/app.py[/]")


if __name__ == "__main__":
    app()
