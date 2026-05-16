"""Codifies the Phase 2 contracts smoke check.

Each schema must:
  - instantiate with realistic data
  - reject typo'd fields (extra=forbid)
  - JSON round-trip cleanly (the storage path is .model_dump_json())
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.messages import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
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
    PipelineRun,
    PipelineStatus,
    StrategySpec,
    TaskType,
    TrainingEnvelope,
    TrainingResult,
    TrialResult,
)


def _dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_path="data/uploads/test.csv",
        n_rows=1000,
        n_cols=12,
        columns=[ColumnProfile(name="age", dtype="int64", missing_pct=0.02)],
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        class_balance={"0": 0.7, "1": 0.3},
        profile_summary="1000x12 tabular, target=churn",
    )


def _strategy_spec() -> StrategySpec:
    return StrategySpec(
        objective="predict churn with F1 >= 0.85",
        task_type=TaskType.BINARY_CLASSIFICATION,
        success_metric="f1",
        success_threshold=0.85,
        candidate_architectures=[
            CandidateArchitecture(
                name="xgb-default",
                family="gradient_boost",
                library="xgboost",
                hyperparameter_space={"max_depth": [3, 9]},
                rationale="strong tabular baseline",
            )
        ],
        citations=[Citation(title="A relevant paper", url="https://arxiv.org/abs/1234.5678")],
    )


def _training_result() -> TrainingResult:
    return TrainingResult(
        best_model_id="m1",
        metric_name="f1",
        best_score=0.87,
        best_params={"max_depth": 6},
        trials_completed=20,
        total_trials=20,
        artifact_path="data/artifacts/m1.pkl",
        library="xgboost",
        all_trials=[TrialResult(trial_id=0, params={"max_depth": 6}, score=0.87)],
    )


def _benchmark_report() -> BenchmarkReport:
    return BenchmarkReport(
        model_id="m1",
        accuracy_metric="f1",
        accuracy_value=0.87,
        latency=LatencyStats(p50_ms=2.1, p95_ms=4.0, p99_ms=5.5, mean_ms=2.5),
        throughput_qps=410.0,
        memory_mb=120.0,
        passed_threshold=True,
        pareto_frontier=[ParetoPoint(config_id="m1", accuracy=0.87, latency_ms=2.1)],
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------
def test_dataset_profile_instantiates():
    dp = _dataset_profile()
    assert dp.task_type is TaskType.BINARY_CLASSIFICATION
    assert dp.columns[0].missing_pct == pytest.approx(0.02)


def test_strategy_spec_instantiates():
    ss = _strategy_spec()
    assert ss.success_threshold == 0.85
    assert ss.candidate_architectures[0].library == "xgboost"


def test_training_envelope_instantiates():
    te = TrainingEnvelope(gpu_available=False, max_train_minutes=5.0, max_trials=20)
    assert te.batch_size_range == (16, 128)


def test_training_result_instantiates():
    tr = _training_result()
    assert tr.best_score == 0.87
    assert tr.trials_completed == 20


def test_benchmark_report_instantiates():
    br = _benchmark_report()
    assert br.passed_threshold is True
    assert br.latency.p99_ms == 5.5


def test_deployment_artifact_instantiates():
    da = DeploymentArtifact(
        artifact_path="data/artifacts/m1.onnx",
        format="onnx",
        quantization="fp16",
        size_mb=12.3,
    )
    assert da.format == "onnx"


# ---------------------------------------------------------------------------
# Composition + JSON round-trip
# ---------------------------------------------------------------------------
def test_pipeline_run_composes_and_round_trips():
    run = PipelineRun(
        run_id="r1",
        objective="predict churn",
        dataset_path="data/uploads/test.csv",
        status=PipelineStatus.RUNNING,
        dataset_profile=_dataset_profile(),
        strategy_spec=_strategy_spec(),
        training_envelope=TrainingEnvelope(),
        training_result=_training_result(),
        benchmark_report=_benchmark_report(),
    )
    blob = run.model_dump_json()
    rehydrated = PipelineRun.model_validate_json(blob)

    assert rehydrated.run_id == "r1"
    assert rehydrated.status is PipelineStatus.RUNNING
    assert rehydrated.benchmark_report.passed_threshold is True
    assert rehydrated.dataset_profile.task_type is TaskType.BINARY_CLASSIFICATION


# ---------------------------------------------------------------------------
# Strictness — typos blow up at the boundary
# ---------------------------------------------------------------------------
def test_extra_forbid_rejects_typos_on_schemas():
    with pytest.raises(ValidationError):
        DatasetProfile(dataset_path="x", n_rows=1, n_cols=1, typoed_field="oops")


def test_extra_forbid_rejects_typos_on_messages():
    with pytest.raises(ValidationError):
        AgentEvent(
            run_id="r1",
            agent=AgentName.DATASET,
            event_type=EventType.STARTED,
            unexpected="boom",
        )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def test_agent_event_instantiates_with_payload():
    ev = AgentEvent(
        run_id="r1",
        agent=AgentName.DATASET,
        event_type=EventType.THINKING,
        message="profiling 1000 rows...",
        payload={"step": "infer_dtypes"},
    )
    assert ev.payload["step"] == "infer_dtypes"
    assert ev.event_id  # auto-generated


def test_approval_request_response_cycle():
    req = ApprovalRequest(
        run_id="r1",
        agent=AgentName.STRATEGY,
        title="Approve strategy spec",
        payload=_strategy_spec().model_dump(mode="json"),
    )
    assert req.status is ApprovalStatus.PENDING
    assert req.decision is None

    resp = ApprovalResponse(
        request_id=req.request_id,
        decision=ApprovalDecision.APPROVED,
        responder="dashboard",
        comment="lgtm",
    )
    assert resp.request_id == req.request_id
    assert resp.decision is ApprovalDecision.APPROVED
