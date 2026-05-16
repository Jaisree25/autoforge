"""End-to-end MemoryStore round-trips against a temp SQLite file."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
    DatasetProfile,
    LatencyStats,
    PipelineStatus,
    StrategySpec,
    TaskType,
    TrainingEnvelope,
    TrainingResult,
)
from memory.store import MemoryStore, OUTPUT_KIND_TO_MODEL


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "autoforge.db"
    s = MemoryStore(db_path=db)
    s.init_schema()
    return s


def test_init_schema_dumps_json_schemas(tmp_path):
    db = tmp_path / "autoforge.db"
    s = MemoryStore(db_path=db)
    s.init_schema()
    # config.ARTIFACTS_DIR is the project-level dir, not tmp_path, so we just
    # verify each expected file exists. (init_schema writes to config.ARTIFACTS_DIR.)
    from config import ARTIFACTS_DIR
    for kind in OUTPUT_KIND_TO_MODEL:
        assert (ARTIFACTS_DIR / "contracts" / f"{kind}.schema.json").exists()


def test_run_lifecycle(store):
    store.create_run("r1", "predict churn", "data/uploads/test.csv")
    run = store.get_run("r1")
    assert run is not None
    assert run.status is PipelineStatus.PENDING
    assert run.dataset_profile is None

    store.update_run_status("r1", PipelineStatus.RUNNING)
    run = store.get_run("r1")
    assert run.status is PipelineStatus.RUNNING

    store.update_run_status("r1", PipelineStatus.FAILED, error="boom")
    run = store.get_run("r1")
    assert run.status is PipelineStatus.FAILED
    assert run.error == "boom"


def test_list_runs_orders_newest_first(store):
    store.create_run("r1", "first", "a.csv")
    store.create_run("r2", "second", "b.csv")
    rows = store.list_runs()
    assert [r["run_id"] for r in rows[:2]] == ["r2", "r1"]


def test_save_and_hydrate_agent_outputs(store):
    store.create_run("r1", "predict churn", "x.csv")

    dp = DatasetProfile(
        dataset_path="x.csv", n_rows=100, n_cols=5,
        target_column="y", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    ss = StrategySpec(
        objective="predict churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        success_metric="f1",
        success_threshold=0.8,
        candidate_architectures=[CandidateArchitecture(
            name="xgb", family="gradient_boost", library="xgboost"
        )],
    )
    te = TrainingEnvelope(gpu_available=True, gpu_name="A100", max_trials=10)
    tr = TrainingResult(
        best_model_id="m1", metric_name="f1", best_score=0.86,
        trials_completed=10, total_trials=10,
        artifact_path="data/artifacts/m1.pkl", library="xgboost",
    )
    br = BenchmarkReport(
        model_id="m1", accuracy_metric="f1", accuracy_value=0.86,
        latency=LatencyStats(p50_ms=1.0), passed_threshold=True,
    )

    store.save_agent_output("r1", AgentName.DATASET, "dataset_profile", dp)
    store.save_agent_output("r1", AgentName.STRATEGY, "strategy_spec", ss)
    store.save_agent_output("r1", AgentName.HARDWARE, "training_envelope", te)
    store.save_agent_output("r1", AgentName.TRAINING, "training_result", tr)
    store.save_agent_output("r1", AgentName.BENCHMARK, "benchmark_report", br)

    run = store.get_run("r1")
    assert run.dataset_profile.n_rows == 100
    assert run.strategy_spec.success_threshold == 0.8
    assert run.training_envelope.gpu_name == "A100"
    assert run.training_result.best_score == 0.86
    assert run.benchmark_report.passed_threshold is True


def test_latest_output_wins_when_multiple_rows(store):
    """The Training Agent may re-run after Benchmark feedback. Latest wins."""
    store.create_run("r1", "x", "x.csv")
    tr1 = TrainingResult(
        best_model_id="m1", metric_name="f1", best_score=0.70,
        trials_completed=10, total_trials=10,
        artifact_path="a.pkl", library="xgboost",
    )
    tr2 = TrainingResult(
        best_model_id="m2", metric_name="f1", best_score=0.85,
        trials_completed=10, total_trials=10,
        artifact_path="b.pkl", library="xgboost",
    )
    store.save_agent_output("r1", AgentName.TRAINING, "training_result", tr1)
    store.save_agent_output("r1", AgentName.TRAINING, "training_result", tr2)

    latest = store.get_agent_output("r1", "training_result")
    assert latest.best_model_id == "m2"


def test_unknown_output_kind_raises(store):
    store.create_run("r1", "x", "x.csv")
    with pytest.raises(ValueError):
        store.save_agent_output("r1", AgentName.DATASET, "not_a_real_kind", {})


def test_approval_request_and_response_roundtrip(store):
    store.create_run("r1", "x", "x.csv")
    req = ApprovalRequest(
        run_id="r1",
        agent=AgentName.STRATEGY,
        title="Approve strategy",
        description="Please review the chosen approach.",
        payload={"plan": "xgboost"},
    )
    store.create_approval_request(req)

    fetched = store.get_approval_request(req.request_id)
    assert fetched is not None
    assert fetched.status is ApprovalStatus.PENDING
    assert fetched.payload == {"plan": "xgboost"}

    pending = store.list_pending_approvals()
    assert any(a.request_id == req.request_id for a in pending)

    resolved = store.respond_to_approval(ApprovalResponse(
        request_id=req.request_id,
        decision=ApprovalDecision.APPROVED,
        responder="dashboard",
        comment="lgtm",
    ))
    assert resolved.status is ApprovalStatus.RESOLVED
    assert resolved.decision is ApprovalDecision.APPROVED
    assert resolved.responder == "dashboard"

    # Second response should fail (no longer pending).
    with pytest.raises(LookupError):
        store.respond_to_approval(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.REJECTED,
        ))

    assert store.list_pending_approvals() == []


def test_list_pending_approvals_filters_by_run(store):
    store.create_run("r1", "x", "x.csv")
    store.create_run("r2", "y", "y.csv")
    a = ApprovalRequest(run_id="r1", agent=AgentName.STRATEGY, title="A")
    b = ApprovalRequest(run_id="r2", agent=AgentName.STRATEGY, title="B")
    store.create_approval_request(a)
    store.create_approval_request(b)

    assert {x.title for x in store.list_pending_approvals("r1")} == {"A"}
    assert {x.title for x in store.list_pending_approvals("r2")} == {"B"}
    assert {x.title for x in store.list_pending_approvals()} == {"A", "B"}


def test_event_log_and_replay(store):
    store.create_run("r1", "x", "x.csv")
    e1 = AgentEvent(
        run_id="r1", agent=AgentName.DATASET, event_type=EventType.STARTED,
        message="profiling",
    )
    e2 = AgentEvent(
        run_id="r1", agent=AgentName.DATASET, event_type=EventType.COMPLETED,
        message="done", payload={"rows": 100},
    )
    store.log_event(e1)
    store.log_event(e2)

    all_events = store.get_events("r1")
    assert len(all_events) == 2
    assert all_events[0].event_type is EventType.STARTED
    assert all_events[1].payload == {"rows": 100}

    # `since` is exclusive
    middle = all_events[0].created_at
    later = store.get_events("r1", since=middle)
    assert [e.event_type for e in later] == [EventType.COMPLETED]


def test_events_dont_leak_across_runs(store):
    store.create_run("r1", "x", "x.csv")
    store.create_run("r2", "y", "y.csv")
    store.log_event(AgentEvent(
        run_id="r1", agent=AgentName.DATASET, event_type=EventType.STARTED,
    ))
    store.log_event(AgentEvent(
        run_id="r2", agent=AgentName.STRATEGY, event_type=EventType.STARTED,
    ))
    r1_events = store.get_events("r1")
    r2_events = store.get_events("r2")
    assert len(r1_events) == 1 and r1_events[0].agent is AgentName.DATASET
    assert len(r2_events) == 1 and r2_events[0].agent is AgentName.STRATEGY


def test_foreign_key_cascade(store):
    """Deleting a run should cascade events and outputs (defensive — schema config)."""
    store.create_run("r1", "x", "x.csv")
    store.log_event(AgentEvent(
        run_id="r1", agent=AgentName.DATASET, event_type=EventType.STARTED,
    ))
    # Directly delete via the connection (admin path; not part of the public API)
    with store._write_tx() as conn:
        conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", ("r1",))
    assert store.get_events("r1") == []
    assert store.get_run("r1") is None
