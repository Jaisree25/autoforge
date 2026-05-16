"""End-to-end skeleton smoke test.

Runs the full pipeline with stub agents and the auto-approve HITL service,
asserts the pipeline reaches COMPLETED with every agent's output persisted.

This is the test that gates Phase 4 → Phase 5.
"""
from __future__ import annotations

import pytest

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EventType,
)
from contracts.schemas import (
    AgentName,
    BenchmarkReport,
    DatasetProfile,
    PipelineStatus,
    StrategySpec,
    TrainingEnvelope,
    TrainingResult,
)
from memory.store import MemoryStore

from agents.coordinator import Coordinator, PipelineRejected
from hitl.auto import AutoApproveHITLService


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "autoforge.db"
    s = MemoryStore(db_path=db)
    s.init_schema()
    return s


def test_end_to_end_skeleton_completes(store):
    hitl = AutoApproveHITLService(store)
    coord = Coordinator(store=store, hitl=hitl)

    run = coord.execute(
        dataset_path="data/uploads/test.csv",
        objective="predict customer churn with F1 >= 0.85",
    )

    # Every persisted agent output materializes on the PipelineRun.
    assert run.status is PipelineStatus.COMPLETED
    assert run.error is None
    assert isinstance(run.dataset_profile, DatasetProfile)
    assert isinstance(run.strategy_spec, StrategySpec)
    assert isinstance(run.training_envelope, TrainingEnvelope)
    assert isinstance(run.training_result, TrainingResult)
    assert isinstance(run.benchmark_report, BenchmarkReport)
    assert run.deployment_artifact is not None
    # Format is now "joblib" (real sklearn pickle); ONNX export is future work.
    assert run.deployment_artifact.format in {"joblib", "onnx"}


def test_live_trace_shows_all_six_agents(store):
    """Dashboard sees STARTED + COMPLETED from each agent + the approval gate."""
    hitl = AutoApproveHITLService(store)
    coord = Coordinator(store=store, hitl=hitl)
    coord.execute(dataset_path="data/uploads/test.csv", objective="x")

    events = store.get_events(coord.run_id)
    agents_seen = {e.agent for e in events if e.event_type is EventType.STARTED}
    assert AgentName.COORDINATOR in agents_seen
    assert AgentName.DATASET in agents_seen
    assert AgentName.STRATEGY in agents_seen
    assert AgentName.HARDWARE in agents_seen
    assert AgentName.TRAINING in agents_seen
    assert AgentName.BENCHMARK in agents_seen

    # Both APPROVAL events surface on the live trace
    types = {e.event_type for e in events}
    assert EventType.APPROVAL_REQUESTED in types
    assert EventType.APPROVAL_RECEIVED in types


def test_auto_approve_marks_request_resolved(store):
    hitl = AutoApproveHITLService(store)
    coord = Coordinator(store=store, hitl=hitl)
    coord.execute(dataset_path="data/uploads/test.csv", objective="x")

    # No pending approvals after a clean run
    assert store.list_pending_approvals(coord.run_id) == []


def test_rejection_cancels_pipeline(store):
    """A rejecting HITL service should leave the pipeline CANCELLED, not FAILED."""
    class RejectingHITL:
        def __init__(self, inner):
            self.inner = inner

        def request_and_wait(self, request: ApprovalRequest, timeout: float = 600.0):
            self.inner.store.create_approval_request(request)
            response = ApprovalResponse(
                request_id=request.request_id,
                decision=ApprovalDecision.REJECTED,
                responder="test",
                comment="not good enough",
            )
            self.inner.store.respond_to_approval(response)
            return response

        def notify(self, run_id, message, payload=None):
            pass

    coord = Coordinator(store=store, hitl=RejectingHITL(AutoApproveHITLService(store)))
    with pytest.raises(PipelineRejected):
        coord.execute(dataset_path="data/uploads/test.csv", objective="x")

    run = store.get_run(coord.run_id)
    assert run.status is PipelineStatus.CANCELLED
    assert "not good enough" in (run.error or "")
    # Training was never started: no training_result row should exist.
    assert run.training_result is None
    assert run.benchmark_report is None


def test_candidate_pick_trims_strategy(store):
    """The Researcher gate is now a candidate picker. Selecting index 1 must
    trim the persisted StrategySpec to just that candidate (the simple-baseline
    stub), not the first one (xgboost stub)."""

    class PickerHITL:
        """Approves every gate. For the candidate_pick gate, returns selected_index=1."""

        def __init__(self, inner):
            self.inner = inner

        def request_and_wait(self, request: ApprovalRequest, timeout: float = 600.0):
            self.inner.store.create_approval_request(request)
            if request.payload.get("kind") == "candidate_pick":
                response = ApprovalResponse(
                    request_id=request.request_id,
                    decision=ApprovalDecision.APPROVED,
                    responder="test",
                    response_payload={"selected_index": 1},
                    comment="picked candidate #2",
                )
            else:
                response = ApprovalResponse(
                    request_id=request.request_id,
                    decision=ApprovalDecision.APPROVED,
                    responder="test",
                    comment="auto-approved",
                )
            self.inner.store.respond_to_approval(response)
            return response

        def notify(self, run_id, message, payload=None):
            pass

    coord = Coordinator(store=store, hitl=PickerHITL(AutoApproveHITLService(store)))
    run = coord.execute(dataset_path="data/uploads/test.csv", objective="x")

    assert run.status is PipelineStatus.COMPLETED
    # After trimming, only the picked candidate should remain.
    assert len(run.strategy_spec.candidate_architectures) == 1
    assert run.strategy_spec.candidate_architectures[0].name == "logreg-baseline-stub"
