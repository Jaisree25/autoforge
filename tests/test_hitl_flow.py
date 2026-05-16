"""Tests for the HITL approval pipeline.

Covers both wakeup paths:
  - in-process: queue.resolve() signals threading.Event → waiter unblocks
                immediately.
  - cross-process: a separate `MemoryStore` (simulating the dashboard) writes
                the response directly → the waiter's DB poll picks it up.
"""
from __future__ import annotations

import threading
import time

import pytest

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from contracts.schemas import AgentName, PipelineStatus
from memory.store import MemoryStore

from agents.coordinator import Coordinator
from hitl.approval_queue import ApprovalQueue
from hitl.coordinator_service import HITLCoordinatorService


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "autoforge.db"
    s = MemoryStore(db_path=db)
    s.init_schema()
    s.create_run("r1", "obj", "x.csv")
    return s


@pytest.fixture()
def queue(store):
    return ApprovalQueue(store)


# ---------------------------------------------------------------------------
# In-process path: queue.resolve() signals the waiter.
# ---------------------------------------------------------------------------
def test_in_process_resolution_signals_waiter(queue, store):
    """resolve() should wake wait_for_approval() in < poll_interval."""
    req = ApprovalRequest(
        run_id="r1", agent=AgentName.STRATEGY, title="approve me",
    )
    queue.request_approval(req)

    def resolve_after_delay():
        time.sleep(0.1)
        queue.resolve(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.APPROVED,
            responder="test",
        ))

    threading.Thread(target=resolve_after_delay, daemon=True).start()

    t0 = time.monotonic()
    # Use a long poll_interval to prove the Event (not the poll) woke us.
    response = queue.wait_for_approval(
        req.request_id, timeout=10.0, poll_interval=5.0
    )
    elapsed = time.monotonic() - t0

    assert response is not None
    assert response.decision is ApprovalDecision.APPROVED
    assert elapsed < 1.0, f"event-based wakeup should be near-instant, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Cross-process path: dashboard simulator writes the response directly to the
# DB via its own store; the pipeline's queue picks it up via DB polling.
# ---------------------------------------------------------------------------
def test_cross_process_resolution_picked_up_by_poll(queue, tmp_path):
    """A second MemoryStore against the same DB simulates the dashboard process."""
    req = ApprovalRequest(
        run_id="r1", agent=AgentName.STRATEGY, title="approve me",
    )
    queue.request_approval(req)

    # Separate MemoryStore = separate process, in effect.
    dashboard_store = MemoryStore(db_path=queue.store.db_path)

    def dashboard_resolves_after_delay():
        time.sleep(0.2)
        dashboard_store.respond_to_approval(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.APPROVED,
            responder="dashboard",
        ))

    threading.Thread(target=dashboard_resolves_after_delay, daemon=True).start()

    t0 = time.monotonic()
    response = queue.wait_for_approval(
        req.request_id, timeout=5.0, poll_interval=0.1
    )
    elapsed = time.monotonic() - t0

    assert response is not None
    assert response.decision is ApprovalDecision.APPROVED
    assert response.responder == "dashboard"
    # Should fire within ~poll_interval after the resolver finishes
    assert elapsed < 1.0, f"poll-based wakeup took unusually long: {elapsed:.2f}s"


def test_timeout_returns_none(queue):
    req = ApprovalRequest(
        run_id="r1", agent=AgentName.STRATEGY, title="approve me",
    )
    queue.request_approval(req)

    response = queue.wait_for_approval(
        req.request_id, timeout=0.3, poll_interval=0.1
    )
    assert response is None

    # The request stays PENDING — caller decides whether to mark it timed-out.
    persisted = queue.store.get_approval_request(req.request_id)
    assert persisted.status is ApprovalStatus.PENDING


def test_unknown_request_raises_lookup_error(queue):
    with pytest.raises(LookupError):
        queue.wait_for_approval("does-not-exist", timeout=0.1)


# ---------------------------------------------------------------------------
# HITLCoordinatorService end-to-end (no Telegram).
# ---------------------------------------------------------------------------
def test_coordinator_service_blocks_then_returns(queue, store):
    service = HITLCoordinatorService(store=store, queue=queue, telegram=None)
    req = ApprovalRequest(
        run_id="r1", agent=AgentName.STRATEGY, title="approve me",
    )

    def resolve_after_delay():
        time.sleep(0.1)
        queue.resolve(ApprovalResponse(
            request_id=req.request_id,
            decision=ApprovalDecision.APPROVED,
            responder="dashboard",
        ))

    threading.Thread(target=resolve_after_delay, daemon=True).start()
    response = service.request_and_wait(req, timeout=5.0)

    assert response.decision is ApprovalDecision.APPROVED
    assert response.responder == "dashboard"


def test_coordinator_service_raises_on_timeout(queue, store):
    service = HITLCoordinatorService(store=store, queue=queue, telegram=None)
    req = ApprovalRequest(
        run_id="r1", agent=AgentName.STRATEGY, title="will time out",
    )
    with pytest.raises(TimeoutError):
        service.request_and_wait(req, timeout=0.2)


# ---------------------------------------------------------------------------
# Full pipeline using HITLCoordinatorService with a "dashboard" simulator.
# This exercises the same cross-process path the real Streamlit app will take.
# ---------------------------------------------------------------------------
def test_full_pipeline_with_dashboard_simulator(store):
    """Pipeline runs, dashboard (separate MemoryStore) approves, pipeline completes."""
    queue = ApprovalQueue(store)
    service = HITLCoordinatorService(store=store, queue=queue, telegram=None)
    coord = Coordinator(store=store, hitl=service)

    # Simulate the dashboard: poll for pending approvals on its own store and
    # approve them with `responder="dashboard"`.
    dashboard_store = MemoryStore(db_path=store.db_path)
    stop_flag = threading.Event()

    def dashboard_loop():
        while not stop_flag.is_set():
            for req in dashboard_store.list_pending_approvals():
                dashboard_store.respond_to_approval(ApprovalResponse(
                    request_id=req.request_id,
                    decision=ApprovalDecision.APPROVED,
                    responder="dashboard",
                    comment="approved by dashboard simulator",
                ))
            time.sleep(0.05)

    t = threading.Thread(target=dashboard_loop, daemon=True)
    t.start()
    try:
        run = coord.execute(
            dataset_path="data/uploads/test.csv",
            objective="predict churn",
        )
    finally:
        stop_flag.set()
        t.join(timeout=2.0)

    assert run.status is PipelineStatus.COMPLETED
    assert run.benchmark_report is not None

    # The approval the coordinator created should be RESOLVED, responded by dashboard
    pending = store.list_pending_approvals(coord.run_id)
    assert pending == []
