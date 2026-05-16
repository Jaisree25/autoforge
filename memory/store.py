"""SQLite-backed memory store for AutoForge.

`MemoryStore` is the single source of truth for pipeline state. All
persistence (runs, agent outputs, approvals, live events) goes through this
class. Agent code MUST NOT touch SQLite or files under `data/` directly.

Threading model
---------------
The dashboard process and the pipeline process both talk to the same SQLite
file. We enable WAL mode so concurrent readers don't block the writer, and
open a fresh connection per call (cheap on SQLite) to sidestep any
thread-affinity concerns.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from pydantic import BaseModel

from config import AUTOFORGE_DB_PATH, ARTIFACTS_DIR, MEMORY_SCHEMA_PATH
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
    DatasetProfile,
    DeploymentArtifact,
    PipelineRun,
    PipelineStatus,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
    TrainingResult,
)

# ---------------------------------------------------------------------------
# Registry: which output_kind hydrates into which schema on read.
# ---------------------------------------------------------------------------
OUTPUT_KIND_TO_MODEL: dict[str, type[BaseModel]] = {
    "dataset_profile": DatasetProfile,
    "strategy_spec": StrategySpec,
    "training_envelope": TrainingEnvelope,
    "preparation_report": PreparationReport,
    "training_result": TrainingResult,
    "deployment_artifact": DeploymentArtifact,
    "benchmark_report": BenchmarkReport,
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json(payload: BaseModel | dict[str, Any] | None) -> str:
    """Serialize a Pydantic model or dict to JSON for storage."""
    if payload is None:
        return "{}"
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, default=str)


def _from_json(raw: str | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    return json.loads(raw)


def _dump_json_schemas(out_dir: Path) -> None:
    """Dump JSON schemas for the contract models to `out_dir/`.

    Lets the dashboard render generic forms over `ApprovalRequest.payload`
    without hardcoding per-schema UIs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for kind, model in OUTPUT_KIND_TO_MODEL.items():
        path = out_dir / f"{kind}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2), encoding="utf-8"
        )
        written.append(path.name)
    logger.debug("Wrote JSON schemas: {}", ", ".join(written))


class MemoryStore:
    """Thin SQLite wrapper. All writes commit on context exit."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path: Path = Path(db_path) if db_path is not None else AUTOFORGE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        # Events fired in rapid succession can share a microsecond timestamp.
        # We enforce strictly-increasing event timestamps so the dashboard's
        # `since=last_seen` poll cursor is well-defined.
        self._ts_lock = threading.Lock()
        self._last_event_ts: datetime | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # autocommit; we manage txns explicitly
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        """Serialized write transaction. SQLite allows one writer at a time."""
        with self._write_lock, self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def init_schema(self) -> None:
        """Create tables if absent and dump JSON schemas for the dashboard."""
        sql = MEMORY_SCHEMA_PATH.read_text(encoding="utf-8")
        # `executescript` implicitly commits any pending transaction, so we
        # don't wrap it in our `_write_tx` context manager (DDL is idempotent
        # via `IF NOT EXISTS` — no atomicity needed across statements).
        with self._write_lock, self._conn() as conn:
            conn.executescript(sql)
        _dump_json_schemas(ARTIFACTS_DIR / "contracts")
        logger.info("MemoryStore initialized at {}", self.db_path)

    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------
    def create_run(self, run_id: str, objective: str, dataset_path: str) -> None:
        now = _utcnow_iso()
        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs
                    (run_id, objective, dataset_path, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    objective,
                    dataset_path,
                    PipelineStatus.PENDING.value,
                    now,
                    now,
                ),
            )
        logger.info("Created run {}: {}", run_id, objective)

    def update_run_status(
        self,
        run_id: str,
        status: PipelineStatus,
        error: str | None = None,
    ) -> None:
        with self._write_tx() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs
                   SET status = ?, error = ?, updated_at = ?
                 WHERE run_id = ?
                """,
                (status.value, error, _utcnow_iso(), run_id),
            )

    def get_run(self, run_id: str) -> PipelineRun | None:
        """Hydrate a `PipelineRun` from `pipeline_runs` + latest `agent_outputs`."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None

            run_kwargs: dict[str, Any] = {
                "run_id": row["run_id"],
                "objective": row["objective"],
                "dataset_path": row["dataset_path"],
                "status": PipelineStatus(row["status"]),
                "error": row["error"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
            }

            # Latest output per kind
            for kind, model in OUTPUT_KIND_TO_MODEL.items():
                out_row = conn.execute(
                    """
                    SELECT payload FROM agent_outputs
                     WHERE run_id = ? AND output_kind = ?
                     ORDER BY created_at DESC, output_id DESC
                     LIMIT 1
                    """,
                    (run_id, kind),
                ).fetchone()
                if out_row is not None:
                    run_kwargs[kind] = model.model_validate_json(out_row["payload"])

        return PipelineRun(**run_kwargs)

    def delete_run(self, run_id: str) -> bool:
        """Hard-delete a run and everything that depends on it.

        FK cascade on `run_id` removes the run's events / agent_outputs /
        approval_requests in one DELETE. Returns True if the row existed.
        """
        with self._write_tx() as conn:
            cur = conn.execute(
                "DELETE FROM pipeline_runs WHERE run_id = ?", (run_id,)
            )
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted run {}", run_id)
        return deleted

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return run summaries (no agent outputs) for the dashboard sidebar."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, objective, dataset_path, status, error,
                       created_at, updated_at
                  FROM pipeline_runs
                 ORDER BY created_at DESC, rowid DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent outputs
    # ------------------------------------------------------------------
    def save_agent_output(
        self,
        run_id: str,
        agent: AgentName,
        output_kind: str,
        payload: BaseModel | dict[str, Any],
    ) -> int:
        if output_kind not in OUTPUT_KIND_TO_MODEL:
            raise ValueError(
                f"Unknown output_kind {output_kind!r}. "
                f"Known: {sorted(OUTPUT_KIND_TO_MODEL)}"
            )
        with self._write_tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_outputs
                    (run_id, agent, output_kind, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, agent.value, output_kind, _to_json(payload), _utcnow_iso()),
            )
            return int(cur.lastrowid)

    def get_agent_output(
        self, run_id: str, output_kind: str
    ) -> BaseModel | None:
        """Latest output of the given kind, hydrated into its Pydantic model."""
        if output_kind not in OUTPUT_KIND_TO_MODEL:
            raise ValueError(f"Unknown output_kind {output_kind!r}")
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT payload FROM agent_outputs
                 WHERE run_id = ? AND output_kind = ?
                 ORDER BY created_at DESC, output_id DESC
                 LIMIT 1
                """,
                (run_id, output_kind),
            ).fetchone()
        if row is None:
            return None
        return OUTPUT_KIND_TO_MODEL[output_kind].model_validate_json(row["payload"])

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------
    def create_approval_request(self, request: ApprovalRequest) -> None:
        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO approval_requests
                    (request_id, run_id, agent, title, description, payload,
                     status, decision, response_payload, responder, comment,
                     created_at, responded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.run_id,
                    request.agent.value,
                    request.title,
                    request.description,
                    json.dumps(request.payload, default=str),
                    request.status.value,
                    request.decision.value if request.decision else None,
                    json.dumps(request.response_payload, default=str)
                        if request.response_payload is not None else None,
                    request.responder,
                    request.comment,
                    request.created_at.isoformat(),
                    request.responded_at.isoformat() if request.responded_at else None,
                ),
            )
        logger.info(
            "Approval requested [{}]: {} (run={})",
            request.request_id,
            request.title,
            request.run_id,
        )

    def get_approval_request(self, request_id: str) -> ApprovalRequest | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_approval(row) if row else None

    def respond_to_approval(self, response: ApprovalResponse) -> ApprovalRequest:
        """Apply a response. Returns the updated request (raises if not found)."""
        responded_at = response.responded_at.isoformat()
        response_payload = (
            json.dumps(response.response_payload, default=str)
            if response.response_payload is not None
            else None
        )
        with self._write_tx() as conn:
            cur = conn.execute(
                """
                UPDATE approval_requests
                   SET status = ?,
                       decision = ?,
                       response_payload = ?,
                       responder = ?,
                       comment = ?,
                       responded_at = ?
                 WHERE request_id = ? AND status = ?
                """,
                (
                    ApprovalStatus.RESOLVED.value,
                    response.decision.value,
                    response_payload,
                    response.responder,
                    response.comment,
                    responded_at,
                    response.request_id,
                    ApprovalStatus.PENDING.value,
                ),
            )
            if cur.rowcount == 0:
                raise LookupError(
                    f"No pending approval request {response.request_id!r} "
                    f"(may not exist or already resolved)"
                )
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE request_id = ?",
                (response.request_id,),
            ).fetchone()
        updated = _row_to_approval(row)
        logger.info(
            "Approval [{}] resolved: {} by {}",
            updated.request_id,
            updated.decision.value if updated.decision else "?",
            updated.responder or "?",
        )
        return updated

    def list_pending_approvals(
        self, run_id: str | None = None
    ) -> list[ApprovalRequest]:
        with self._conn() as conn:
            if run_id is None:
                rows = conn.execute(
                    """
                    SELECT * FROM approval_requests
                     WHERE status = ?
                     ORDER BY created_at ASC
                    """,
                    (ApprovalStatus.PENDING.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM approval_requests
                     WHERE status = ? AND run_id = ?
                     ORDER BY created_at ASC
                    """,
                    (ApprovalStatus.PENDING.value, run_id),
                ).fetchall()
        return [_row_to_approval(r) for r in rows]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def log_event(self, event: AgentEvent) -> None:
        # Use a monotonic timestamp for storage. The caller's event.created_at
        # may collide with the previous event when fired in rapid succession;
        # we bump by 1us so the storage order is well-defined.
        with self._ts_lock:
            ts = datetime.now(timezone.utc)
            if self._last_event_ts is not None and ts <= self._last_event_ts:
                ts = self._last_event_ts + timedelta(microseconds=1)
            self._last_event_ts = ts

        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO agent_events
                    (event_id, run_id, agent, event_type, message, payload,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.agent.value,
                    event.event_type.value,
                    event.message,
                    json.dumps(event.payload, default=str),
                    ts.isoformat(),
                ),
            )

    def get_events(
        self,
        run_id: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        """Live-trace query. `since` is exclusive (events strictly after it)."""
        with self._conn() as conn:
            if since is None:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_events
                     WHERE run_id = ?
                     ORDER BY created_at ASC, event_id ASC
                     LIMIT ?
                    """,
                    (run_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_events
                     WHERE run_id = ? AND created_at > ?
                     ORDER BY created_at ASC, event_id ASC
                     LIMIT ?
                    """,
                    (run_id, since.isoformat(), limit),
                ).fetchall()
        return [_row_to_event(r) for r in rows]


# ---------------------------------------------------------------------------
# Row → model adapters
# ---------------------------------------------------------------------------
def _row_to_event(row: sqlite3.Row) -> AgentEvent:
    return AgentEvent(
        event_id=row["event_id"],
        run_id=row["run_id"],
        agent=AgentName(row["agent"]),
        event_type=EventType(row["event_type"]),
        message=row["message"],
        payload=_from_json(row["payload"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_approval(row: sqlite3.Row) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=row["request_id"],
        run_id=row["run_id"],
        agent=AgentName(row["agent"]),
        title=row["title"],
        description=row["description"],
        payload=_from_json(row["payload"]),
        status=ApprovalStatus(row["status"]),
        decision=ApprovalDecision(row["decision"]) if row["decision"] else None,
        response_payload=_from_json(row["response_payload"])
            if row["response_payload"] is not None else None,
        responder=row["responder"],
        comment=row["comment"],
        created_at=datetime.fromisoformat(row["created_at"]),
        responded_at=datetime.fromisoformat(row["responded_at"])
            if row["responded_at"] else None,
    )
