-- AutoForge persistent state — SQLite schema.
--
-- Four tables:
--   pipeline_runs       one row per end-to-end run (status + objective)
--   agent_outputs       structured outputs from each agent (DatasetProfile, StrategySpec, ...)
--   approval_requests   HITL gates with their decisions
--   agent_events        the live-trace event stream surfaced by the dashboard
--
-- Apply with `python scripts/init_db.py` (which calls MemoryStore.init_schema()).
-- All statements are idempotent (`IF NOT EXISTS`).

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- pipeline_runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        TEXT PRIMARY KEY,
    objective     TEXT NOT NULL,
    dataset_path  TEXT NOT NULL,
    status        TEXT NOT NULL,                  -- PipelineStatus enum
    error         TEXT,
    created_at    TEXT NOT NULL,                  -- ISO-8601 UTC
    updated_at    TEXT NOT NULL                   -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_runs_status_created
    ON pipeline_runs (status, created_at DESC);

-- ---------------------------------------------------------------------------
-- agent_outputs
--   Multiple rows per (run_id, output_kind) allowed: the Training Agent may
--   re-run on Benchmark feedback. Latest row by created_at is the "current"
--   output for hydration.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_outputs (
    output_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    agent         TEXT NOT NULL,                  -- AgentName enum
    output_kind   TEXT NOT NULL,                  -- field name on PipelineRun
    payload       TEXT NOT NULL,                  -- JSON-serialized Pydantic model
    created_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outputs_run_kind_created
    ON agent_outputs (run_id, output_kind, created_at DESC);

-- ---------------------------------------------------------------------------
-- approval_requests
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS approval_requests (
    request_id        TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    agent             TEXT NOT NULL,              -- AgentName enum
    title             TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    payload           TEXT NOT NULL,              -- JSON
    status            TEXT NOT NULL,              -- ApprovalStatus enum
    decision          TEXT,                       -- ApprovalDecision enum, NULL while pending
    response_payload  TEXT,                       -- JSON, NULL until resolved
    responder         TEXT,
    comment           TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    responded_at      TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_approvals_status_run
    ON approval_requests (status, run_id);

CREATE INDEX IF NOT EXISTS idx_approvals_run_created
    ON approval_requests (run_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- agent_events
--   High write volume during a run. The dashboard polls this with
--   (run_id, created_at > last_seen) — the index is shaped for that query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_events (
    event_id      TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    agent         TEXT NOT NULL,                  -- AgentName enum
    event_type    TEXT NOT NULL,                  -- EventType enum
    message       TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL DEFAULT '{}',     -- JSON
    created_at    TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run_created
    ON agent_events (run_id, created_at);

-- ---------------------------------------------------------------------------
-- slack_posted
--   Tracker for approvals the Slack bot has already posted to a channel.
--   The bot runs in a separate process (NemoClaw sandbox) and discovers
--   new gates by polling `approval_requests` for rows whose request_id
--   is NOT in this table. After a successful chat_postMessage, the bot
--   inserts the request_id here so the next poll skips it.
--   PRIMARY KEY on request_id makes re-posts a no-op on retries.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slack_posted (
    request_id    TEXT PRIMARY KEY,
    posted_at     TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES approval_requests(request_id)
        ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- slack_outbox
--   Host-side notification queue drained by the Slack bot.
--   `notify()` used to call `slack.notify()` synchronously; with the bot
--   in a separate process we persist the notification here and let the
--   bot tail this table. `sent_at IS NULL` means undelivered.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slack_outbox (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    agent         TEXT,                            -- AgentName enum, NULL = main channel
    message       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    sent_at       TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outbox_unsent
    ON slack_outbox (sent_at, created_at);
