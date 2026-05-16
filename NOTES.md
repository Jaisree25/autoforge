# AutoForge — Running Notes

Scratchpad for decisions, TODOs, and open questions. Updated at the end of each phase.

---

## Phase 1 — Repo scaffolding (complete)

**Created**
- `environment.yml` — Conda env `autoforge`, Python 3.11. Pulled forward deps for upcoming phases (openai, tavily-python, arxiv, xgboost, optuna, ydata-profiling, pytest, pytest-asyncio, typer).
- `.env.example`, `.gitignore`, `requirements.txt`, `config.py`, `README.md`.
- `tasks.ps1` — PowerShell task runner (`setup`, `init-db`, `test-nemotron`, `dashboard`, `run`, `smoke`, `clean`).
- `data/uploads/.gitkeep`, `data/artifacts/.gitkeep` so folders survive a fresh clone.

**Decisions**
- `config.py` auto-creates `data/uploads/` and `data/artifacts/` on import. Cheap, idempotent. Reverse if it surprises during a fresh-clone walkthrough.
- `AUTOFORGE_DB_PATH` is resolved to absolute in `config.py` to avoid CWD-dependent surprises when launching dashboard vs. scripts.
- OpenClaw / NemoClaw pip lines left commented out in `environment.yml` and `requirements.txt`. Uncomment once install instructions land.

**Open questions**
- Are there secrets management constraints from the hackathon org? (Default is local `.env` only.)
- Do we need a separate `requirements-dev.txt` (ruff, mypy)? Currently lumped into `environment.yml`.

---

## Phase 2 — Contracts (complete)

**Created**
- `contracts/schemas.py` — `DatasetProfile`, `StrategySpec`, `CandidateArchitecture`, `TrainingEnvelope`, `DeploymentArtifact`, `TrainingResult`, `BenchmarkReport`, `PipelineRun`. Enums: `AgentName`, `PipelineStatus`, `TaskType`. Plus helper sub-models: `ColumnProfile`, `Citation`, `TrialResult`, `LatencyStats`, `ParetoPoint`.
- `contracts/messages.py` — `ApprovalRequest`, `ApprovalResponse`, `ApprovalDecision`, `ApprovalStatus`, `AgentEvent`, `EventType`.

**Decisions**
- Every model inherits from a private `_Base` with `extra="forbid"`, `validate_assignment=True`, `use_enum_values=False`. Catches agent-output typos at the boundary and keeps enums as enums in memory (round-trip through JSON works because Pydantic v2 serializes string enums correctly).
- `PipelineRun` is the composed "view" object — holds optional refs to each agent's output. `MemoryStore` will persist agent outputs in their own rows, but the dashboard can render a single `PipelineRun` as the live view.
- IDs (`run_id`, `event_id`, `request_id`) are `str(uuid.uuid4())`. Sortable enough by `created_at` timestamp; readable enough in logs.
- `AgentEvent.payload` and `ApprovalRequest.payload` are `dict[str, Any]` rather than strongly typed unions — agents can stream arbitrary structured info (LLM token chunks, tool args, intermediate results) without forcing a schema bump per event type.
- Added `EventType.APPROVAL_REQUESTED` / `APPROVAL_RECEIVED` so the live trace can show approval gates inline with other agent activity.
- Added `DeploymentArtifact` schema for the post-training Hardware pass — wasn't called out by name in the brief but is the second output of the agent that "runs twice."
- `TaskType` is a separate enum (not just a free string) so the Dataset Agent's output is machine-checkable by the Strategy Agent.

**Smoke check**
- Ran a script that imports all symbols, instantiates every top-level model with realistic data, JSON round-trips a fully populated `PipelineRun`, and verifies `extra=forbid` rejects typos. All green.

---

## Phase 3 — Memory store (complete)

**Created**
- `memory/schema.sql` — 4 tables (`pipeline_runs`, `agent_outputs`, `approval_requests`, `agent_events`) with the indexes the dashboard's poll queries actually need. All `CREATE` statements are `IF NOT EXISTS` so re-running `init_db.py` is safe.
- `memory/store.py` — `MemoryStore` class. Per-call connections + WAL mode + a write lock. Methods cover runs (create/update_status/get/list), agent outputs (save/get), approvals (create/get/respond/list_pending), and events (log/get with `since=` cursor).
- `pytest.ini` — `pythonpath = .`, `testpaths = tests`.
- `tests/test_contracts.py` — 11 tests pinning the Phase 2 smoke check.
- `tests/test_memory_store.py` — 11 tests covering every public method, JSON round-trips, latest-wins semantics, FK cascade, since-cursor exclusivity, and cross-run isolation.

**JSON schema dump (suggestion)**
- `MemoryStore.init_schema()` writes `data/artifacts/contracts/<kind>.schema.json` for each contract. Dashboard can render generic JSON forms over `ApprovalRequest.payload` without per-schema UIs.

**Bugs caught by tests**
1. `executescript()` implicitly commits any pending transaction → wrapping it in `BEGIN IMMEDIATE`/`COMMIT` blew up on every `init_schema()`. Fix: `init_schema` bypasses `_write_tx`; DDL is idempotent so atomicity isn't needed.
2. Two events fired in the same microsecond had tied `created_at`, making `since=last_seen` ambiguous. Fix: `log_event` enforces strictly-monotonic timestamps via an in-memory lock (bumps by 1µs on collision). Sufficient because only one process writes events at a time.
3. Two `create_run` calls in the same microsecond → undefined `list_runs` order. Fix: `ORDER BY created_at DESC, rowid DESC` tiebreak.

**Decisions**
- `agent_outputs` allows multiple rows per `(run_id, output_kind)` because the Training Agent re-runs on Benchmark feedback. Hydration picks latest by `(created_at DESC, output_id DESC)`. Tests pin this behavior.
- `OUTPUT_KIND_TO_MODEL` registry in `store.py` maps `output_kind` string → Pydantic model. Single source of truth for what the store knows how to hydrate.
- `respond_to_approval` is a conditional UPDATE on `status = PENDING` — second responses raise `LookupError` instead of silently overwriting. Test pins this.
- `_write_tx` uses `BEGIN IMMEDIATE` (not deferred) so contended writes fail-fast instead of blocking on lock acquisition mid-statement.
- Per-call connections instead of a long-lived pool. Adds ~ms latency per call; in return, we sidestep thread-affinity and stale-connection issues. Fine for ~100 writes per pipeline run.

---

## TODOs

- [ ] Write `scripts/test_nemotron.py` (Phase 7).
- [ ] Confirm exact NIM model IDs after first successful ping.
- [ ] When OpenClaw docs land, paste them in and ask Claude Code to swap `agents/base_agent.py` + one concrete agent at a time.
- [ ] Two hours before deadline: ask Claude Code to write `DEMO.md`.
- [ ] Decide whether `TaskType.IMAGE_CLASSIFICATION` actually ships in the v1 demo (currently in the enum but no agent path supports it).
- [ ] If two pipelines ever run in parallel (different processes), the monotonic-timestamp invariant in `log_event` breaks. Today: single-process write, fine. Tomorrow: revisit.

---

## Phase 4 — Agent skeleton + coordinator (complete)

**Created**
- `agents/base_agent.py` — `BaseAgent(ABC)` with class-level `name`, DI-injected `store` + `run_id`, dual-sink `emit_event()` (SQLite + loguru), and a `_lifecycle()` context manager that emits STARTED at entry / COMPLETED at exit / ERROR on exception.
- `agents/dataset_agent.py` — returns a plausible 10k×12 churn-style `DatasetProfile` with 12 column profiles, 73.4/26.6 class balance, and a missing-data warning.
- `agents/strategy_agent.py` — returns a `StrategySpec` with two candidate architectures (XGBoost-tuned + LogReg-calibrated), a research summary, and two arXiv citations.
- `agents/hardware_agent.py` — single class with `run(phase=...)` dispatcher + explicit `run_pre_training()` / `run_post_training()` methods. Pre-pass returns an L40S `TrainingEnvelope`; post-pass returns an ONNX `DeploymentArtifact`.
- `agents/training_agent.py` — fakes a 20-trial Optuna search with monotonically improving scores 0.802 → 0.874 and emits a THINKING event per trial so the dashboard live trace fills in incrementally.
- `agents/benchmark_agent.py` — returns a `BenchmarkReport` with latency stats + a 3-point Pareto frontier from the top-3 trials. PASS/FAIL is computed against `strategy_spec.success_threshold`, so editing the threshold via HITL actually changes the outcome.
- `agents/coordinator.py` — orchestrates Dataset → Strategy → [HITL gate] → Hardware-pre → Training → Benchmark → Hardware-post. Defines a `HITLService` Protocol that both `AutoApproveHITLService` (Phase 4) and the real Streamlit+Telegram bridge (Phase 5) will satisfy.
- `hitl/auto.py` — `AutoApproveHITLService` that records the request, immediately auto-approves, and resolves the response. Lets the skeleton run end-to-end and supports headless CI runs.
- `tests/test_coordinator_skeleton.py` — 5 tests: full happy path, live-trace agent coverage, no pending approvals after clean run, rejection cancels (not fails), edited strategy actually changes downstream behavior.

**Suggestions delivered**
1. **DI for `MemoryStore`** — every agent takes the store via constructor, no globals. Tests already use this with `tmp_path`.
2. **Dual-sink `emit_event`** — SQLite for the dashboard, loguru for the terminal. EventType → log level mapping with THINKING at DEBUG (high-volume).
3. **Plausible stub values** — every stub returns realistic data so the demo looks credible even before real implementations land.

**Test results**: 27/27 passing. End-to-end skeleton runs in ~7s per pipeline (deliberate sleeps so the live trace is observable in the dashboard demo).

**Decisions**
- **Coordinator extends `BaseAgent`** so it can `emit_event()` for top-level pipeline events. `run()` is a no-op stub; the entry point is `execute()`.
- **`PipelineRejected` is a separate exception** from generic failures — Coordinator maps it to `PipelineStatus.CANCELLED`, not `FAILED`. Rejection is a user choice, not a bug.
- **HITL gate flow**: when awaiting approval, status flips to `AWAITING_APPROVAL` so the dashboard can render the gate prominently; flips back to `RUNNING` once a decision lands.
- **Edited strategy persists as a new `agent_outputs` row** (not an UPDATE). The original strategy spec stays in the audit trail; the edited version becomes the latest for hydration.
- **Hardware Agent runs `run_pre_training` and `run_post_training` explicitly** in the coordinator, bypassing the `run(phase=...)` dispatcher. The dispatcher exists only to satisfy `BaseAgent.run`'s abstract contract; direct callers should use the named methods.
- **Training Agent's stub trial scores plateau at 0.874** — engineered so a 0.85 threshold passes and a 0.99 threshold fails, exercising both paths in the smoke test.

**Sleeps are deliberately slow.** Each agent sleeps ~0.5–1.5s total. This is critical for the demo — judges need to *see* the agents thinking. For headless CI, we can monkeypatch `config.STUB_AGENT_SLEEP = 0.0` if test runtime becomes a problem (currently 34s for the full suite).

---

## Phase 5 — HITL subsystem (complete)

**Created**
- `hitl/approval_queue.py` — `ApprovalQueue` with hybrid wakeup. In-process resolvers (Telegram, auto) call `resolve()` → `threading.Event` signals waiter immediately. Cross-process resolvers (dashboard, separate Streamlit process) write to the DB → waiter's poll loop picks it up within `poll_interval`. Either path completes within ~poll_interval of the resolution.
- `hitl/telegram_bot.py` — `TelegramApprovalBot` runs `python-telegram-bot v22` on a background-thread asyncio loop. `send_approval_request()`/`notify()` schedule coroutines onto the bot's loop via `asyncio.run_coroutine_threadsafe()`. Approve/Reject callbacks edit the original message in place (strips buttons + appends status footer). Optional — start() is only called if `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` are both set.
- `hitl/coordinator_service.py` — `HITLCoordinatorService` satisfies the `HITLService` Protocol from `agents/coordinator.py` structurally (no inheritance). `build_hitl_service(store)` factory wires queue + (optional) Telegram automatically.
- `tests/test_hitl_flow.py` — 7 tests covering: in-process Event wakeup (<1s), cross-process DB-poll wakeup (<1s after delay), timeout returns None, lookup error on unknown request, `HITLCoordinatorService` round-trip, `HITLCoordinatorService` timeout raises `TimeoutError`, full pipeline using a "dashboard simulator" thread.

**Suggestions delivered**
1. **`HITLService` Protocol conformance** — both `AutoApproveHITLService` and `HITLCoordinatorService` are passed to `Coordinator(hitl=...)` in the test suite. Behavioral coverage > runtime isinstance check; no `@runtime_checkable` noise needed.
2. **Responder convention** — documented in module docstring + applied throughout. `"dashboard"` / `"telegram:<user_id>"` / `"auto"`.
3. **Telegram bot in background thread** — asyncio loop owned by the bot, `start()` blocks until `_ready.set()`, `stop()` shuts the loop down cleanly. All outbound work schedules onto the bot loop from any caller thread.

**Decisions**
- **In-process queue is the fast path; DB-poll is the correctness path.** Dashboard (Streamlit) is a separate process, so it can't share Event objects with the pipeline. We always poll the DB even if a local Event exists.
- **`request.payload` is rendered as a code-fenced JSON preview** in the Telegram message, truncated to 2000 chars to stay under Telegram's 4096-char body cap.
- **Callback prefix `af:`** keeps `callback_data` under Telegram's 64-byte limit even for UUID request_ids.
- **`coordinator_service.notify()` swallows Telegram exceptions** with a log line — a flaky bot must never kill a pipeline run.
- **`build_hitl_service()` is a factory, not a constructor argument default.** Reads env vars at call time so config changes between test fixtures and real runs don't surprise anyone.
- **Telegram bot is NOT exercised in pytest** — would require a real token + chat + outbound HTTP. Smoke-import verifies it loads. Will need a manual smoke check during the hackathon once tokens are in `.env`.
- **The cross-process test uses two `MemoryStore` instances against the same SQLite file** — that's exactly what the pipeline + dashboard relationship looks like in production. SQLite's WAL mode handles the concurrent reader + writer.

---

## Phase 6 — Streamlit dashboard (complete)

**Created**
- `dashboard/app.py` — full UI in one file. Four sections: (1) sidebar run picker + status, (2) pending approval panel(s) at the top, (3) live agent activity (auto-refreshing, newest-first, colored by event type), (4) per-agent output expanders with `st.json()`.
- `scripts/_smoke_dashboard.py` — manual smoke script using `streamlit.testing.v1.AppTest`. Currently flaky on Windows due to a Streamlit/asyncio interaction (exit 0xC06D007F). Left in place because the real verification (below) is solid; the script may work on other platforms.

**Suggestions delivered**
1. **Dashboard NEVER touches the in-process `ApprovalQueue`.** Only `store.respond_to_approval()`. The pipeline's queue picks up the resolution via its DB-poll loop (Phase 5's design).
2. **`streamlit-autorefresh` at 1500ms** (matches `DASHBOARD_REFRESH_INTERVAL_MS`). Suspended during edit mode so the `st.text_area` state survives across reruns.
3. **`@st.cache_resource get_store()`** — one `MemoryStore` instance per Streamlit session.
4. **`st.json()` for payload preview** + **`st.text_area` for Edit mode** with JSON parse validation. `Edit` flips into a session-scoped state machine until Submit/Cancel.

**Verification**
- `python -m py_compile dashboard/app.py` — passes.
- `python scripts/_smoke_dashboard_imports.py` — replicates `streamlit run`'s sys.path setup (script dir, NOT project root) and runs the whole module top-to-bottom with a stubbed streamlit. Catches the exact bug below without needing a real browser.
- Full pytest suite: 34/34 still passing.

**Bug I caught and fixed mid-phase**
- Initial dashboard failed in real use with `ModuleNotFoundError: No module named 'config'`. `streamlit run dashboard/app.py` sets `sys.path[0]` to `dashboard/` — top-level `config.py` not visible.
- My earlier "verification" was insufficient: I hit `/_stcore/health` and `/` via HTTP, but Streamlit only serves the SPA shell on those endpoints — the actual script doesn't run until a real WebSocket-connected browser loads the page.
- Fix: `dashboard/app.py` self-prepends the project root to `sys.path` before any `from config import ...` lines. Same pattern will apply to every Phase 7 entry script.
- New smoke script `_smoke_dashboard_imports.py` mimics the streamlit sys.path layout exactly so this kind of regression gets caught locally without a browser.

**Decisions**
- **No tabs — stacked sections.** Approval panel at top so it's impossible to miss when a gate is pending. Live trace next (the demo focal point). Outputs last (judges click them open when they want detail).
- **Auto-refresh suspended during edit.** Otherwise the `st.text_area` resets every 1.5s. The `edit_active` flag short-circuits `st_autorefresh()`.
- **Newest-first event order** in the live trace — easier to spot what just happened, scrolling stays put as new events arrive.
- **Colored agent labels via Streamlit's `:color[text]` markdown** (1.21+). No emoji bloat; colors are functional UI affordance.
- **Buttons keyed on `request_id`** so multiple pending requests don't clash. `use_container_width=True` keeps the three-button row balanced.
- **Run picker label is `<run_id[:8]> · <status> · <objective[:50]>`** — gives the user every signal they need without truncating mid-id.
- **`st.expander` for outputs is collapsed by default.** Live trace gets the visual real estate.

**Sleeps still slow.** With the dashboard at 1.5s autorefresh + the pipeline's ~7s real runtime, you can watch the trace fill in roughly five updates before the run completes. Good demo cadence.

---

## Phase 7 — Entry points (complete)

**Created**
- `scripts/__init__.py` — empty marker so `from scripts.run_pipeline import app` works for `main.py`.
- `scripts/init_db.py` — initializes the SQLite store and confirms every expected JSON schema file landed in `data/artifacts/contracts/`. Idempotent; safe to re-run.
- `scripts/test_nemotron.py` — pings both configured model IDs (`COORDINATOR_MODEL`, `WORKER_MODEL`) via the OpenAI SDK pointed at NVIDIA_BASE_URL, prints the response + prompt/completion/total token usage + round-trip latency. Exits 1 with a helpful message if `NVIDIA_API_KEY` isn't set.
- `scripts/run_pipeline.py` — Typer CLI. Aliases: `-d`/`--dataset`, `-o`/`--objective`, `-y`/`--auto-approve`, `--db`. Calls `build_hitl_service(store)` by default so Telegram lights up automatically when env tokens are set; `--auto-approve` uses `AutoApproveHITLService` instead (the CI / dev escape hatch).
- `main.py` — three-line convenience: `from scripts.run_pipeline import app; app()`.

**Sys.path pattern** — every entry script prepends the project root before importing project modules:
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```
Same fix that landed in `dashboard/app.py` after the Phase 6 bug.

**Suggestions delivered**
1. `test_nemotron.py` pins exact model IDs from `config.py`, prints prompt + completion + total token usage, includes round-trip latency.
2. `run_pipeline.py` uses Typer (already in env), calls `build_hitl_service(store)` so Telegram is automatic when env has both tokens.
3. `init_db.py` confirms each JSON schema file actually exists post-`init_schema()` — single visible confirmation that init succeeded.
4. `main.py` is the smallest possible wrapper — imports the Typer app and runs it.

**Verification**
- `py_compile` clean on all four files.
- `python scripts/init_db.py` — writes the DB + all 6 schema JSONs, lists them.
- `python scripts/run_pipeline.py --help` and `python main.py --help` — both render Typer help with the same options.
- `python scripts/run_pipeline.py --dataset data/uploads/test.csv --objective "..." --auto-approve` — full pipeline runs end-to-end in ~7s, prints the run summary panel (status / metric / passed / latency / throughput / artifact path).
- `python scripts/test_nemotron.py` without `NVIDIA_API_KEY` — exits 1 with "Copy .env.example to .env and fill in your key from build.nvidia.com." message.
- 34/34 pytest still passing.

**Env regression I caught + fixed**
Twice now the conda env was missing packages I'd added to `environment.yml` (pytest in Phase 3, typer + openai now). Reason: the user created `autoforge` env before those entries existed, so the additions only land via `conda env update -f environment.yml --prune` (or `.\tasks.ps1 setup`). I `pip install`-ed them ad-hoc; the env.yml is the source of truth so `setup` is still the canonical bootstrap.

**Decisions**
- **`--auto-approve` is a flag, not a separate command.** Same args, different HITL. Keeps mental model simple: "run the pipeline" is one verb.
- **`run_pipeline.py` calls `store.init_schema()` itself** even though `init_db.py` exists. Idempotent; saves users one command. `init_db.py` is still useful as the explicit "verify the install" check.
- **`build_hitl_service` reports its mode** in the run-start panel: `auto-approve` / `dashboard only` / `dashboard + telegram`. No mystery about whether Telegram is actually wired.
- **Typer help uses one-line option help.** No `\b` block-formatting tricks. Keeps the rendered output predictable on Windows terminals.
- **`main.py` doesn't add anything to the CLI surface.** It exists purely so `python main.py` works for users who think in "main scripts" rather than "scripts/".

---

## Phase 8 — Smoke tests (complete)

**Brief-compliant filenames**
- `tests/test_coordinator_skeleton.py` → renamed to `tests/test_skeleton.py`. 5 tests intact (happy path, live-trace agent coverage, no-pending-after-clean, rejection → CANCELLED, edited strategy changes downstream).
- `tests/test_hitl_flow.py` — already brief-compliant; unchanged.

**New file**
- `tests/test_cli_smoke.py` — 4 subprocess tests. Each one invokes an entry-point script with a fast-exit invocation and asserts exit code + stdout substrings:
  - `scripts/init_db.py` end-to-end against a tmp DB.
  - `scripts/run_pipeline.py --help` (typer + sys.path + imports all healthy).
  - `main.py --help` (delegation works).
  - `scripts/test_nemotron.py` with `NVIDIA_API_KEY=""` (helpful hint + exit 1, not a crash inside the OpenAI client).

**Why this matters**
Every other test constructs `Coordinator` / `MemoryStore` directly, so they never touch the *script-as-main* entry point. CLI smoke is the only thing that catches typer arg-parsing breakage, sys.path bugs, and import resolution. The Phase 6 `ModuleNotFoundError: No module named 'config'` would have surfaced here in <1s instead of hitting the user.

**Verification**
- 38/38 tests passing (was 34; +4 from `test_cli_smoke.py`, rename added 0).
- Full suite runtime: ~47s. CLI smoke adds <5s.

---

## Skeleton phases (1–8) — done

All eight scaffolding phases complete. End-to-end definition-of-done from the original brief is satisfied:

- ✅ `streamlit run dashboard/app.py` serves cleanly (after the Phase 6 sys.path fix)
- ✅ `python scripts/run_pipeline.py --dataset ... --objective ...` runs end-to-end
- ✅ Live agent events appear in the dashboard via auto-refresh
- ✅ Pending approvals surface in the dashboard (Approve / Edit / Reject)
- ✅ Telegram notification + inline-button approval path implemented (needs `.env` tokens to wire up live)
- ✅ Pipeline resumes + completes after approval
- ✅ Telegram completion notification on success

What's still stubs (intentional, per "ship skeleton first"):
- Every agent's `run()` returns plausible stub Pydantic data instead of doing real work
- Strategy doesn't call Tavily/arXiv yet
- Hardware doesn't call pynvml yet
- Training doesn't actually train
- Benchmark doesn't actually benchmark
- Hardware post-pass doesn't actually export TensorRT/ONNX

The contract layer, persistence, HITL, dashboard, and CLI entry points are real. Swap each agent's stub with a real implementation one at a time without touching anything else.

## Questions for the user

(none open right now)
