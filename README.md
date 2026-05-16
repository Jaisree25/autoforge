# AutoForge

**Multi-agent autonomous ML pipeline** — built for Hack-a-Claw × NVIDIA (UC Santa Cruz, May 15–16, 2026).

Hand AutoForge a dataset and a plain-English objective ("classify handwritten digits with accuracy ≥ 0.90"). Six specialized agents, coordinated by a Director, autonomously profile the data, research candidate architectures from live literature, prepare the data, generate training code, run a real subprocess training loop, evaluate the result, and optimize the model for deployment. A human stays in the loop via a Streamlit dashboard, approving the handoff between every agent.

## The six agents

| Agent          | Role |
| ---            | --- |
| **Profiler**   | Nemotron drives the inspection — calls pandas tools to read the CSV and psutil/nvidia-smi tools to probe hardware, then emits `DatasetProfile` + `TrainingEnvelope` (task type, target column, warnings). |
| **Researcher** | Nemotron calls Tavily web-search + arXiv tools (4 concurrent fetches) to ground the design choice, then composes a `StrategySpec` with 1–3 candidate sklearn architectures, citations, and a research summary. |
| **Preparer**   | Nemotron picks an ordered plan from an enum-locked set of 8 supported ops (`drop_columns`, `impute_missing`, `encode_categoricals`, `scale_features`, `train_test_split`, etc.); the agent dispatches each op safely. A programmatic split backstop guarantees a train/test layout. Writes `prep_config.json` recording scaling decisions for the Trainer. |
| **Trainer**    | Linear flow: Nemotron writes `design.md` → HITL approval gate → Nemotron writes `model.py` (structured choice: sklearn class + hyperparameters) → AutoForge synthesizes it + drops in a templated `train.py` → smoke harness (py_compile, import, `build_model()` instantiates, `train.py --help` works) → `train.py` runs as a subprocess with an Optuna HP search around `build_model()`. On benchmark failure, the Evaluator's feedback is fed back to Nemotron for up to 2 retries; each attempt lives under `training/attempt-N/`. |
| **Evaluator**  | Loads the trained model, runs `sklearn.metrics` on the held-out test set, measures latency p50/p95/p99 on a 100-sample probe, computes throughput, and produces a PASS/FAIL `BenchmarkReport` against the StrategySpec's success threshold. |
| **Optimizer**  | LZMA-compresses the trained model artifact (joblib `compress=9`). Real bytes saved, not stubbed. ONNX + int8 quantization is a future step. |

A **Coordinator** (`agents/coordinator.py`) orchestrates the six in a fixed order with a HITL approval gate after each. The Researcher gate is specialized — the human picks one of N candidate architectures via radio buttons; the spec is trimmed to that choice before the Preparer runs.

## Stack

- **LLMs:** NVIDIA Nemotron via NIM (`build.nvidia.com`).
  - Coordinator / planning calls: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
  - Worker calls (Profiler judgment, free-form reasoning): `nvidia/nvidia-nemotron-nano-9b-v2`.
- **Sandboxing (planned):** NemoClaw `openshell` will sandbox the Trainer's `train.py` subprocess via a policy YAML (read-only access to prepared/, write to models/, no network egress).
- **HITL:** Streamlit dashboard on `localhost:8501`. Optional Slack bot (token-gated).
- **Contracts:** Pydantic v2 (`extra="forbid"`, `validate_assignment=True`) between every agent.
- **Memory:** SQLite.

## Quickstart (Brev / Linux)

See [`BREV_SETUP.md`](BREV_SETUP.md) for the full bootstrap. Short version:

```bash
# 1. Conda env
./tasks.sh setup
conda activate autoforge

# 2. Configure secrets
cp .env.example .env
# Edit .env: paste NVIDIA_API_KEY (from build.nvidia.com) + TAVILY_API_KEY.

# 3. Smoke-test the Nemotron endpoint
./tasks.sh test-nemotron

# 4. Initialize the SQLite store
./tasks.sh init-db

# 5. Launch the dashboard (Terminal A)
./tasks.sh dashboard

# 6. Run a pipeline (Terminal B)
python scripts/run_pipeline.py \
    --dataset data/fixtures/mnist \
    --objective "classify handwritten digits with accuracy >= 0.90"
```

The run appears live in the dashboard; you approve the handoff between each agent. The Trainer's design-gate pauses for human review of the proposed `design.md`. The full pipeline takes ~3–5 minutes end-to-end on a CPU-only box, ~1–2 minutes on an L40S.

## Project layout

```
agents/         Six agents + Coordinator + base class + LLM client
contracts/      Pydantic v2 schemas (schemas.py) + message types (messages.py)
memory/         SQLite schema + MemoryStore (single source of truth for state)
hitl/           HITLCoordinatorService + ApprovalQueue (SQLite-backed, hybrid
                threading.Event + DB-poll wakeup) + auto-approve service.
                  slack_bot.py        — host-side Slack bot (in-process; reply
                                        parsing for CONFIRM/REJECT/digit picks,
                                        per-agent channel routing)
                  slack_bot_runner.py — long-lived entrypoint to run the same
                                        bot inside a NemoClaw sandbox
                  telegram_bot.py     — alternative HITL surface
dashboard/      Streamlit app (app.py + components/) — live agent trace, approval UI,
                attempt-history surface for the agentic Trainer
tools/          Per-agent tools: training_pipeline (codegen, smoke harness, subprocess),
                training_tools (sklearn helpers), preparation_tools (split /
                impute / encode / scale), research_tools (Tavily + arXiv),
                train_helpers (copied into each Trainer attempt as
                autoforge_helpers.py), optuna_search (per-class HP search spaces)
policies/       NemoClaw / OpenShell YAML policies (e.g. autoforge-slack-bot.yaml
                — slack-only egress preset for the sandboxed bot)
scripts/        CLI entry points (run_pipeline, init_db, test_nemotron) +
                per-agent smoke helpers (_smoke_*.py) +
                nemoclaw_slack_setup.sh (one-time sandbox bridge setup)
data/
  fixtures/     iris.csv / titanic.csv / housing.csv / churn_sample.csv
                (+ legacy MNIST PNG fixture, unused since the sklearn-only constraint)
  artifacts/    Per-run outputs: profile/spec/preparation/training/attempt-N/...
  uploads/      User-supplied datasets
tests/          pytest suite (contracts, MemoryStore, HITL, CLI smoke,
                end-to-end skeleton) — 23 tests at last count
tasks.sh        Linux task runner (setup, init-db, dashboard, run, smoke, clean,
                slack-bot-setup, slack-bot-up, slack-bot-down, slack-bot-logs)
tasks.ps1       Windows mirror
environment.yml Conda env spec
.env            API keys (gitignored)
BREV_SETUP.md   Brev / NemoClaw bootstrap
NOTES.md        Phase-by-phase running notes
```
