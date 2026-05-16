# AutoForge

**Multi-agent autonomous ML pipeline** — built for Hack-a-Claw × NVIDIA (UC Santa Cruz, May 15–16, 2026).

Hand AutoForge a dataset and a plain-English objective ("classify handwritten digits with accuracy ≥ 0.90"). Six specialized agents, coordinated by a Director, autonomously profile the data, research candidate architectures from live literature, prepare the data, generate training code, run a real subprocess training loop, evaluate the result, and optimize the model for deployment. A human stays in the loop via a Streamlit dashboard, approving the handoff between every agent.

## The six agents

| Agent          | Role |
| ---            | --- |
| **Profiler**   | Inspects the dataset (pandas/PIL), probes hardware (psutil + nvidia-smi), and asks Nemotron for judgment fields (task type, target column, warnings). Emits `DatasetProfile` + `TrainingEnvelope`. |
| **Researcher** | Pre-fetches Tavily web search + arXiv in parallel (4 concurrent), then asks Nemotron to compose a `StrategySpec` with 1–3 candidate architectures, citations, and a research summary. |
| **Preparer**   | Enum-locked planner (8 supported operations). LLM picks an ordered plan; agent dispatches each op (`resize_images`, `train_test_split_images`, `impute_missing`, `encode_categoricals`, etc.). Programmatic split backstop guarantees a train/test layout. Writes `prep_config.json` recording normalization/scaling decisions for the Trainer. |
| **Trainer**    | Agentic-pipeline pattern. Computes a sklearn LogReg oracle baseline, asks Nemotron for `design.md` + `model.py`, runs a smoke harness (py_compile, import, `build_model()` instantiates, `train.py --help` works, design.md has 7 markdown headers), pauses for a HITL design-gate approval, then runs `train.py` as a subprocess. AutoForge owns the templated `train.py` (modality-specific, reads `prep_config.json` at runtime). On failure, retries up to 5 times with smoke errors / stderr tail fed back to the LLM. Filesystem state machine: each attempt lives under `training/in-progress/attempt-N/` and moves to `failed/` or `done/` based on outcome. |
| **Evaluator**  | Loads the trained model, runs `sklearn.metrics` on the held-out test set, measures latency p50/p95/p99 on a 100-sample probe, computes throughput, and produces a PASS/FAIL `BenchmarkReport` against the StrategySpec's success threshold. |
| **Optimizer**  | LZMA-compresses the trained model artifact (joblib `compress=9`). Real bytes saved, not stubbed. ONNX + int8 quantization is a future step. |

A **Coordinator** (`agents/coordinator.py`) orchestrates the six in a fixed order with a HITL approval gate after each. The Researcher gate is specialized — the human picks one of N candidate architectures via radio buttons; the spec is trimmed to that choice before the Preparer runs.

## Stack

- **LLMs:** NVIDIA Nemotron via NIM (`build.nvidia.com`).
  - Coordinator / planning calls: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
  - Worker calls (Profiler judgment, free-form reasoning): `nvidia/nvidia-nemotron-nano-9b-v2`.
- **Compute target:** NVIDIA Brev L40S (cloud track).
- **Sandboxing (planned):** NemoClaw `openshell` will sandbox the Trainer's `train.py` subprocess via a policy YAML (read-only access to prepared/, write to models/, no network egress).
- **HITL:** Streamlit dashboard on `localhost:8501`. Optional Slack bot (token-gated).
- **Contracts:** Pydantic v2 (`extra="forbid"`, `validate_assignment=True`) between every agent.
- **Memory:** SQLite (WAL mode, monotonic-event timestamps).
- **Training stack:** sklearn-only by deliberate constraint — LLM is hard-restricted to `MLPClassifier`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`. Keeps the demo fast and reliable on CPU; ONNX/PyTorch deferred.
- **Logging:** loguru with a dual-sink `emit_event()` (SQLite for the dashboard, stderr for the terminal).

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

For headless / CI runs:
```bash
python scripts/run_pipeline.py --dataset data/fixtures/mnist \
    --objective "..." --auto-approve
```

## Project layout

```
agents/         Six agents + Coordinator + base class + LLM client
contracts/      Pydantic v2 schemas (schemas.py) + message types (messages.py)
memory/         SQLite schema + MemoryStore (single source of truth for state)
hitl/           Approval queue + auto-approve service + (planned) Slack bot
dashboard/      Streamlit app (app.py + components/) — live agent trace, approval UI,
                attempt-history surface for the agentic Trainer
tools/          Per-agent tools: training_pipeline (codegen, smoke harness, subprocess),
                training_tools (sklearn helpers), preparation_tools (resize / split /
                impute / encode), research_tools (Tavily + arXiv), train_helpers
                (copied into each Trainer attempt as autoforge_helpers.py)
scripts/        CLI entry points (run_pipeline, init_db, test_nemotron) +
                per-agent smoke helpers (_smoke_*.py)
data/
  fixtures/     MNIST 500-PNG fixture + churn_sample.csv + sample_images
  artifacts/    Per-run outputs: profile/spec/preparation/oracle/training/done/...
  uploads/      User-supplied datasets
tests/          38 pytest tests covering contracts, MemoryStore, HITL, CLI smoke,
                end-to-end skeleton
tasks.sh        Linux task runner (setup, init-db, dashboard, run, smoke, clean)
tasks.ps1       Windows mirror
environment.yml Conda env spec
.env            API keys (gitignored)
BREV_SETUP.md   Brev / NemoClaw bootstrap
NOTES.md        Phase-by-phase running notes
```

## What's real vs. stub

Honest audit, no marketing:

**Real:** Nemotron via NIM, Profiler (pandas + PIL + LLM + real hardware probe via psutil/nvidia-smi), Researcher (parallel Tavily + arXiv + Nemotron), Preparer (resize/split/impute/encode all hit disk; `prep_config.json` persisted), Oracle baseline (real sklearn LogReg fit), LLM codegen of design.md + model.py, smoke harness (4 checks: model.py syntax + import, train.py --help, design.md markdown structure), 5-attempt retry loop with smoke + subprocess error feedback, filesystem state machine (in-progress → failed/done), subprocess training (real fit, real best.pkl), Evaluator (real joblib.load + sklearn.metrics + 100-sample latency probe), Optimizer (LZMA recompression — real byte savings), SQLite memory (WAL + monotonic events), HITL queue, Streamlit dashboard with live trace and per-attempt artifact surfacing.

**Stub / partial:** Optimizer "quantization" is just LZMA compression (no int8/fp16); ONNX export not implemented. Slack HITL bot exists but is wired via env tokens — not configured by default. NemoClaw sandboxing is planned but not yet wired; the Trainer's subprocess currently runs via plain `subprocess.run`.

## Prize tracks

| Track | Status |
| --- | --- |
| **Best Use of Nemotron** (Cloud + Brev) | ✅ Primary — six Nemotron-powered agents, real tool integration (Tavily/arXiv), visible chain-of-thought, multi-agent coordination, HITL gates, codegen + retry loop. |
| **Best Use of NemoClaw** | ⏳ Planned — sandbox the Trainer's LLM-generated subprocess via `openshell` with a YAML policy (read prepared/, write models/, no network, no shell escape). Audit log surfaces a blocked action for the demo. |
