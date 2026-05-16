# AutoForge

**Multi-agent autonomous ML pipeline** — built for Hack-a-Claw x NVIDIA (UC Santa Cruz, May 15–16, 2026).

Hand AutoForge a dataset and a plain-English objective ("predict churn with F1 ≥ 0.85"). A team of specialized agents profiles the data, researches relevant approaches from live literature, picks a training envelope for the available hardware, runs HPO, optimizes the model for the target hardware, and delivers a Pareto-frontier benchmark report. A human stays in the loop via a Streamlit dashboard and a Telegram bot, approving key decisions.

## The five agents

| Agent | Role |
| --- | --- |
| **Dataset** | Profiles data, infers task type, cleans |
| **Strategy** | Formalizes the objective + researches relevant papers/architectures |
| **Hardware** | Pre-train: chooses training envelope. Post-train: TensorRT export + quantization |
| **Training** | Optuna-driven HPO within the hardware envelope |
| **Benchmark** | Accuracy + latency/throughput, builds a Pareto frontier, can feed back to Training |

A **Coordinator** orchestrates the five and brokers Human-in-the-Loop approvals.

## Stack

- **Agent framework:** OpenClaw (hackathon-mandated)
- **LLMs:** NVIDIA Nemotron via NIM
  - Coordinator + Strategy: `nvidia/llama-3_3-nemotron-super-49b-v1_5`
  - Worker agents: `nvidia/nvidia-nemotron-nano-9b-v2`
- **Compute target:** NVIDIA Brev (Cloud Track)
- **HITL:** Streamlit dashboard + Telegram bot
- **Contracts:** Pydantic v2 between every agent
- **Memory:** SQLite (Chroma vector DB later)
- **Training stack:** XGBoost + scikit-learn + Optuna (light — story is autonomy, not heavy compute)
- **Logging:** loguru

## Quickstart (Windows / PowerShell)

```powershell
# 1) Create the conda environment
conda env create -f environment.yml
conda activate autoforge

# 2) Configure secrets
Copy-Item .env.example .env
# Edit .env and fill in NVIDIA_API_KEY, TAVILY_API_KEY, TELEGRAM_*

# 3) Smoke-test the Nemotron endpoint
python scripts/test_nemotron.py

# 4) Initialize the SQLite store
python scripts/init_db.py

# 5) Launch the dashboard (terminal A)
streamlit run dashboard/app.py

# 6) Run a pipeline (terminal B)
python scripts/run_pipeline.py --dataset data/uploads/test.csv --objective "predict churn"
```

You should see the run appear live in the dashboard, a Telegram approval ping arrive, and the rest of the pipeline complete once you approve.

## Project layout

```
agents/         Five agents + base class + coordinator
contracts/      Pydantic v2 schemas + message types (the inter-agent contract layer)
memory/         SQLite schema + MemoryStore (single source of truth for state)
hitl/           Approval queue + Telegram bot + coordinator service
dashboard/      Streamlit app (live agent trace + approval UI)
tools/          Per-agent tools (ydata, Tavily, pynvml, TensorRT, sklearn metrics, ...)
scripts/        CLI entry points (run_pipeline, init_db, test_nemotron)
data/           uploads/ (input datasets), artifacts/ (model outputs) — managed by MemoryStore
tests/          Smoke tests for skeleton + HITL flow
config.py       Centralized constants
main.py         Convenience entry point
```

## Status

Skeleton-first. The five agents currently return stub Pydantic objects matching their schemas; the architecture, HITL gates, and live tracing are real. Each agent is swapped to a real OpenClaw + Nemotron implementation in subsequent phases (Dataset → Strategy → Training → Benchmark → Hardware).

## Prize tracks

| Track | Status |
| --- | --- |
| Cloud (Brev) | **Primary target** — requires Nemotron |
| Edge (DGX Spark) | Secondary |
| Bonus (NemoClaw) | Stretch — wired in last if time allows |
