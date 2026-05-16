#!/usr/bin/env bash
# AutoForge task runner (Bash). Linux/macOS mirror of tasks.ps1. Usage:
#   ./tasks.sh <verb>
# Verbs: setup, init-db, test-nemotron, dashboard, run, smoke, clean, help
#
# Examples:
#   ./tasks.sh setup          # conda env create / update from environment.yml
#   ./tasks.sh dashboard      # launch Streamlit
#   ./tasks.sh run            # run pipeline with a default test dataset
#   ./tasks.sh smoke          # run skeleton smoke test
#
# Note: when the NemoClaw install line lands from the hackathon portal,
# uncomment `- nemoclaw` in environment.yml + requirements.txt, then re-run
# `./tasks.sh setup`. Leave `# openclaw` commented — we chose the NemoClaw
# variant for the NVIDIA-side framework integration.

set -e
cd "$(dirname "$0")"

verb="${1:-help}"
shift || true

invoke_setup() {
    echo "Creating/updating conda env 'autoforge'..."
    conda env update -f environment.yml --prune
}

invoke_init_db() {
    echo "Initializing SQLite store..."
    python scripts/init_db.py
}

invoke_test_nemotron() {
    echo "Pinging Nemotron endpoint..."
    python scripts/test_nemotron.py
}

invoke_dashboard() {
    echo "Launching Streamlit dashboard on http://localhost:8501 ..."
    streamlit run dashboard/app.py
}

invoke_run() {
    dataset="${1:-data/fixtures/mnist}"
    objective="${2:-classify handwritten digits with accuracy >= 0.90}"
    echo "Running pipeline: dataset=$dataset objective='$objective'"
    python scripts/run_pipeline.py --dataset "$dataset" --objective "$objective"
}

invoke_smoke() {
    echo "Running skeleton smoke tests..."
    pytest tests/ -v
}

invoke_clean() {
    echo "Removing caches and SQLite DB..."
    find . -type d \( \
        -name __pycache__ -o \
        -name .pytest_cache -o \
        -name .ruff_cache -o \
        -name .mypy_cache \
    \) -prune -exec rm -rf {} + 2>/dev/null || true
    rm -f data/autoforge.db data/autoforge.db-journal \
          data/autoforge.db-wal data/autoforge.db-shm
}

invoke_help() {
    cat <<'EOF'
AutoForge tasks
---------------
setup           Create/update the 'autoforge' conda env from environment.yml
init-db         Initialize the SQLite memory store
test-nemotron   Smoke-test the NVIDIA Nemotron endpoint
dashboard       Launch the Streamlit dashboard
run [dataset] [objective]
                Run the pipeline (defaults: data/uploads/test.csv, 'predict churn')
smoke           Run the pytest smoke suite
clean           Remove caches and the SQLite DB
help            Show this message
EOF
}

case "$(echo "$verb" | tr '[:upper:]' '[:lower:]')" in
    setup)         invoke_setup ;;
    init-db)       invoke_init_db ;;
    test-nemotron) invoke_test_nemotron ;;
    dashboard)     invoke_dashboard ;;
    run)           invoke_run "$@" ;;
    smoke)         invoke_smoke ;;
    clean)         invoke_clean ;;
    help)          invoke_help ;;
    *)
        echo "Unknown verb: $verb" >&2
        invoke_help
        exit 1
        ;;
esac
