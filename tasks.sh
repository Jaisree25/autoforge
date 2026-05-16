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

invoke_slack_bot_setup() {
    bash "$(dirname "$0")/scripts/nemoclaw_slack_setup.sh"
}

invoke_slack_bot_up() {
    sandbox="${SANDBOX:-autoforge}"
    log_dir="$(dirname "$0")/data"
    log_file="$log_dir/slack-bot.log"
    pid_file="$log_dir/slack-bot.pid"
    mkdir -p "$log_dir"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "Slack bridge already running (PID $(cat "$pid_file")). Use slack-bot-down first."
        return 1
    fi

    echo "Launching Slack bridge inside sandbox '$sandbox'..."
    # Bot reads SLACK_BOT_TOKEN/SLACK_CHANNEL_ID via gateway-substituted
    # openshell:resolve:env:* placeholders. Source + DB live in the shared
    # mount at /sandbox/autoforge.
    nohup openshell sandbox exec -n "$sandbox" -- bash -c \
        'cd /sandbox/autoforge && AUTOFORGE_DB_PATH=/sandbox/autoforge/autoforge.db PYTHONPATH=/sandbox/autoforge /sandbox/autoforge/.venv/bin/python -m hitl.slack_bot_runner' \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    sleep 1
    if kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "Started PID $(cat "$pid_file"). Logs: $log_file"
    else
        echo "Failed to start. Check $log_file"
        rm -f "$pid_file"
        return 1
    fi
}

invoke_slack_bot_down() {
    pid_file="$(dirname "$0")/data/slack-bot.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill "$pid" 2>/dev/null; then
            echo "Stopped Slack bridge (PID $pid)"
        else
            echo "PID $pid not running"
        fi
        rm -f "$pid_file"
    else
        pkill -f "hitl.slack_bot_runner" 2>/dev/null \
            && echo "Killed any matching slack_bot_runner processes" \
            || echo "No Slack bridge process found"
    fi
}

invoke_slack_bot_logs() {
    log_file="$(dirname "$0")/data/slack-bot.log"
    if [ ! -f "$log_file" ]; then
        echo "No log file yet at $log_file — has slack-bot-up been run?"
        return 1
    fi
    tail -f "$log_file"
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

slack-bot-setup  One-time: relocate Slack bot into the NemoClaw sandbox
                 (installs deps, share-mounts DB+source, locks down policy
                 to slack-only egress). Run before slack-bot-up.
slack-bot-up     Launch the Slack bridge inside the sandbox (background)
slack-bot-down   Stop the Slack bridge
slack-bot-logs   Tail data/slack-bot.log (Ctrl-C to detach)

help            Show this message
EOF
}

case "$(echo "$verb" | tr '[:upper:]' '[:lower:]')" in
    setup)             invoke_setup ;;
    init-db)           invoke_init_db ;;
    test-nemotron)     invoke_test_nemotron ;;
    dashboard)         invoke_dashboard ;;
    run)               invoke_run "$@" ;;
    smoke)             invoke_smoke ;;
    clean)             invoke_clean ;;
    slack-bot-setup)   invoke_slack_bot_setup ;;
    slack-bot-up)      invoke_slack_bot_up ;;
    slack-bot-down)    invoke_slack_bot_down ;;
    slack-bot-logs)    invoke_slack_bot_logs ;;
    help)              invoke_help ;;
    *)
        echo "Unknown verb: $verb" >&2
        invoke_help
        exit 1
        ;;
esac
