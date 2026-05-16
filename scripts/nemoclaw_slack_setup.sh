#!/usr/bin/env bash
# scripts/nemoclaw_slack_setup.sh
#
# One-time setup to relocate the AutoForge Slack bot from the host pipeline
# into the NemoClaw `autoforge` sandbox. After this runs:
#
#   - The bot lives inside the sandbox under an `autoforge-slack-bot` policy
#     that allows ONLY egress to slack.com / api.slack.com / hooks.slack.com.
#   - The Slack bot token + app token resolve via the OpenShell gateway
#     store (placeholders on disk, real values injected at egress) — they
#     no longer need to live in `.env` on host.
#   - The host pipeline and the sandboxed bot share a single SQLite DB at a
#     mount that's visible to both.
#
# Run this BEFORE `./tasks.sh slack-bot-up`. Idempotent — safe to re-run.
#
# Assumptions:
#   - A NemoClaw sandbox named `autoforge` already exists (you ran
#     `nemoclaw onboard` at some point — `~/.nemoclaw/sandboxes.json`
#     shows it).
#   - Slack credentials are already registered in the gateway store
#     (visible via `openshell provider list` — they were stored when you
#     did `nemoclaw autoforge channels add slack` during onboard).
#   - `nemoclaw` and `openshell` are on PATH.
#
# What it does, in order:
#   1. Confirms the sandbox is up.
#   2. Installs the bot's Python dependencies inside the sandbox (this
#      requires the default `pypi` preset to still be active — we install
#      BEFORE locking down).
#   3. Share-mounts `/sandbox/autoforge` (sandbox-side) onto
#      `<repo>/data/sandbox-share` (host-side). Both see the same files.
#   4. Copies (or refreshes) the AutoForge source dirs the bot imports.
#   5. Migrates the SQLite DB into the shared mount.
#   6. Tightens the network policy: removes `pypi`, `npm`, `huggingface`,
#      `brew`, `brave`; adds the `autoforge-slack-bot` preset (Slack-only).
#   7. Prints the `.env` lines you should set on host.

set -euo pipefail

SANDBOX="${SANDBOX:-autoforge}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARE_DIR_SANDBOX="/sandbox/autoforge"
SHARE_DIR_HOST="$REPO_DIR/data/sandbox-share"
BOT_PYDEPS=(loguru "slack_sdk" pydantic python-dotenv)

step() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
step "1/7  Verify sandbox '$SANDBOX' exists"
if ! nemoclaw list --json 2>/dev/null | grep -q "\"$SANDBOX\""; then
    echo "ERROR: sandbox '$SANDBOX' not found. Run 'nemoclaw onboard' first." >&2
    exit 1
fi
echo "OK"

# ---------------------------------------------------------------------------
step "2/7  Install bot Python deps inside sandbox (needs pypi preset still active)"
# These are tiny; reuse host pip cache via openshell-injected proxy.
openshell sandbox exec -n "$SANDBOX" -- pip install --user "${BOT_PYDEPS[@]}"

# ---------------------------------------------------------------------------
step "3/7  Share-mount sandbox $SHARE_DIR_SANDBOX onto $SHARE_DIR_HOST"
mkdir -p "$SHARE_DIR_HOST"
# Ensure target dir exists inside sandbox before mount.
openshell sandbox exec -n "$SANDBOX" -- mkdir -p "$SHARE_DIR_SANDBOX"
# `share mount` is idempotent — if already mounted, it no-ops or replaces.
nemoclaw "$SANDBOX" share mount "$SHARE_DIR_SANDBOX" "$SHARE_DIR_HOST" || true

# ---------------------------------------------------------------------------
step "4/7  Refresh source dirs the bot needs (host → shared mount)"
# Anything imported by hitl/slack_bot_runner.py must be present:
#   hitl, memory, contracts, agents (for BaseAgent), config.py, .env (optional)
for path in hitl memory contracts agents config.py; do
    if [ -e "$REPO_DIR/$path" ]; then
        echo "  copying $path"
        cp -r "$REPO_DIR/$path" "$SHARE_DIR_HOST/"
    fi
done

# ---------------------------------------------------------------------------
step "5/7  Migrate SQLite DB into the shared mount"
HOST_DB="$REPO_DIR/data/autoforge.db"
SHARED_DB="$SHARE_DIR_HOST/autoforge.db"
if [ -f "$HOST_DB" ] && [ ! -f "$SHARED_DB" ]; then
    cp "$HOST_DB" "$SHARED_DB"
    cp "$HOST_DB-wal" "$SHARED_DB-wal" 2>/dev/null || true
    cp "$HOST_DB-shm" "$SHARED_DB-shm" 2>/dev/null || true
    echo "Migrated $HOST_DB → $SHARED_DB"
elif [ -f "$SHARED_DB" ]; then
    echo "Shared DB already exists at $SHARED_DB (skipping copy)"
else
    echo "No existing DB found; bot will initialize at $SHARED_DB on first start"
fi

# ---------------------------------------------------------------------------
step "6/7  Lock down network policy (remove all but slack)"
for p in pypi npm huggingface brew brave; do
    nemoclaw "$SANDBOX" policy-remove "$p" -y 2>/dev/null || true
done
nemoclaw "$SANDBOX" policy-add --from-file "$REPO_DIR/policies/autoforge-slack-bot.yaml" -y

echo "Applied policies (should show slack + autoforge-slack-bot only):"
nemoclaw "$SANDBOX" policy-list || true

# ---------------------------------------------------------------------------
step "7/7  Done — next steps"
cat <<EOF

Update your .env (on host) to point at the shared DB:

    AUTOFORGE_DB_PATH=$SHARED_DB

And REMOVE these lines — the gateway substitutes them at egress:

    # SLACK_BOT_TOKEN=...      ← delete
    # SLACK_APP_TOKEN=...      ← delete (if present)
    # SLACK_CHANNEL_ID=...     ← KEEP if you want host AutoForge to also
    #                            be able to fall back to in-process bot
    #                            (AUTOFORGE_SLACK_IN_PROCESS=1)

Then launch the bot inside the sandbox:
    ./tasks.sh slack-bot-up

Watch it work:
    ./tasks.sh slack-bot-logs

Stop it:
    ./tasks.sh slack-bot-down

To verify the slack-only egress in front of judges, exec a curl to
something that should be blocked:
    openshell sandbox exec $SANDBOX -- curl -sS --max-time 5 https://pypi.org \\
        || echo '✓ pypi correctly blocked by policy'
EOF
