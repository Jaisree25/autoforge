"""Smoke test for hitl/slack_bot_runner.py with a mocked Slack WebClient.

Verifies the three loops (reply-poll, unposted-approval, outbox-drain) work
end-to-end against a real SQLite DB without making any actual Slack calls.

Run from the autoforge env:
    python scripts/_smoke_slack_bot_runner.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import threading
import time
import uuid

# Set fake Slack env BEFORE importing autoforge modules so config picks them up.
TMP_DB = pathlib.Path(tempfile.mkstemp(suffix="_smoke.db")[1])
os.environ.update(
    AUTOFORGE_DB_PATH=str(TMP_DB),
    SLACK_BOT_TOKEN="xoxb-FAKE-FOR-SMOKE",
    SLACK_CHANNEL_ID="C_MAIN",
    SLACK_CHANNEL_PROFILER="C_PROFILER",
    AUTOFORGE_SLACK_BRIDGE_POLL_SEC="0.2",
)

sys.path.insert(0, "/home/ubuntu/autoforge")

# Monkey-patch slack_sdk.WebClient before slack_bot.py imports it.
import slack_sdk  # noqa: E402


class FakeWebClient:
    """Records calls; returns what _poll_loop / send / notify expect."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.posted_messages: list[dict] = []   # (channel, text)
        self.history_responses: list[list[dict]] = []  # queued responses
        self._lock = threading.Lock()

    def chat_postMessage(self, *, channel: str, text: str, **_: object) -> dict:
        with self._lock:
            self.posted_messages.append({"channel": channel, "text": text})
        return {"ok": True, "ts": f"{time.time():.6f}"}

    def conversations_history(self, *, channel: str, oldest: str, limit: int) -> dict:
        with self._lock:
            msgs = self.history_responses.pop(0) if self.history_responses else []
        return {"ok": True, "messages": msgs}

    def queue_reply(self, text: str, user: str = "U_FAKE") -> None:
        """Inject a fake user reply for the next conversations_history call."""
        with self._lock:
            self.history_responses.append([{
                "text": text,
                "user": user,
                "ts": f"{time.time():.6f}",
            }])


# Replace WebClient symbol on the slack_sdk module BEFORE slack_bot imports it.
fake_client_holder: list[FakeWebClient] = []


def _fake_factory(token: str) -> FakeWebClient:
    c = FakeWebClient(token)
    fake_client_holder.append(c)
    return c


slack_sdk.WebClient = _fake_factory  # type: ignore[assignment]

# Now import the rest (slack_bot.py will pick up the patched WebClient).
from contracts.messages import (  # noqa: E402
    ApprovalDecision, ApprovalRequest, ApprovalStatus,
)
from contracts.schemas import AgentName  # noqa: E402
from memory.store import MemoryStore  # noqa: E402
from hitl import slack_bot_runner  # noqa: E402


def _seed_and_verify(store: MemoryStore, approval_id: str) -> None:
    """Worker thread: wait for pollers to drain, assert, then send SIGINT.

    Main thread runs `slack_bot_runner.main()` (signal handlers must be
    registered from main thread). This worker drives the assertions.
    """
    time.sleep(1.5)  # 3 cycles at 0.2s

    assert fake_client_holder, "FakeWebClient was never constructed"
    client = fake_client_holder[0]
    posted = client.posted_messages
    print(f"chat_postMessage called {len(posted)} time(s):")
    for p in posted:
        print(f"  → channel={p['channel']!r} text={p['text'][:80]!r}")

    approval_posts = [p for p in posted if "Approve smoke plan" in p["text"]]
    assert approval_posts, "Approval was not posted to Slack"
    assert approval_posts[0]["channel"] == "C_MAIN", \
        f"Approval should hit main channel, got {approval_posts[0]['channel']}"
    print("[1/4] approval posted to main channel ✓")

    notif_posts = [p for p in posted if "profiler completed" in p["text"]]
    assert notif_posts, "Outbox notification was not posted"
    assert notif_posts[0]["channel"] == "C_PROFILER", \
        f"Profiler notification should hit C_PROFILER, got {notif_posts[0]['channel']}"
    print("[2/4] outbox notification routed to per-agent channel ✓")

    assert store.list_unposted_pending_approvals() == []
    assert store.list_unsent_notifications() == []
    print("[3/4] slack_posted + slack_outbox.sent_at marks set ✓")

    client.queue_reply("CONFIRM", user="U_REVIEWER")
    deadline = time.time() + 5.0
    resolved = None
    while time.time() < deadline:
        req = store.get_approval_request(approval_id)
        if req and req.status is ApprovalStatus.RESOLVED:
            resolved = req
            break
        time.sleep(0.1)
    assert resolved is not None, "CONFIRM reply did not resolve the approval"
    assert resolved.decision is ApprovalDecision.APPROVED
    assert resolved.responder == "slack:U_REVIEWER"
    print(f"[4/4] CONFIRM reply parsed → resolved by {resolved.responder} ✓")

    print("\nALL CHECKS PASS: slack_bot_runner logic verified without real Slack")

    import signal as _sig
    os.kill(os.getpid(), _sig.SIGINT)


def main() -> int:
    store = MemoryStore(db_path=TMP_DB)
    store.init_schema()
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "smoke", "smoke.csv")

    approval_id = str(uuid.uuid4())
    store.create_approval_request(ApprovalRequest(
        request_id=approval_id, run_id=run_id, agent=AgentName.TRAINING,
        title="Approve smoke plan", description="ok?",
        payload={"kind": "plan_approval"}, status=ApprovalStatus.PENDING,
    ))
    store.enqueue_slack_notification(
        run_id, "profiler completed", agent=AgentName.PROFILER,
    )
    print(f"Seeded run={run_id[:8]} approval={approval_id[:8]}")

    # Verification runs in a worker; main thread blocks in runner.main()
    # so signal.signal() registers cleanly.
    worker = threading.Thread(
        target=_seed_and_verify, args=(store, approval_id), daemon=True,
    )
    worker.start()

    rc = slack_bot_runner.main()  # blocks; worker SIGINTs us when done
    worker.join(timeout=2.0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
