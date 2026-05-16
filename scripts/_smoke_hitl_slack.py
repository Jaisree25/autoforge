"""Smoke the full HITL chain: HITLCoordinatorService + Slack + dashboard approval.

What it does:
  1. Build `HITLCoordinatorService` via `build_hitl_service(store)` — wires
     Slack automatically if `SLACK_BOT_TOKEN`+`SLACK_CHANNEL_ID` are set.
  2. Fire `service.notify(...)` — a plain Slack message ("test notification").
  3. Spawn a background thread that, after 3 seconds, simulates a dashboard
     approval by calling `store.respond_to_approval(...)`.
  4. Call `service.request_and_wait(...)` on the main thread — this also pushes
     to Slack, then blocks waiting for the response.
  5. Print the resolved response so you can confirm everything wired up.

Will send 2 messages to the configured Slack channel — keep that in mind.
Run only when you're ready to test end-to-end.
"""
from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from config import (
    AUTOFORGE_DB_PATH,
    SLACK_BOT_TOKEN,
    SLACK_CHANNEL_ID,
    configure_logging,
)
from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
)
from contracts.schemas import AgentName
from memory.store import MemoryStore

from hitl.coordinator_service import build_hitl_service

configure_logging()
console = Console()


def main() -> int:
    console.rule("[bold cyan]HITL + Slack smoke")
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        console.print("[yellow]Warning:[/] SLACK_BOT_TOKEN or SLACK_CHANNEL_ID "
                      "not set — service will run dashboard-only.")
    else:
        console.print(f"  Slack channel: [cyan]{SLACK_CHANNEL_ID}[/]")

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "HITL+Slack smoke", "(none)")

    service = build_hitl_service(store)
    console.print(f"  service: [green]{type(service).__name__}[/]")
    console.print(f"  slack:   {'[green]wired[/]' if service.slack else '[yellow]disabled[/]'}")
    console.print()

    # --- 1. Notify ---
    console.rule("[cyan]Step 1: service.notify()")
    service.notify(run_id, "AutoForge HITL smoke test — notification path")
    console.print("[green]✓[/] notify() returned (check Slack channel for ':mega:' message)")
    console.print()

    # --- 2. Request + auto-approve via background simulator ---
    console.rule("[cyan]Step 2: request_and_wait()")
    request = ApprovalRequest(
        run_id=run_id,
        agent=AgentName.TRAINING,
        title="Smoke test — approve me",
        description=(
            "This is a test approval request from the HITL smoke script. "
            "It will be auto-approved by a simulator thread after 3 seconds. "
            "If you see this in Slack, the request path works."
        ),
        payload={
            "summary": "Smoke test of HITLCoordinatorService.request_and_wait",
            "next_agent": "(smoke complete)",
        },
    )

    def dashboard_simulator() -> None:
        time.sleep(3.0)
        console.print("[dim]  [simulator] approving via store.respond_to_approval()[/]")
        store.respond_to_approval(ApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVED,
            responder="dashboard-simulator",
            comment="auto-approved by HITL smoke script",
        ))

    t = threading.Thread(target=dashboard_simulator, daemon=True)
    t.start()

    console.print("  Sending approval request (Slack message + DB persist)...")
    response = service.request_and_wait(request, timeout=15.0)
    console.print("[green]✓[/] request_and_wait returned")
    console.print()

    # --- 3. Final notify ---
    service.notify(run_id, f"AutoForge smoke complete — response: {response.decision.value}")

    # --- Summary ---
    console.rule("[bold green]Summary")
    console.print(Panel(
        f"[bold]run_id[/]      {run_id}\n"
        f"[bold]decision[/]    {response.decision.value}\n"
        f"[bold]responder[/]   {response.responder}\n"
        f"[bold]comment[/]     {response.comment}\n"
        f"[bold]request_id[/]  {response.request_id}",
        border_style="green",
    ))
    console.print()
    console.print("If you saw 2 Slack messages in your channel, the HITL chain is healthy.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Crash:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
