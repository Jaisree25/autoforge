"""Smoke test the real Researcher end-to-end.

Runs Profiler then Researcher against the churn fixture. Prints each
thinking paragraph + tool call + result summary as they happen. Final
output: the structured `StrategySpec` and the path to the written
`research.md`.

Expected runtime: ~60-90s (Researcher does multiple Nemotron calls + tool
dispatches against Tavily + arXiv).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from config import AUTOFORGE_DB_PATH, configure_logging
from contracts.messages import EventType
from contracts.schemas import AgentName
from memory.store import MemoryStore

from agents.profiler_agent import ProfilerAgent
from agents.strategy_agent import StrategyAgent

configure_logging()
console = Console()

CSV_PATH = _PROJECT_ROOT / "data" / "fixtures" / "churn_sample.csv"


def _tee(agent) -> None:
    """Mirror the agent's emit_event into the terminal too."""
    orig_emit = agent.emit_event

    def tee_emit(event_type, message="", payload=None):
        orig_emit(event_type, message=message, payload=payload)
        if event_type == EventType.THINKING:
            preview = (message or "")[:120]
            console.print(
                f"  [dim]💭 {preview}{'…' if len(message or '') > 120 else ''}[/dim]"
            )
        elif event_type == EventType.TOOL_CALL:
            console.print(f"  [blue]🔧 {message}[/blue]")
        elif event_type == EventType.INFO:
            console.print(f"  [green]ℹ {message}[/green]")
        elif event_type == EventType.WARNING:
            console.print(f"  [yellow]⚠ {message}[/yellow]")
        elif event_type == EventType.ERROR:
            console.print(f"  [red]✗ {message}[/red]")

    agent.emit_event = tee_emit


def main() -> int:
    if not CSV_PATH.exists():
        console.print(f"[red]Missing fixture[/]: {CSV_PATH}")
        console.print("Run: [bold]python scripts/create_fixtures.py[/]")
        return 1

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    objective = "Predict customer churn with F1 >= 0.85"
    store.create_run(run_id, objective, str(CSV_PATH))

    # ---------- Stage 1: Profiler (real) ----------
    console.rule("[bold cyan]1. Profiler")
    profiler = ProfilerAgent(store=store, run_id=run_id)
    _tee(profiler)
    try:
        profile = profiler.run(dataset_path=str(CSV_PATH), objective=objective)
    except Exception as exc:
        console.print(f"[red]Profiler failed:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return 1

    # ---------- Stage 2: Researcher (real) ----------
    console.rule("[bold cyan]2. Researcher")
    researcher = StrategyAgent(store=store, run_id=run_id)
    _tee(researcher)
    try:
        spec = researcher.run(objective=objective, dataset_profile=profile)
    except Exception as exc:
        console.print(f"[red]Researcher failed:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return 1

    # ---------- Show structured output ----------
    console.rule("[bold green]StrategySpec")
    body = [
        f"[bold]objective[/]         {spec.objective}",
        f"[bold]task_type[/]         {spec.task_type.value}",
        f"[bold]success_metric[/]    {spec.success_metric}",
        f"[bold]success_threshold[/] {spec.success_threshold:.3f}",
        f"[bold]architectures[/]     {len(spec.candidate_architectures)}",
    ]
    for i, a in enumerate(spec.candidate_architectures, 1):
        body.append(
            f"  {i}. `{a.name}` ({a.family} / {a.library})  "
            f"— {len(a.hyperparameter_space)} HP dims"
        )
    body.append(f"[bold]citations[/]         {len(spec.citations)}")
    for c in spec.citations[:5]:
        body.append(f"  • {c.title[:60]}")
    console.print(Panel("\n".join(body), border_style="green"))

    # ---------- Show research.md preview ----------
    md_path = _PROJECT_ROOT / "data" / "artifacts" / run_id / "research.md"
    if md_path.exists():
        console.rule("[bold cyan]research.md (first 1500 chars)")
        text = md_path.read_text(encoding="utf-8")
        console.print(text[:1500] + ("\n…\n" if len(text) > 1500 else ""))
        console.print(f"\n[dim]Full file: {md_path}[/]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Crash:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
