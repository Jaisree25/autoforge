"""End-to-end smoke: Profiler → Researcher → Preparer on MNIST.

Three real Nemotron agents:
  1. Profiler reads the MNIST fixture (500 PNGs across 10 classes), returns
     a DatasetProfile (modality=image, task=image_classification).
  2. Researcher reads the profile + objective, searches Tavily + arXiv, picks
     a CNN architecture, writes research.md.
  3. Preparer reads the profile + StrategySpec, plans prep operations
     (split, normalize, augment), executes them.

Total runtime: ~3-5 minutes (three real LLM agents). Run this once after
the API keys are set and the MNIST fixture exists.
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
from memory.store import MemoryStore

from agents.dataset_agent import DatasetAgent
from agents.profiler_agent import ProfilerAgent
from agents.strategy_agent import StrategyAgent

configure_logging()
console = Console()

MNIST_DIR = _PROJECT_ROOT / "data" / "fixtures" / "mnist"


def _tee(agent) -> None:
    orig = agent.emit_event

    def tee_emit(event_type, message="", payload=None):
        orig(event_type, message=message, payload=payload)
        if event_type == EventType.THINKING:
            preview = (message or "")[:120]
            console.print(f"  [dim]💭 {preview}{'…' if len(message or '') > 120 else ''}[/dim]")
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
    if not MNIST_DIR.exists() or not any(MNIST_DIR.rglob("*.png")):
        console.print(f"[red]MNIST fixture missing:[/] {MNIST_DIR}")
        console.print("Run [bold]python scripts/create_mnist_fixture.py[/] first.")
        return 1

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    objective = "Classify handwritten digits (MNIST) with accuracy >= 0.95"
    store.create_run(run_id, objective, str(MNIST_DIR))

    # ===== 1. Profiler =====
    console.rule("[bold cyan]1. Profiler (image)")
    profiler = ProfilerAgent(store=store, run_id=run_id)
    _tee(profiler)
    profile = profiler.run(dataset_path=str(MNIST_DIR), objective=objective)

    # ===== 2. Researcher =====
    console.rule("[bold cyan]2. Researcher")
    researcher = StrategyAgent(store=store, run_id=run_id)
    _tee(researcher)
    spec = researcher.run(objective=objective, dataset_profile=profile)

    # ===== 3. Preparer =====
    console.rule("[bold cyan]3. Preparer")
    preparer = DatasetAgent(store=store, run_id=run_id)
    _tee(preparer)
    prep_report = preparer.run(dataset_profile=profile, strategy_spec=spec)

    # ===== summary =====
    console.rule("[bold green]PreparationReport")
    body = [
        f"[bold]original[/]  {prep_report.original_dataset_path}",
        f"[bold]prepared[/]  {prep_report.prepared_dataset_path or '(none — config-only)'}",
        f"[bold]operations[/]  {len(prep_report.operations)}",
    ]
    for op in prep_report.operations:
        body.append(f"  • {op[:90]}")
    body.append(f"[bold]summary[/]")
    body.append(prep_report.summary)
    if prep_report.notes:
        body.append(f"[bold]notes[/]")
        body.append(prep_report.notes)
    console.print(Panel("\n".join(body), border_style="green"))

    # ===== Show prepared directory tree if it exists =====
    if prep_report.prepared_dataset_path:
        prep_path = Path(prep_report.prepared_dataset_path)
        if prep_path.is_dir():
            children = sorted(prep_path.iterdir())[:6]
            console.print(f"\n[bold]Prepared dataset tree (first 6):[/]")
            for c in children:
                if c.is_dir():
                    n = len(list(c.rglob("*.png")))
                    console.print(f"  📁 {c.name}/  ({n} files)")
                else:
                    console.print(f"  📄 {c.name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Crash:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
