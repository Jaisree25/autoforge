"""Smoke test the real Profiler against both fixture modalities.

Runs Profiler.run() on:
  - data/fixtures/churn_sample.csv         (tabular)
  - data/fixtures/sample_images            (image directory)

Prints each thinking paragraph as it streams + the final `DatasetProfile`
fields so you can confirm the LLM filled judgment fields correctly.
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

configure_logging()
console = Console()

CSV_PATH = _PROJECT_ROOT / "data" / "fixtures" / "churn_sample.csv"
IMG_DIR = _PROJECT_ROOT / "data" / "fixtures" / "sample_images"


def smoke(path: Path, label: str, objective: str) -> None:
    console.rule(f"[bold cyan]{label}: {path.relative_to(_PROJECT_ROOT)}")
    if not path.exists():
        console.print(
            f"[red]Missing fixture[/]: {path}. "
            "Run [bold]python scripts/create_fixtures.py[/] first."
        )
        return

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    store.create_run(run_id, objective, str(path))

    # Tap into the event stream so we see thinking as it arrives.
    agent = ProfilerAgent(store=store, run_id=run_id)
    orig_emit = agent.emit_event

    def tee_emit(event_type, message="", payload=None):  # type: ignore[no-untyped-def]
        orig_emit(event_type, message=message, payload=payload)
        if event_type == EventType.THINKING:
            preview = (message or "")[:100]
            console.print(
                f"  [dim]💭 {preview}{'…' if len(message or '') > 100 else ''}[/dim]"
            )
        elif event_type == EventType.TOOL_CALL:
            console.print(f"  [blue]🔧 {message}[/blue]")
        elif event_type == EventType.INFO:
            console.print(f"  [green]ℹ {message}[/green]")
        elif event_type == EventType.WARNING:
            console.print(f"  [yellow]⚠ {message}[/yellow]")

    agent.emit_event = tee_emit  # type: ignore[method-assign]

    try:
        profile = agent.run(dataset_path=str(path), objective=objective)
    except Exception as exc:
        console.print(f"[bold red]FAIL[/]: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return

    # Render the final profile
    lines = [
        f"[bold]modality[/]   {profile.modality.value}",
        f"[bold]task[/]       {profile.task_type.value}",
        f"[bold]n_rows[/]     {profile.n_rows}",
    ]
    if profile.modality.value == "tabular":
        lines += [
            f"[bold]n_cols[/]     {profile.n_cols}",
            f"[bold]target[/]     {profile.target_column!r}",
        ]
    else:
        lines += [
            f"[bold]n_classes[/]  {profile.n_classes}",
            f"[bold]channels[/]   {profile.image_channels}",
            f"[bold]formats[/]    {profile.image_formats}",
            f"[bold]resolutions[/]  {profile.image_resolutions[:5]}…",
        ]
    if profile.class_balance:
        lines.append(f"[bold]class_balance[/]  {profile.class_balance}")
    if profile.warnings:
        lines.append("[bold]warnings[/]")
        for w in profile.warnings:
            lines.append(f"  • {w}")
    lines.append(f"[bold]summary[/]")
    lines.append(profile.profile_summary)
    console.print(Panel("\n".join(lines), border_style="green", title="DatasetProfile"))


if __name__ == "__main__":
    try:
        smoke(CSV_PATH, "1. CSV (tabular)",
              objective="Predict customer churn with F1 >= 0.85")
        console.print()
        smoke(IMG_DIR, "2. Image directory",
              objective="Classify images of cats vs dogs with >90% accuracy")
        console.rule()
        console.print("[bold green]Profiler smoke OK[/]")
    except Exception as exc:
        console.print(f"[bold red]FAIL:[/] {type(exc).__name__}: {exc}")
        sys.exit(1)
