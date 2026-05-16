"""Run a full AutoForge pipeline end-to-end.

By default uses `build_hitl_service(store)`, which wires Telegram automatically
if `TELEGRAM_BOT_TOKEN`+`TELEGRAM_CHAT_ID` are set and falls back to
dashboard-only otherwise. The pipeline will BLOCK at the Strategy approval
gate until someone approves via the dashboard or Telegram — so the typical
flow is:

    Terminal A:  .\tasks.ps1 dashboard      (or `streamlit run dashboard/app.py`)
    Terminal B:  python scripts/run_pipeline.py --dataset ... --objective ...

For dev / CI / quick tries, pass `--auto-approve` to skip the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console
from rich.panel import Panel

from config import AUTOFORGE_DB_PATH, configure_logging
from contracts.schemas import PipelineStatus
from memory.store import MemoryStore

from agents.coordinator import Coordinator, PipelineRejected
from hitl.auto import AutoApproveHITLService
from hitl.coordinator_service import build_hitl_service

configure_logging()
console = Console()

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the input dataset (CSV or image dir).",
    ),
    objective: str = typer.Option(
        ...,
        "--objective",
        "-o",
        help="Plain-English objective, e.g. 'predict churn with F1 >= 0.85'.",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "-y",
        help="Auto-approve all HITL gates. Useful for CI / quick dev runs.",
    ),
    db_path: Path = typer.Option(
        AUTOFORGE_DB_PATH,
        "--db",
        help="Override the SQLite DB path (defaults to AUTOFORGE_DB_PATH).",
    ),
) -> None:
    """Run a full pipeline against the configured Nemotron + HITL stack."""
    store = MemoryStore(db_path=db_path)
    # Be friendly — init_schema is idempotent; saves users one command.
    store.init_schema()

    if auto_approve:
        # Even in auto-approve, reuse the real Slack bot so the per-agent
        # Slack feed is still exercised. The auto service just resolves
        # the approval immediately after broadcasting.
        real_service = build_hitl_service(store)
        hitl = AutoApproveHITLService(store, slack=real_service.slack)
        hitl_mode = (
            "auto-approve + slack" if real_service.slack else "auto-approve"
        )
    else:
        hitl = build_hitl_service(store)
        if getattr(hitl, "slack", None) is not None:
            hitl_mode = "dashboard + slack"
        else:
            hitl_mode = "dashboard only"

    coord = Coordinator(store=store, hitl=hitl)

    console.print(Panel.fit(
        f"[bold]run_id[/]    [cyan]{coord.run_id}[/]\n"
        f"[bold]dataset[/]   {dataset}\n"
        f"[bold]objective[/] {objective}\n"
        f"[bold]HITL[/]      {hitl_mode}\n"
        f"[bold]db[/]        {db_path}",
        title="AutoForge run starting",
        border_style="cyan",
    ))

    try:
        run = coord.execute(dataset_path=dataset, objective=objective)
    except PipelineRejected as exc:
        console.print(f"[yellow]CANCELLED:[/] {exc}")
        raise typer.Exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAILED:[/] {type(exc).__name__}: {exc}")
        raise typer.Exit(1)

    br = run.benchmark_report
    status_color = "green" if run.status is PipelineStatus.COMPLETED else "yellow"
    summary_lines = [
        f"[bold]status[/]      {run.status.value}",
    ]
    if br is not None:
        summary_lines.extend([
            f"[bold]metric[/]      {br.accuracy_metric} = {br.accuracy_value:.3f}",
            f"[bold]passed[/]      {br.passed_threshold}",
            f"[bold]latency p50[/] {br.latency.p50_ms:.1f} ms",
            f"[bold]throughput[/]  {br.throughput_qps:.0f} QPS",
        ])
    if run.deployment_artifact is not None:
        summary_lines.append(
            f"[bold]artifact[/]    {run.deployment_artifact.artifact_path}"
        )

    console.print(Panel.fit(
        "\n".join(summary_lines),
        title="Run complete",
        border_style=status_color,
    ))


if __name__ == "__main__":
    app()
