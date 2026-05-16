"""Smoke test Tavily + arXiv.

Confirms both research tools work end-to-end before we wire them into the
Researcher agent. Loads .env via config.py so TAVILY_API_KEY is picked up
from the project's standard place.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from config import TAVILY_API_KEY, configure_logging

configure_logging()
console = Console()


def smoke_tavily() -> None:
    console.rule("[bold cyan]Tavily")
    if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("tvly-...") or TAVILY_API_KEY == "":
        console.print(
            "[yellow]TAVILY_API_KEY not set (or still a placeholder).[/]\n"
            "Edit .env and replace `tvly-...` with your real key."
        )
        return
    try:
        from tavily import TavilyClient
    except ImportError:
        console.print("[red]tavily-python not installed.[/] Run: pip install tavily-python")
        return

    client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        resp = client.search(
            "xgboost tabular customer churn 2025",
            max_results=3,
            search_depth="basic",
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Tavily API error:[/] {type(exc).__name__}: {exc}")
        return

    results = resp.get("results", [])
    console.print(f"[green]OK[/] · {len(results)} result(s)")
    for r in results[:3]:
        console.print(Panel(
            f"[bold]{r.get('title', '(no title)')}[/]\n"
            f"[dim]{r.get('url', '')}[/]\n\n"
            f"{r.get('content', '')[:300]}...",
            border_style="dim",
        ))


def smoke_arxiv() -> None:
    console.rule("[bold cyan]arXiv")
    try:
        import arxiv
    except ImportError:
        console.print("[red]arxiv not installed.[/] Run: pip install arxiv")
        return

    search = arxiv.Search(
        query="imbalanced binary classification calibration tabular",
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = list(arxiv.Client().results(search))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]arXiv error:[/] {type(exc).__name__}: {exc}")
        return

    console.print(f"[green]OK[/] · {len(results)} result(s)")
    for r in results[:3]:
        console.print(Panel(
            f"[bold]{r.title}[/]\n"
            f"[dim]{r.entry_id}[/]\n"
            f"[dim]{', '.join(a.name for a in r.authors[:3])}{'...' if len(r.authors) > 3 else ''}[/]\n\n"
            f"{r.summary[:300]}...",
            border_style="dim",
        ))


if __name__ == "__main__":
    smoke_tavily()
    console.print()
    smoke_arxiv()
    console.rule()
    console.print("[bold green]Tools smoke done[/]")
