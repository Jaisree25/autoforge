"""Smoke test for agents/_llm_client.py.

Exercises both methods of NemotronClient:
  1. think_and_answer — free-form text answer with streamed reasoning
  2. think_and_answer_structured — JSON answer validated against Pydantic

Prints each thinking paragraph as it arrives so you can see the streaming
working. Run this once after .env is configured to confirm everything is
wired up before plugging the client into real agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from config import WORKER_MODEL, configure_logging
from agents._llm_client import NemotronClient

configure_logging()
console = Console()


# -- Free-form -------------------------------------------------------------
def test_free_form() -> None:
    console.rule("[bold cyan]1. think_and_answer (free-form)")
    client = NemotronClient(model=WORKER_MODEL)

    paragraphs: list[str] = []
    def on_thinking(p: str) -> None:
        paragraphs.append(p)
        console.print(f"  [dim]💭 {p[:100]}{'…' if len(p) > 100 else ''}[/dim]")

    answer = client.think_and_answer(
        system="You are a friendly tutor.",
        user="In one sentence, what's the difference between bagging and boosting?",
        on_thinking=on_thinking,
        max_tokens=600,
    )
    console.print()
    console.print(Panel(answer or "(empty)", border_style="dim", title="answer"))
    console.print(f"[green]→ {len(paragraphs)} thinking paragraph(s) received[/]")


# -- Structured ------------------------------------------------------------
class TabularRecommendation(BaseModel):
    library: str = Field(description="ML library to use (e.g. xgboost, sklearn).")
    reason: str = Field(description="One-sentence rationale.")
    confidence: float = Field(ge=0.0, le=1.0)


def test_structured() -> None:
    console.rule("[bold cyan]2. think_and_answer_structured (JSON → Pydantic)")
    client = NemotronClient(model=WORKER_MODEL)

    paragraphs: list[str] = []
    def on_thinking(p: str) -> None:
        paragraphs.append(p)
        console.print(f"  [dim]💭 {p[:100]}{'…' if len(p) > 100 else ''}[/dim]")

    result = client.think_and_answer_structured(
        system="You recommend ML libraries for tabular classification problems.",
        user=(
            "Dataset: 10,000 rows × 12 columns, binary classification, "
            "moderately imbalanced (73/27). Target: predict customer churn. "
            "Recommend one library."
        ),
        schema=TabularRecommendation,
        on_thinking=on_thinking,
        max_tokens=6000,
    )
    console.print()
    console.print(Panel(
        f"library: [cyan]{result.library}[/]\n"
        f"reason:  {result.reason}\n"
        f"confidence: {result.confidence:.2f}",
        border_style="dim", title="structured answer",
    ))
    console.print(f"[green]→ {len(paragraphs)} thinking paragraph(s) received[/]")


if __name__ == "__main__":
    try:
        test_free_form()
        console.print()
        test_structured()
        console.rule()
        console.print("[bold green]Client smoke OK[/]")
    except Exception as exc:
        console.print(f"[bold red]FAIL:[/] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
