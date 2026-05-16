"""Ping the NVIDIA Nemotron NIM endpoint for both configured models.

Confirms:
  - `NVIDIA_API_KEY` and base URL are wired up correctly.
  - The two model IDs pinned in `config.py` actually exist on the endpoint.
  - Round-trip latency + token usage look sane.

    python scripts/test_nemotron.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    COORDINATOR_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    WORKER_MODEL,
    configure_logging,
)

configure_logging()
console = Console()

_PROMPT = (
    "In one sentence, explain what gradient boosting is to someone who "
    "already knows linear regression."
)


def _ping(client: OpenAI, model: str) -> bool:
    """Single chat-completion roundtrip. Returns True on success."""
    console.rule(f"[bold cyan]{model}[/]")
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            # Nemotron models default to "thinking" mode where most tokens go
            # into a reasoning trace that doesn't surface in `message.content`.
            # `/no_think` disables that — gives a fast, visible answer for
            # smoke purposes. Drop it (or switch to `/think`) when you actually
            # want reasoning.
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": _PROMPT},
            ],
            max_tokens=400,
            temperature=0.3,
        )
    except OpenAIError as exc:
        console.print(f"[red]API error:[/] {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Unexpected error:[/] {type(exc).__name__}: {exc}")
        return False

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    content = (resp.choices[0].message.content or "").strip() or "(empty response)"
    console.print(Panel(content, border_style="dim", title="response"))

    usage = resp.usage
    table = Table(show_header=False, box=None, pad_edge=False)
    if usage is not None:
        table.add_row("prompt tokens",     str(usage.prompt_tokens))
        table.add_row("completion tokens", str(usage.completion_tokens))
        table.add_row("total tokens",      str(usage.total_tokens))
    table.add_row("latency", f"{elapsed_ms:.0f} ms")
    console.print(table)
    return True


def main() -> int:
    if not NVIDIA_API_KEY:
        console.print(
            "[red]NVIDIA_API_KEY is not set.[/]\n"
            "Copy [cyan].env.example[/] to [cyan].env[/] and fill in your "
            "key from build.nvidia.com."
        )
        return 1

    console.print(Panel.fit(
        f"[bold]endpoint[/]    {NVIDIA_BASE_URL}\n"
        f"[bold]coordinator[/] {COORDINATOR_MODEL}\n"
        f"[bold]worker[/]      {WORKER_MODEL}",
        title="Nemotron smoke",
        border_style="cyan",
    ))

    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

    results = {
        COORDINATOR_MODEL: _ping(client, COORDINATOR_MODEL),
        WORKER_MODEL:      _ping(client, WORKER_MODEL),
    }

    console.rule()
    failed = [m for m, ok in results.items() if not ok]
    if failed:
        console.print(f"[red]FAIL[/] · {len(failed)} of {len(results)} models unreachable: {failed}")
        return 1
    console.print(f"[green]OK[/] · {len(results)} of {len(results)} models reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
