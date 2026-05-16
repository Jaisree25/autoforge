"""Initialize the SQLite store and confirm the JSON-schema dump landed.

Idempotent — re-running this is safe; every DDL statement is `IF NOT EXISTS`
and the JSON schemas are overwritten on each call.

    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# CLI scripts and the dashboard add the project root themselves because
# Python doesn't add the script's parent dir's parent automatically.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console

from config import ARTIFACTS_DIR, AUTOFORGE_DB_PATH, configure_logging
from memory.store import OUTPUT_KIND_TO_MODEL, MemoryStore

configure_logging()
console = Console()


def main() -> int:
    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    console.print(
        f"[green]SQLite store ready[/]  ·  [cyan]{AUTOFORGE_DB_PATH}[/]"
    )

    contracts_dir = ARTIFACTS_DIR / "contracts"
    expected = [f"{kind}.schema.json" for kind in OUTPUT_KIND_TO_MODEL]
    missing = [name for name in expected if not (contracts_dir / name).exists()]

    if missing:
        console.print(
            f"[red]JSON schema files missing under {contracts_dir}: {missing}[/]"
        )
        return 1

    console.print(
        f"[green]JSON schemas dumped[/]  ·  [cyan]{contracts_dir}[/]"
    )
    for name in expected:
        console.print(f"  • {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
