"""Smoke the FULL agentic Trainer pipeline on MNIST.

Stages exercised (all real):
  1. Oracle baseline (sklearn LogReg)
  2. Generate design.md via Nemotron
  3. HITL gate — auto-approved here by AutoApproveHITLService
  4. Generate code/model.py + code/train.py via Nemotron
  5. Smoke harness on generated code
  6. Subprocess training (real fit, real best.pkl)
  7. Build TrainingResult

Estimated runtime: ~3-5 minutes (two Nemotron calls + training).
Output artifacts land in `data/artifacts/<run_id>/`:
  design.md  oracle.json  verify_report.json  code/{model.py,train.py}
  models/{model_id}.pkl  logs/{train_stdout.log, train_stderr.log}
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
from contracts.schemas import (
    CandidateArchitecture,
    Citation,
    DatasetProfile,
    Modality,
    PreparationReport,
    StrategySpec,
    TaskType,
    TrainingEnvelope,
)
from memory.store import MemoryStore

from agents.training_agent import TrainingAgent
from hitl.approval_queue import ApprovalQueue
from hitl.auto import AutoApproveHITLService
from tools import preparation_tools as prep_tools

configure_logging()
console = Console()

MNIST_DIR = _PROJECT_ROOT / "data" / "fixtures" / "mnist"
SPLIT_DIR = _PROJECT_ROOT / "data" / "artifacts" / "smoke" / "split"


def _tee(agent) -> None:
    orig = agent.emit_event

    def tee(et, message="", payload=None):
        orig(et, message=message, payload=payload)
        if et == EventType.THINKING:
            console.print(f"  [dim]💭 {(message or '')[:120]}[/dim]")
        elif et == EventType.TOOL_CALL:
            console.print(f"  [blue]🔧 {message}[/blue]")
        elif et == EventType.INFO:
            console.print(f"  [green]ℹ {message}[/green]")
        elif et == EventType.WARNING:
            console.print(f"  [yellow]⚠ {message}[/yellow]")
        elif et == EventType.ERROR:
            console.print(f"  [red]✗ {message}[/red]")
        elif et == EventType.APPROVAL_REQUESTED:
            console.print(f"  [magenta]🔔 {message}[/magenta]")
        elif et == EventType.APPROVAL_RECEIVED:
            console.print(f"  [magenta]📩 {message}[/magenta]")

    agent.emit_event = tee


def main() -> int:
    if not MNIST_DIR.exists() or not any(MNIST_DIR.rglob("*.png")):
        console.print(f"[red]MNIST fixture missing:[/] {MNIST_DIR}")
        console.print("Run [bold]python scripts/create_mnist_fixture.py[/] first.")
        return 1

    if not (SPLIT_DIR / "train").exists():
        console.print("[dim]splitting MNIST fixture 80/20 for smoke…[/]")
        prep_tools.split_image_dir(
            source_dir=MNIST_DIR, test_size=0.2, output_dir=SPLIT_DIR,
        )

    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "smoke: agentic Trainer on MNIST", str(MNIST_DIR))

    # Fake upstream agent outputs
    profile = DatasetProfile(
        dataset_path=str(MNIST_DIR),
        modality=Modality.IMAGE,
        n_rows=500, n_cols=0,
        image_channels=1, image_formats=["png"],
        image_resolutions=[(28, 28)] * 5,
        n_classes=10,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        class_balance={str(i): 0.1 for i in range(10)},
        profile_summary="MNIST sample, 500 images × 10 classes, 28×28 grayscale.",
    )
    spec = StrategySpec(
        objective="Classify handwritten digits with accuracy >= 0.90",
        task_type=TaskType.IMAGE_CLASSIFICATION,
        success_metric="accuracy",
        success_threshold=0.90,
        candidate_architectures=[
            CandidateArchitecture(
                name="MLP-medium",
                family="neural_net",
                library="sklearn",
                hyperparameter_space={
                    "hidden_layer_sizes": [128, 64],
                    "alpha": 1e-3,
                    "learning_rate_init": 1e-3,
                },
                rationale="MLP fits sklearn pipeline; fast on CPU for 500 imgs.",
            ),
        ],
        research_summary="Smoke test — MLP baseline on MNIST.",
        citations=[Citation(title="smoke", source="test")],
    )
    envelope = TrainingEnvelope(
        gpu_available=False, gpu_name=None, gpu_memory_gb=None,
        cpu_count=4, system_memory_gb=8.0,
        max_train_minutes=1.0, max_trials=1,
        allowed_libraries=["sklearn"], notes="smoke",
    )
    prep_report = PreparationReport(
        original_dataset_path=str(MNIST_DIR),
        prepared_dataset_path=str(SPLIT_DIR),
        operations=["train_test_split_images(test_size=0.2)"],
        summary="80/20 split for smoke.",
    )

    # HITL — auto-approve the design.md gate
    queue = ApprovalQueue(store)
    hitl = AutoApproveHITLService(store)

    # Run the agentic Trainer
    console.rule("[bold cyan]Agentic Trainer (real)")
    trainer = TrainingAgent(store=store, run_id=run_id, hitl=hitl)
    _tee(trainer)
    try:
        training_result = trainer.run(
            strategy_spec=spec,
            training_envelope=envelope,
            dataset_profile=profile,
            preparation_report=prep_report,
        )
    except Exception as exc:
        console.print(f"[bold red]Trainer failed:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return 1

    # --- Summary ---
    run_dir = _PROJECT_ROOT / "data" / "artifacts" / run_id
    console.rule("[bold green]Trainer summary")
    body = [
        f"[bold]model_id[/]    {training_result.best_model_id}",
        f"[bold]score[/]       {training_result.best_score:.3f} ({training_result.metric_name})",
        f"[bold]wall[/]        {training_result.training_time_seconds:.1f}s",
        f"[bold]artifact[/]    {training_result.artifact_path}",
        f"[bold]notes[/]       {training_result.notes}",
        "",
        f"[bold]Per-run dir[/]  {run_dir}",
    ]
    for p in [
        "design.md",
        "oracle.json",
        "verify_report.json",
        "code/model.py",
        "code/train.py",
        "logs/train_stdout.log",
    ]:
        full = run_dir / p
        if full.exists():
            body.append(f"  ✓ {p}  ({full.stat().st_size:,} bytes)")
        else:
            body.append(f"  ✗ {p}  MISSING")
    console.print(Panel("\n".join(body), border_style="green"))

    # Show design.md preview
    design = run_dir / "design.md"
    if design.exists():
        console.rule("[bold cyan]design.md preview (first 1500 chars)")
        console.print(design.read_text(encoding="utf-8")[:1500])

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Crash:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
