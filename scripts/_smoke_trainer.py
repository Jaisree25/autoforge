"""Quick smoke for the real Trainer + Evaluator + Optimizer.

Bypasses the slow Profiler/Researcher/Preparer chain by synthesizing
realistic-enough inputs in place. Lets us verify the three new agents work
end-to-end on MNIST in <60s without burning Nemotron tokens.
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
    AgentName,
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

from agents.benchmark_agent import BenchmarkAgent
from agents.dataset_agent import DatasetAgent  # only used if we need to split fresh
from agents.hardware_agent import HardwareAgent
from agents.training_agent import TrainingAgent
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
            console.print(f"  [dim]💭 {(message or '')[:100]}[/dim]")
        elif et == EventType.TOOL_CALL:
            console.print(f"  [blue]🔧 {message}[/blue]")
        elif et == EventType.INFO:
            console.print(f"  [green]ℹ {message}[/green]")
        elif et == EventType.WARNING:
            console.print(f"  [yellow]⚠ {message}[/yellow]")
        elif et == EventType.ERROR:
            console.print(f"  [red]✗ {message}[/red]")

    agent.emit_event = tee


def main() -> int:
    if not MNIST_DIR.exists() or not any(MNIST_DIR.rglob("*.png")):
        console.print(f"[red]MNIST fixture missing:[/] {MNIST_DIR}")
        console.print("Run [bold]python scripts/create_mnist_fixture.py[/] first.")
        return 1

    # 1. Split MNIST into train/test on disk (what Preparer would do).
    if not (SPLIT_DIR / "train").exists():
        console.print("[dim]splitting MNIST fixture 80/20 for smoke…[/]")
        prep_tools.split_image_dir(
            source_dir=MNIST_DIR, test_size=0.2, output_dir=SPLIT_DIR,
        )

    # 2. Fake the upstream outputs the Coordinator would have produced.
    store = MemoryStore(db_path=AUTOFORGE_DB_PATH)
    store.init_schema()
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "smoke: train+eval+optimize on MNIST", str(MNIST_DIR))

    profile = DatasetProfile(
        dataset_path=str(MNIST_DIR),
        modality=Modality.IMAGE,
        n_rows=500,
        n_cols=0,
        image_channels=1,
        image_formats=["png"],
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
                name="MLP-small",
                family="neural_net",
                library="sklearn",
                hyperparameter_space={
                    "hidden_layer_sizes": [[128], [128, 64]],
                    "alpha": [1e-4, 1e-3],
                    "learning_rate_init": [1e-3, 3e-3],
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
        max_train_minutes=2.0, max_trials=4,  # small for fast smoke
        allowed_libraries=["sklearn"], notes="smoke",
    )
    prep_report = PreparationReport(
        original_dataset_path=str(MNIST_DIR),
        prepared_dataset_path=str(SPLIT_DIR),
        operations=["train_test_split_images(test_size=0.2)"],
        summary="80/20 split for smoke.",
    )

    # 3. Run Trainer
    console.rule("[bold cyan]Trainer (real)")
    trainer = TrainingAgent(store=store, run_id=run_id)
    _tee(trainer)
    training_result = trainer.run(
        strategy_spec=spec,
        training_envelope=envelope,
        dataset_profile=profile,
        preparation_report=prep_report,
    )

    # 4. Run Evaluator
    console.rule("[bold cyan]Evaluator (real)")
    evaluator = BenchmarkAgent(store=store, run_id=run_id)
    _tee(evaluator)
    benchmark = evaluator.run(
        training_result=training_result,
        strategy_spec=spec,
        dataset_profile=profile,
        preparation_report=prep_report,
    )

    # 5. Run Optimizer
    console.rule("[bold cyan]Optimizer (real)")
    optimizer = HardwareAgent(store=store, run_id=run_id)
    _tee(optimizer)
    artifact = optimizer.run_post_training(training_result=training_result)

    # --- Summary ---
    console.rule("[bold green]Summary")
    body = [
        f"[bold]Trainer[/]",
        f"  model_id:  {training_result.best_model_id}",
        f"  best:      {training_result.best_score:.3f} ({training_result.metric_name})",
        f"  trials:    {training_result.trials_completed}/{training_result.total_trials}",
        f"  wall:      {training_result.training_time_seconds:.1f}s",
        f"  artifact:  {training_result.artifact_path}",
        "",
        f"[bold]Evaluator[/]",
        f"  metric:    {benchmark.accuracy_metric} = {benchmark.accuracy_value:.3f}",
        f"  passed:    {benchmark.passed_threshold}  (threshold {spec.success_threshold:.2f})",
        f"  latency:   p50={benchmark.latency.p50_ms:.2f}ms · p95={benchmark.latency.p95_ms:.2f}ms",
        f"  throughput: {benchmark.throughput_qps:.0f} QPS",
        "",
        f"[bold]Optimizer[/]",
        f"  format:    {artifact.format}",
        f"  size:      {artifact.size_mb:.2f} MB",
        f"  path:      {artifact.artifact_path}",
    ]
    console.print(Panel("\n".join(body), border_style="green"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Crash:[/] {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
