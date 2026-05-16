"""Pydantic v2 schemas for inter-agent contracts.

Every agent's `run()` returns a dict that validates against one of these
schemas. Every schema is the canonical data shape persisted by `MemoryStore`
and surfaced by the dashboard. Keep these stable — changing them is a breaking
change across the whole pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AgentName(str, Enum):
    """Canonical identifier for each agent in the pipeline.

    Display names (rendered in the dashboard) are decoupled — see
    `dashboard.agent_identity`. The enum values are stable identifiers used
    across persistence + contracts and should not be renamed.

    Pipeline role mapping (post-refactor):
      PROFILER     — observes dataset shape, hardware, parses objective
      STRATEGY     — Researcher: literature + candidate architectures
      DATASET      — Preparer: cleans CSV / resizes-augments images
      TRAINING     — Trainer: HPO loop
      BENCHMARK    — Evaluator
      HARDWARE     — Optimizer (post-training quantize/export only)
      COORDINATOR  — Director / orchestrator
    """

    PROFILER = "profiler"
    DATASET = "dataset"
    STRATEGY = "strategy"
    HARDWARE = "hardware"
    TRAINING = "training"
    BENCHMARK = "benchmark"
    COORDINATOR = "coordinator"


class PipelineStatus(str, Enum):
    """Lifecycle states for a single end-to-end pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Supervised-learning task type, inferred by the Dataset Agent."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    IMAGE_CLASSIFICATION = "image_classification"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    """Input data modality. Determines which set of `DatasetProfile` fields
    are populated and which Preparer tool surface the next stage will use."""

    TABULAR = "tabular"
    IMAGE = "image"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Base config — shared by all schemas
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    """Project-wide base model.

    - `extra="forbid"` so a typo in an agent's output blows up loud and early.
    - `use_enum_values=False` so enums round-trip cleanly through SQLite JSON.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Dataset Agent output
# ---------------------------------------------------------------------------
class ColumnProfile(_Base):
    name: str
    dtype: str
    nullable: bool = False
    missing_pct: float = Field(0.0, ge=0.0, le=1.0)
    unique_count: int | None = None
    sample_values: list[Any] = Field(default_factory=list)


class DatasetProfile(_Base):
    """Output of the Profiler. Common across modalities, with modality-
    specific fields populated only when relevant.

    Tabular CSV path uses: `n_rows`, `n_cols`, `columns`, `target_column`.
    Image directory path uses: `image_resolutions`, `image_channels`,
    `image_formats`, `n_classes`. `n_rows` doubles as image count.
    `class_balance`, `task_type`, `warnings`, `profile_summary` apply to both.
    """

    dataset_path: str
    modality: Modality = Modality.TABULAR
    n_rows: int = Field(ge=0)  # n_samples — rows for CSV, images for image dir
    n_cols: int = Field(ge=0, default=0)  # CSV-only

    # CSV-specific
    columns: list[ColumnProfile] = Field(default_factory=list)
    target_column: str | None = None

    # Image-specific
    image_resolutions: list[tuple[int, int]] = Field(default_factory=list)  # (W, H) samples
    image_channels: int | None = None                # 1 grayscale, 3 RGB, 4 RGBA
    image_formats: list[str] = Field(default_factory=list)
    n_classes: int | None = None

    # Common
    task_type: TaskType = TaskType.UNKNOWN
    class_balance: dict[str, float] | None = None
    warnings: list[str] = Field(default_factory=list)
    profile_summary: str = ""


# ---------------------------------------------------------------------------
# Strategy Agent output
# ---------------------------------------------------------------------------
class Citation(_Base):
    title: str
    url: str | None = None
    source: str = "tavily"  # "tavily" | "arxiv" | "manual"
    snippet: str = ""


class CandidateArchitecture(_Base):
    """One model family the Strategy Agent recommends trying."""

    name: str
    family: str  # "gradient_boost" | "linear" | "neural_net" | ...
    library: str  # "xgboost" | "sklearn" | "torch" | ...
    hyperparameter_space: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class StrategySpec(_Base):
    """Output of the Strategy Agent — formalized objective + research."""

    objective: str
    task_type: TaskType
    success_metric: str  # "f1" | "accuracy" | "auc" | "rmse" | ...
    success_threshold: float
    candidate_architectures: list[CandidateArchitecture]
    research_summary: str = ""
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hardware Agent output
# ---------------------------------------------------------------------------
class TrainingEnvelope(_Base):
    """Output of the Hardware Agent (pre-training pass).

    Bounds the Training Agent: what hardware is available, what training budget
    fits, which libraries are usable.
    """

    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    cpu_count: int = 1
    system_memory_gb: float = 0.0
    max_train_minutes: float = 5.0
    max_trials: int = 20
    batch_size_range: tuple[int, int] = (16, 128)
    allowed_libraries: list[str] = Field(default_factory=lambda: ["xgboost", "sklearn"])
    mixed_precision: bool = False
    notes: str = ""


class DeploymentArtifact(_Base):
    """Output of the Optimizer (post-training pass) — optimized model."""

    artifact_path: str
    format: str  # "onnx" | "tensorrt" | "torchscript" | "pickle"
    quantization: str | None = None  # "fp16" | "int8" | None
    size_mb: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Dataset Preparer output (was Dataset Agent before the refactor; now sits
# AFTER Researcher and applies cleaning / augmentation per research advice).
# ---------------------------------------------------------------------------
class PreparationReport(_Base):
    """Output of the Data Preparer."""

    original_dataset_path: str
    prepared_dataset_path: str | None = None
    operations: list[str] = Field(default_factory=list)  # ["impute_median(age)", "resize(224x224)", ...]
    # JSON file recording the Preparer's config-only decisions (normalization
    # mean/std, augmentation transforms, feature scaling method). The Trainer's
    # generated train.py reads this so the operations actually take effect.
    prep_config_path: str | None = None
    summary: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Training Agent output
# ---------------------------------------------------------------------------
class TrialResult(_Base):
    trial_id: int
    params: dict[str, Any] = Field(default_factory=dict)
    score: float
    duration_seconds: float = 0.0
    status: str = "completed"  # "completed" | "pruned" | "failed"


class TrainingResult(_Base):
    """Output of the Training Agent.

    Lane discipline note: `best_score` is the val-accuracy the training
    subprocess measured at the end of fit() — it's a sanity check before
    handoff, NOT the official benchmark. The Evaluator re-loads the saved
    model and produces the authoritative BenchmarkReport (accuracy, latency,
    throughput, PASS/FAIL). The Trainer's UI focuses on training PROCESS:
    iterations, loss curve, wall time, effective hyperparameters.
    """

    best_model_id: str
    metric_name: str
    best_score: float
    best_params: dict[str, Any] = Field(default_factory=dict)
    trials_completed: int = Field(ge=0)
    total_trials: int = Field(ge=0)
    training_time_seconds: float = 0.0
    artifact_path: str
    library: str = ""
    all_trials: list[TrialResult] = Field(default_factory=list)
    # Training-process metrics extracted from the fitted estimator (sklearn
    # attributes like `n_iter_`, `loss_curve_`, `best_loss_`, `train_score_`,
    # plus `get_params()` snapshot). Populated by the Trainer subprocess via
    # introspection of the fitted model. Empty if the estimator class exposes
    # none of those attributes.
    training_process: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Benchmark Agent output
# ---------------------------------------------------------------------------
class LatencyStats(_Base):
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0


class ParetoPoint(_Base):
    config_id: str
    accuracy: float
    latency_ms: float
    memory_mb: float = 0.0


class BenchmarkReport(_Base):
    """Output of the Benchmark Agent."""

    model_id: str
    accuracy_metric: str
    accuracy_value: float
    latency: LatencyStats = Field(default_factory=LatencyStats)
    throughput_qps: float = 0.0
    memory_mb: float = 0.0
    passed_threshold: bool = False
    pareto_frontier: list[ParetoPoint] = Field(default_factory=list)
    feedback_to_training: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Pipeline-level state (composed from above)
# ---------------------------------------------------------------------------
class PipelineRun(_Base):
    """Top-level record for one end-to-end run.

    Persisted by `MemoryStore`. Each agent's output is referenced by its own
    table row; this object is the convenience view the dashboard renders.
    """

    run_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    objective: str
    dataset_path: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    dataset_profile: DatasetProfile | None = None
    strategy_spec: StrategySpec | None = None
    training_envelope: TrainingEnvelope | None = None
    preparation_report: PreparationReport | None = None
    training_result: TrainingResult | None = None
    deployment_artifact: DeploymentArtifact | None = None
    benchmark_report: BenchmarkReport | None = None
    error: str | None = None
