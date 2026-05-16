"""Profiler Agent — the first observer.

Detects input modality (CSV vs image dir), reads structural info mechanically
(pandas for CSV, PIL for images), and asks Nemotron to fill judgment fields
(task type, target column / class count, warnings, plain-English summary).

Streams `/think` reasoning into the chat feed via `EventType.THINKING` so
judges see the agent actually thinking before it commits to a profile.

Schema split:
  - **Mechanical fields** (n_rows, columns, image_resolutions, ...) come from
    pandas / PIL directly. The LLM never sees these.
  - **Judgment fields** (task_type, target_column, warnings, summary) come
    from a small private schema (`_CsvJudgment` / `_ImageJudgment`) that the
    LLM populates. We then merge with the mechanical fields to build the
    final `DatasetProfile`.

This keeps the LLM's structured-output surface tiny → reliable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import pandas as pd
import psutil
from pydantic import BaseModel, ConfigDict, Field

from config import WORKER_MODEL
from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    ColumnProfile,
    DatasetProfile,
    Modality,
    TaskType,
    TrainingEnvelope,
)

from agents._llm_client import NemotronClient
from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Small private schemas for the LLM judgment call.
# Kept flat (no nested models) so OpenAI's strict json_schema mode accepts it.
# ---------------------------------------------------------------------------
class _CsvJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: TaskType
    warnings: list[str] = Field(default_factory=list)
    summary: str
    target_column: str | None = None


# ---------------------------------------------------------------------------
class ProfilerAgent(BaseAgent):
    name: ClassVar[AgentName] = AgentName.PROFILER

    CSV_SYSTEM_PROMPT = (
        "You are the Profiler, the first agent in the AutoForge pipeline. "
        "You observe a tabular dataset and the user's objective, then hand "
        "off your observations to the Researcher who will pick architectures.\n\n"
        "Your job is to infer:\n"
        "  - task_type: which kind of supervised problem is this\n"
        "  - target_column: which column the model should predict (or null if "
        "    not obvious from the data + objective)\n"
        "  - warnings: things the Preparer should know — missing data, "
        "    cardinality issues, leakage risks, class imbalance\n"
        "  - summary: one short paragraph describing the dataset suitable "
        "    for a teammate who will only read this sentence\n\n"
        "Be honest about uncertainty. Better to leave target_column null "
        "than to guess wrong."
    )

    def __init__(self, store, run_id: str) -> None:
        super().__init__(store=store, run_id=run_id)
        self.llm = NemotronClient(model=WORKER_MODEL)

    # ------------------------------------------------------------------
    # Public entry — dispatches by modality
    # ------------------------------------------------------------------
    def run(self, dataset_path: str, objective: str) -> DatasetProfile:  # type: ignore[override]
        path = Path(dataset_path)
        summary = f"observe {path.name} (tabular)"

        with self._lifecycle(summary):
            self.emit_event(
                EventType.INFO,
                message="detected modality: tabular (sklearn-only)",
                payload={"path": str(path)},
            )
            # AutoForge narrowed to tabular CSV + sklearn. Image inputs
            # used to be supported via _profile_images but were dropped
            # when the Preparer / Trainer dropped their image paths.
            if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
                raise ValueError(
                    f"AutoForge supports only CSV/TSV inputs. Got: {path}"
                )
            return self._profile_csv(path, objective)

    # ------------------------------------------------------------------
    # CSV branch
    # ------------------------------------------------------------------
    def _profile_csv(self, path: Path, objective: str) -> DatasetProfile:
        self.emit_event(EventType.TOOL_CALL, message=f"pandas.read_csv('{path.name}')")
        df = pd.read_csv(path)
        n_rows, n_cols = df.shape

        # Mechanical column profiles
        columns: list[ColumnProfile] = []
        for col in df.columns:
            series = df[col]
            try:
                missing_pct = float(series.isna().mean())
            except Exception:
                missing_pct = 0.0
            try:
                unique = int(series.nunique(dropna=True))
            except Exception:
                unique = 0
            columns.append(ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                missing_pct=missing_pct,
                unique_count=unique,
            ))

        # Compact preview the LLM can reason over (don't dump the whole frame)
        preview_csv = df.head(8).to_csv(index=False)
        col_summary_lines = [
            f"  - {c.name}: dtype={c.dtype}, missing={c.missing_pct:.1%}, "
            f"unique={c.unique_count}"
            for c in columns
        ]
        user_prompt = (
            f"Objective: {objective}\n\n"
            f"Dataset: {path.name} ({n_rows:,} rows × {n_cols} columns)\n\n"
            f"Columns:\n" + "\n".join(col_summary_lines) + "\n\n"
            f"First 8 rows (CSV):\n```\n{preview_csv}```\n\n"
            "Infer the task type, target column, any warnings the Preparer "
            "should know about, and a one-paragraph summary."
        )

        self.emit_event(
            EventType.TOOL_CALL,
            message=f"nemotron.think (model={self.llm.model})",
        )
        judgment: _CsvJudgment = self.llm.think_and_answer_structured(
            system=self.CSV_SYSTEM_PROMPT,
            user=user_prompt,
            schema=_CsvJudgment,
            on_thinking=lambda p: self.emit_event(
                EventType.THINKING, message=p,
            ),
        )

        # Derive class balance from the inferred target if classification.
        class_balance: dict[str, float] | None = None
        if (
            judgment.target_column
            and judgment.target_column in df.columns
            and judgment.task_type in (
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
            )
        ):
            try:
                counts = df[judgment.target_column].value_counts(normalize=True, dropna=True)
                class_balance = {str(k): float(v) for k, v in counts.items()}
            except Exception:
                class_balance = None

        profile = DatasetProfile(
            dataset_path=str(path),
            modality=Modality.TABULAR,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            target_column=judgment.target_column,
            task_type=judgment.task_type,
            class_balance=class_balance,
            warnings=judgment.warnings,
            profile_summary=judgment.summary,
        )
        self.emit_event(
            EventType.INFO,
            message=(
                f"CSV profile ready: {n_rows}×{n_cols}, "
                f"task={judgment.task_type.value}, "
                f"target={judgment.target_column!r}"
            ),
        )
        return profile

    # ------------------------------------------------------------------
    # Hardware envelope — real probe.
    # CPU + memory via psutil; GPU via `nvidia-smi` subprocess (best effort,
    # gracefully reports CPU-only when the binary isn't on PATH). Trainer
    # budget (max_trials, max_train_minutes, allowed libraries) is derived
    # from what was actually measured.
    # ------------------------------------------------------------------
    def run_envelope(self) -> TrainingEnvelope:
        with self._lifecycle("detect hardware envelope"):
            cpu_count, mem_gb = self._probe_cpu_mem()
            gpu_name, gpu_mem_gb = self._probe_gpu()

            gpu_available = gpu_name is not None
            if gpu_available:
                max_trials = 20
                max_train_minutes = 5.0
                batch_size_range = (32, 256)
                allowed_libraries = ["sklearn", "pytorch", "torchvision"]
                mixed_precision = (gpu_mem_gb or 0.0) >= 16.0
                notes = (
                    f"GPU detected: {gpu_name} "
                    f"({gpu_mem_gb:.0f}GB). "
                    f"CPU: {cpu_count} cores · system RAM: {mem_gb:.0f}GB. "
                    f"Allowing 20 trials over ≤5 min."
                )
            else:
                # CPU-only: shrink the envelope. sklearn is the only
                # library we exercise without a GPU.
                max_trials = max(4, min(8, cpu_count))
                max_train_minutes = 2.0
                batch_size_range = (16, 128)
                allowed_libraries = ["sklearn"]
                mixed_precision = False
                notes = (
                    f"No GPU detected — running CPU-only on "
                    f"{cpu_count} cores ({mem_gb:.0f}GB RAM). "
                    f"Capping trials at {max_trials} over ≤"
                    f"{max_train_minutes:.0f} min; sklearn-only."
                )

            envelope = TrainingEnvelope(
                gpu_available=gpu_available,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_mem_gb,
                cpu_count=cpu_count,
                system_memory_gb=mem_gb,
                max_train_minutes=max_train_minutes,
                max_trials=max_trials,
                batch_size_range=batch_size_range,
                allowed_libraries=allowed_libraries,
                mixed_precision=mixed_precision,
                notes=notes,
            )
            self.emit_event(
                EventType.INFO,
                message=(
                    f"envelope: gpu={envelope.gpu_name or 'none'}, "
                    f"cpu={envelope.cpu_count}, "
                    f"max_trials={envelope.max_trials}"
                ),
            )
        return envelope

    # ------------------------------------------------------------------
    # Hardware probes — keep tiny + side-effect-free.
    # ------------------------------------------------------------------
    def _probe_cpu_mem(self) -> tuple[int, float]:
        self.emit_event(
            EventType.TOOL_CALL,
            message="psutil.cpu_count() + psutil.virtual_memory()",
        )
        cpu_count = psutil.cpu_count(logical=True) or 1
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        return cpu_count, round(mem_gb, 1)

    def _probe_gpu(self) -> tuple[str | None, float | None]:
        """Try `nvidia-smi`. Returns `(name, memory_gb)` or `(None, None)`.

        We deliberately avoid `pynvml` so this works on machines that have
        the NVIDIA driver but not the Python binding installed.
        """
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            self.emit_event(
                EventType.INFO,
                message="nvidia-smi not on PATH — assuming no GPU",
            )
            return None, None

        self.emit_event(
            EventType.TOOL_CALL,
            message="nvidia-smi --query-gpu=name,memory.total",
        )
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            self.emit_event(
                EventType.WARNING,
                message=f"nvidia-smi failed: {type(exc).__name__}: {exc}",
            )
            return None, None

        if result.returncode != 0 or not result.stdout.strip():
            return None, None

        first = result.stdout.strip().splitlines()[0]
        try:
            name_part, mem_part = first.split(",", 1)
            return name_part.strip(), round(float(mem_part.strip()) / 1024.0, 1)
        except (ValueError, IndexError):
            return None, None


