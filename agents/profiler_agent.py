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

import time
from pathlib import Path
from typing import ClassVar

import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from config import STUB_AGENT_SLEEP, WORKER_MODEL
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


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Small private schemas for the LLM judgment calls.
# Kept flat (no nested models) so OpenAI's strict json_schema mode accepts them.
# ---------------------------------------------------------------------------
class _Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: TaskType
    warnings: list[str] = Field(default_factory=list)
    summary: str


class _CsvJudgment(_Judgment):
    target_column: str | None = None


class _ImageJudgment(_Judgment):
    recommended_input_height: int | None = None
    recommended_input_width: int | None = None


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

    IMAGE_SYSTEM_PROMPT = (
        "You are the Profiler, the first agent in the AutoForge pipeline. "
        "You observe an image dataset and the user's objective, then hand "
        "off your observations to the Researcher.\n\n"
        "Your job is to infer:\n"
        "  - task_type: typically image_classification when classes are "
        "    folders\n"
        "  - recommended_input_height/width: the input resolution the "
        "    Trainer should feed to its model (typical CNN inputs are "
        "    224x224 or 256x256; ViT-Base wants 224x224)\n"
        "  - warnings: e.g. resolution variation, class imbalance, very "
        "    small training set\n"
        "  - summary: one short paragraph describing the dataset\n\n"
        "Use round, standard input sizes. Don't invent unusual resolutions."
    )

    def __init__(self, store, run_id: str) -> None:
        super().__init__(store=store, run_id=run_id)
        self.llm = NemotronClient(model=WORKER_MODEL)

    # ------------------------------------------------------------------
    # Public entry — dispatches by modality
    # ------------------------------------------------------------------
    def run(self, dataset_path: str, objective: str) -> DatasetProfile:  # type: ignore[override]
        path = Path(dataset_path)
        modality = _detect_modality(path)
        summary = f"observe {path.name} ({modality.value})"

        with self._lifecycle(summary):
            self.emit_event(
                EventType.INFO,
                message=f"detected modality: {modality.value}",
                payload={"path": str(path)},
            )
            if modality == Modality.TABULAR:
                return self._profile_csv(path, objective)
            if modality == Modality.IMAGE:
                return self._profile_images(path, objective)
            raise ValueError(
                f"Unsupported modality at {path}. Profiler currently handles "
                "CSV files and directories of images."
            )

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
    # Image branch
    # ------------------------------------------------------------------
    def _profile_images(self, path: Path, objective: str) -> DatasetProfile:
        # If a single image file was passed, treat its parent as the dataset.
        if path.is_file() and path.suffix.lower() in _IMG_EXTS:
            path = path.parent

        self.emit_event(
            EventType.TOOL_CALL,
            message=f"glob images under '{path.name}/'",
        )
        image_paths = [p for p in path.rglob("*") if p.suffix.lower() in _IMG_EXTS]
        n_images = len(image_paths)

        if n_images == 0:
            raise ValueError(f"No images found under {path}")

        # Class structure: immediate subdirectories of `path` count as classes
        class_dirs = [d for d in path.iterdir() if d.is_dir()]
        class_balance: dict[str, float] | None = None
        n_classes: int | None = None
        if class_dirs:
            counts: dict[str, int] = {}
            for d in class_dirs:
                cls_imgs = [p for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS]
                counts[d.name] = len(cls_imgs)
            total = sum(counts.values())
            if total > 0:
                class_balance = {k: v / total for k, v in counts.items()}
                n_classes = len(counts)

        # Sample images to derive resolution / format / channels
        self.emit_event(
            EventType.TOOL_CALL,
            message=f"PIL.Image.open × {min(20, n_images)} samples",
        )
        sample = image_paths[: min(20, n_images)]
        resolutions: list[tuple[int, int]] = []
        formats: set[str] = set()
        channels: int | None = None
        for ip in sample:
            try:
                with Image.open(ip) as img:
                    resolutions.append((img.size[0], img.size[1]))  # (W, H)
                    if img.format:
                        formats.add(img.format.lower())
                    if channels is None:
                        channels = len(img.getbands())
            except Exception as exc:  # noqa: BLE001
                self.emit_event(
                    EventType.WARNING,
                    message=f"could not read {ip.name}: {exc}",
                )

        # Build prompt for LLM judgment
        res_min = (min(r[0] for r in resolutions), min(r[1] for r in resolutions)) if resolutions else None
        res_max = (max(r[0] for r in resolutions), max(r[1] for r in resolutions)) if resolutions else None
        class_block = (
            "\n".join(f"  - {c}: {counts[c]} images" for c in counts)
            if class_dirs else "  (no class subdirectories — unlabeled images)"
        )
        user_prompt = (
            f"Objective: {objective}\n\n"
            f"Dataset: {path.name} ({n_images} images)\n"
            f"Classes ({n_classes or 0}):\n{class_block}\n\n"
            f"Sample resolutions (W×H): {resolutions[:5]}\n"
            f"Resolution range: {res_min} to {res_max}\n"
            f"Image formats: {sorted(formats)}\n"
            f"Channels per image: {channels}\n\n"
            "Infer task type, recommended input resolution for the Trainer, "
            "warnings (resolution variation? class imbalance? tiny dataset?), "
            "and a one-paragraph summary."
        )

        self.emit_event(
            EventType.TOOL_CALL,
            message=f"nemotron.think (model={self.llm.model})",
        )
        judgment: _ImageJudgment = self.llm.think_and_answer_structured(
            system=self.IMAGE_SYSTEM_PROMPT,
            user=user_prompt,
            schema=_ImageJudgment,
            on_thinking=lambda p: self.emit_event(
                EventType.THINKING, message=p,
            ),
        )

        profile = DatasetProfile(
            dataset_path=str(path),
            modality=Modality.IMAGE,
            n_rows=n_images,
            n_cols=0,  # not applicable
            image_resolutions=resolutions,
            image_channels=channels,
            image_formats=sorted(formats),
            n_classes=n_classes,
            task_type=judgment.task_type,
            class_balance=class_balance,
            warnings=judgment.warnings,
            profile_summary=judgment.summary,
        )
        self.emit_event(
            EventType.INFO,
            message=(
                f"image profile ready: {n_images} images, "
                f"{n_classes or 0} class(es), "
                f"task={judgment.task_type.value}"
            ),
        )
        return profile

    # ------------------------------------------------------------------
    # Hardware envelope (still a stub — pynvml integration is a separate
    # task; this lets Trainer have a valid envelope downstream).
    # ------------------------------------------------------------------
    def run_envelope(self) -> TrainingEnvelope:
        with self._lifecycle("detect hardware envelope"):
            gpu_available = False
            gpu_name = None
            gpu_memory_gb = 0.0
            mixed_precision = False
            notes_parts = []
            
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    raw_name = pynvml.nvmlDeviceGetName(handle)
                    gpu_name = (
                        raw_name.decode() if isinstance(raw_name, bytes) else raw_name)
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_gb = round(mem_info.total / 1024**3, 1)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    mem_util_pct = util.memory
                    
                    temp_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    
                    gpu_available = True
                    mixed_precision = gpu_memory_gb >= 16.0  # safe threshold
                    
                    self.emit_event(
                        EventType.TOOL_CALL,
                        message=(
                            f"pynvml: {gpu_name} | "
                            f"{gpu_memory_gb:.1f} GB VRAM | "
                            f"GPU util {gpu_util_pct}% | "
                            f"Mem util {mem_util_pct}% | "
                            f"Temp {temp_c}°C"
                        ),
                    )
                    notes_parts.append(
                        f"{gpu_name} detected ({gpu_memory_gb:.1f} GB VRAM, "
                        f"{gpu_util_pct}% util, {temp_c}°C)."
                    )
                    
                else:
                    self.emit_event(EventType.WARNING, message="pynvml: no GPUs found")
                    notes_parts.append("No GPU detected — CPU-only mode.")
                
                pynvml.nvmlShutdown()
                    
            except Exception as exc:
                self.emit_event(
                    EventType.WARNING,
                    message=f"pynvml unavailable: {exc} — falling back to CPU-only",
                )
                notes_parts.append(f"GPU detection failed ({exc}); CPU-only mode.")
                    
            try:
                        
                import psutil
                cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
                cpu_util_pct = psutil.cpu_percent(interval=0.5)
                        
                ram = psutil.virtual_memory()
                system_memory_gb = round(ram.total / 1024**3, 1)
                ram_used_pct = ram.percent
                        
                self.emit_event(
                    EventType.TOOL_CALL,
                    message=(
                        f"psutil: {cpu_count} cores | "
                        f"CPU util {cpu_util_pct}% | "
                        f"{system_memory_gb:.1f} GB RAM | "
                        f"RAM used {ram_used_pct}%"
                    ),
                )
                
                notes_parts.append(
                    f"{cpu_count} CPU cores, {system_memory_gb:.1f} GB RAM "
                    f"({ram_used_pct}% used)."
                )
                        
            except Exception as exc:
                self.emit_event(
                    EventType.WARNING,
                    message=f"psutil unavailable: {exc} — using fallback defaults",
                )
                cpu_count = 4
                system_memory_gb = 16.0
                notes_parts.append(f"CPU/RAM detection failed ({exc}); using defaults.")

        # ── Derive trial budget from available VRAM ─────────────────
            if gpu_memory_gb >= 40:
                max_trials = 20
                max_train_minutes = 5.0
                batch_size_range = (32, 256)
            elif gpu_memory_gb >= 16:
                max_trials = 12
                max_train_minutes = 8.0
                batch_size_range = (16, 128)
            elif gpu_memory_gb > 0:
                max_trials = 6
                max_train_minutes = 15.0
                batch_size_range = (8, 64)
            else:
            # CPU-only
                max_trials = 3
                max_train_minutes = 20.0
                batch_size_range = (8, 32)

            notes_parts.append(
                f"Trial budget: {max_trials} trials, "
                f"{max_train_minutes:.0f} min cap, "
                f"batch {batch_size_range[0]}–{batch_size_range[1]}."
            )

            envelope = TrainingEnvelope(
                gpu_available=gpu_available,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                cpu_count=cpu_count,
                system_memory_gb=system_memory_gb,
                max_train_minutes=max_train_minutes,
                max_trials=max_trials,
                batch_size_range=batch_size_range,
                allowed_libraries=["xgboost", "sklearn", "pytorch", "torchvision"],
                mixed_precision=mixed_precision,
                notes=" ".join(notes_parts),
            )

            self.emit_event(
                EventType.INFO,
                message=(
                    f"envelope: gpu={envelope.gpu_name or 'none'} | "
                    f"{envelope.gpu_memory_gb:.1f} GB VRAM | "
                    f"{envelope.cpu_count} cores | "
                    f"{envelope.system_memory_gb:.1f} GB RAM | "
                    f"max_trials={envelope.max_trials}"
                ),
            )
        return envelope
        # with self._lifecycle("detect hardware envelope"):
        #     self.emit_event(
        #         EventType.TOOL_CALL,
        #         message="pynvml.nvmlDeviceGetCount() [stub: simulating L40S]",
        #     )
        #     time.sleep(STUB_AGENT_SLEEP)
        #     envelope = TrainingEnvelope(
        #         gpu_available=True,
        #         gpu_name="NVIDIA L40S",
        #         gpu_memory_gb=48.0,
        #         cpu_count=16,
        #         system_memory_gb=128.0,
        #         max_train_minutes=5.0,
        #         max_trials=20,
        #         batch_size_range=(32, 256),
        #         allowed_libraries=["xgboost", "sklearn", "pytorch", "torchvision"],
        #         mixed_precision=False,
        #         notes=(
        #             "L40S has headroom for tabular and small-image workloads. "
        #             "Capping trials at 20 for ≤5-min iteration time."
        #         ),
        #     )
        #     self.emit_event(
        #         EventType.INFO,
        #         message=(
        #             f"envelope: gpu={envelope.gpu_name}, "
        #             f"max_trials={envelope.max_trials}"
        #         ),
        #     )
        # return envelope


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------
def _detect_modality(path: Path) -> Modality:
    if not path.exists():
        return Modality.UNKNOWN

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            return Modality.TABULAR
        if suffix in _IMG_EXTS:
            return Modality.IMAGE
        return Modality.UNKNOWN

    if path.is_dir():
        # Prefer images if directory contains them; else look for a CSV.
        for p in path.rglob("*"):
            if p.suffix.lower() in _IMG_EXTS:
                return Modality.IMAGE
        for p in path.glob("*.csv"):
            return Modality.TABULAR

    return Modality.UNKNOWN
