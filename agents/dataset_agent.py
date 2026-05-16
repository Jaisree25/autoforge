"""Dataset Agent — the Data Preparer.

Real LLM-driven Preparer. Architecture:

  1. LLM (Nemotron-49B with strict json_schema) gets the `DatasetProfile`
     from Profiler and the `StrategySpec` from Researcher. It returns a
     structured prep **plan** — an ordered list of operations to apply.
  2. The agent then dispatches each operation to the matching function in
     `tools/preparation_tools.py`. Data-modifying ops (resize, split,
     impute) actually transform files on disk; config-only ops (normalize,
     augment, scale) record values for the Trainer to apply at runtime.
  3. **Programmatic split backstop** — if the LLM forgot a train/test split,
     the Preparer runs one itself before returning. Trainer + Evaluator both
     require a split, so this guarantees the contract.
  4. Returns a `PreparationReport` listing every applied operation, the
     final prepared-dataset path, and any Trainer config recorded.

Why a strict enum on operation names? Earlier versions used a free-text
`str` field + an alias map to coerce LLM drift (`"Model Definition"`,
`"Preprocessing"`, etc.). The LLM cannot stay in its lane with a free-text
field — it tries to plan the WHOLE pipeline. With a typing.Literal on the
op name, OpenAI strict json_schema mode rejects any name outside the eight
allowed ops at the server. No drift possible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from config import ARTIFACTS_DIR, COORDINATOR_MODEL
from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    DatasetProfile,
    Modality,
    PreparationReport,
    StrategySpec,
)

from agents._llm_client import NemotronClient
from agents.base_agent import BaseAgent
from tools import preparation_tools as prep


# Canonical op names. Strict json_schema with Literal[...] forces the LLM to
# pick from exactly these eight values — no aliases, no drift, no "Model
# Definition" stealing the Trainer's lane.
PrepOpName = Literal[
    # Image ops
    "resize_images",
    "train_test_split_images",
    "set_normalization",
    "set_augmentation",
    # CSV ops
    "impute_missing",
    "encode_categoricals",
    "train_test_split_csv",
    "set_feature_scaling",
]


_IMAGE_OPS: frozenset[str] = frozenset({
    "resize_images", "train_test_split_images",
    "set_normalization", "set_augmentation",
})
_CSV_OPS: frozenset[str] = frozenset({
    "impute_missing", "encode_categoricals",
    "train_test_split_csv", "set_feature_scaling",
})


# ---------------------------------------------------------------------------
class _PrepOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: PrepOpName = Field(
        description="Operation name — MUST be one of the eight enum values.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Operation arguments (key/value pairs).",
    )
    rationale: str = Field(description="One sentence: why this operation is needed.")


class _PrepPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operations: list[_PrepOperation] = Field(
        description="Ordered list of operations to apply (applied sequentially).",
    )
    summary: str = Field(
        description="One-paragraph plain-English summary of the prep plan.",
    )


# ---------------------------------------------------------------------------
class DatasetAgent(BaseAgent):
    """Data Preparer — cleans/augments per Researcher advice + applies tools."""

    name: ClassVar[AgentName] = AgentName.DATASET

    SYSTEM_PROMPT = (
        "You are the Data Preparer, the third agent in the AutoForge pipeline. "
        "The Profiler has handed you a dataset profile; the Researcher has "
        "recommended candidate architectures. Your job is to plan an ordered "
        "list of preparation operations the Trainer needs.\n\n"
        "## Lane discipline (READ THIS FIRST)\n"
        "You ONLY do data preparation. You DO NOT:\n"
        "  - Define the model architecture (that's the Trainer's job)\n"
        "  - Choose loss functions, optimizers, learning rates, or epochs\n"
        "  - Plan training / evaluation / optimization steps\n"
        "  - Load data into DataLoaders or batch it\n"
        "  - Flatten images, convert to tensors, or build pipelines\n"
        "Those belong to downstream agents. Stick to the eight operations below.\n\n"
        "## The ONLY operations you may emit\n"
        "Your output schema enforces these names — anything else is rejected.\n\n"
        "### For IMAGE datasets:\n"
        "  - `resize_images`             args: target_h (int), target_w (int)\n"
        "  - `train_test_split_images`   args: test_size (float, default 0.2)\n"
        "  - `set_normalization`         args: mean (list[float]), std (list[float])\n"
        "  - `set_augmentation`          args: transforms (list[str])\n\n"
        "### For TABULAR (CSV) datasets:\n"
        "  - `impute_missing`            args: strategy (str), columns (list[str])\n"
        "  - `encode_categoricals`       args: method (str), columns (list[str])\n"
        "  - `train_test_split_csv`      args: test_size (float), stratify_by (str | null)\n"
        "  - `set_feature_scaling`       args: method (str), columns (list[str])\n\n"
        "## Argument details\n"
        "- `set_normalization`: For MNIST use `mean=[0.1307], std=[0.3081]`. "
        "  For ImageNet RGB use `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`.\n"
        "- `set_augmentation` transforms: 'rotation:10', 'translate:0.1', "
        "  'hflip', 'vflip', 'crop:28', 'brightness:0.1'. "
        "  Do NOT use 'hflip' on digit-classification (mirrors '6' into '9').\n"
        "- `impute_missing` strategy: one of `median`, `mean`, `mode`, `drop`.\n"
        "- `encode_categoricals` method: one of `onehot`, `label`.\n"
        "- `set_feature_scaling` method: one of `standard`, `minmax`, `robust`.\n\n"
        "## Required ordering\n"
        "- Images: (optional resize) → `train_test_split_images` → `set_normalization` → "
        "  (optional `set_augmentation`).\n"
        "- CSV: `impute_missing` → `encode_categoricals` → `train_test_split_csv` → "
        "  (optional `set_feature_scaling`).\n"
        "- **ALWAYS include a train_test_split op** — the Evaluator needs held-out data.\n\n"
        "## Worked example output (MNIST)\n"
        "```json\n"
        "{\n"
        '  "operations": [\n'
        "    {\n"
        '      "name": "train_test_split_images",\n'
        '      "args": {"test_size": 0.2},\n'
        '      "rationale": "Evaluator needs a held-out test set."\n'
        "    },\n"
        "    {\n"
        '      "name": "set_normalization",\n'
        '      "args": {"mean": [0.1307], "std": [0.3081]},\n'
        '      "rationale": "Standard MNIST normalization stabilizes training."\n'
        "    },\n"
        "    {\n"
        '      "name": "set_augmentation",\n'
        '      "args": {"transforms": ["rotation:10", "translate:0.1"]},\n'
        '      "rationale": "Mild augmentation helps generalization on tiny dataset."\n'
        "    }\n"
        "  ],\n"
        '  "summary": "Split MNIST 80/20, normalize with standard mean/std, '
        'add mild rotation+translate augmentation."\n'
        "}\n"
        "```\n"
    )

    def __init__(self, store, run_id: str) -> None:
        super().__init__(store=store, run_id=run_id)
        # 49B for prep planning — judgment-heavy; the 9B sometimes picks
        # operations outside the supported list (the enum stops that, but the
        # 49B also writes better rationales).
        self.llm = NemotronClient(model=COORDINATOR_MODEL)

    # ------------------------------------------------------------------
    def run(  # type: ignore[override]
        self,
        dataset_profile: DatasetProfile,
        strategy_spec: StrategySpec,
    ) -> PreparationReport:
        source_path = Path(dataset_profile.dataset_path)
        with self._lifecycle(f"prepare {source_path.name}"):
            # --- 1. Plan ---
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"nemotron.plan (model={self.llm.model})",
            )
            plan: _PrepPlan = self.llm.think_and_answer_structured(
                system=self.SYSTEM_PROMPT,
                user=self._build_user_prompt(dataset_profile, strategy_spec),
                schema=_PrepPlan,
                on_thinking=lambda p: self.emit_event(
                    EventType.THINKING, message=p,
                ),
                no_think=True,  # enum-locked schema; no reasoning needed
            )
            self.emit_event(
                EventType.INFO,
                message=f"plan: {len(plan.operations)} operation(s)",
                payload={"summary": plan.summary},
            )

            # --- 2. Execute ---
            artifact_dir = ARTIFACTS_DIR / self.run_id / "prepared"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            applied_ops: list[str] = []
            current_path = source_path
            notes_lines: list[str] = []
            split_applied = False

            for op in plan.operations:
                # Skip ops that target the wrong modality (LLM rarely does
                # this with the enum, but cheap to check).
                if dataset_profile.modality == Modality.IMAGE and op.name in _CSV_OPS:
                    self.emit_event(
                        EventType.WARNING,
                        message=f"skipping `{op.name}` (CSV op on image dataset)",
                    )
                    continue
                if dataset_profile.modality == Modality.TABULAR and op.name in _IMAGE_OPS:
                    self.emit_event(
                        EventType.WARNING,
                        message=f"skipping `{op.name}` (image op on tabular dataset)",
                    )
                    continue

                args_preview = json.dumps(op.args, default=str)
                if len(args_preview) > 100:
                    args_preview = args_preview[:97] + "…"
                self.emit_event(
                    EventType.TOOL_CALL,
                    message=f"{op.name}({args_preview})",
                    payload={"op": op.name, "args": op.args,
                             "rationale": op.rationale},
                )
                try:
                    new_path, note = self._dispatch_op(
                        op, current_path, dataset_profile, artifact_dir,
                    )
                    applied_ops.append(
                        f"{op.name}({json.dumps(op.args, default=str)})"
                    )
                    if op.name in ("train_test_split_images", "train_test_split_csv"):
                        split_applied = True
                    if note:
                        notes_lines.append(f"{op.name}: {note}")
                        self.emit_event(
                            EventType.INFO,
                            message=f"{op.name} → {note}",
                        )
                    if new_path is not None:
                        current_path = new_path
                except Exception as exc:  # noqa: BLE001
                    self.emit_event(
                        EventType.WARNING,
                        message=f"{op.name} failed: {type(exc).__name__}: {exc}",
                    )
                    notes_lines.append(
                        f"FAILED {op.name}: {type(exc).__name__}: {exc}"
                    )

            # --- 3. Programmatic backstop: guarantee a split exists ---
            if not split_applied:
                current_path, note = self._ensure_split(
                    current_path, dataset_profile, artifact_dir,
                )
                if note:
                    applied_ops.append(f"AUTO_SPLIT_BACKSTOP({note})")
                    notes_lines.append(f"auto-split backstop: {note}")

            prepared_path: str | None = None
            if current_path != source_path:
                prepared_path = str(current_path)

            report = PreparationReport(
                original_dataset_path=str(source_path),
                prepared_dataset_path=prepared_path,
                operations=applied_ops,
                summary=plan.summary,
                notes="\n".join(notes_lines),
            )
            self.emit_event(
                EventType.INFO,
                message=(
                    f"prepared: {len(applied_ops)} op(s) applied"
                    + (f", output → `{prepared_path}`" if prepared_path else "")
                ),
            )
        return report

    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        profile: DatasetProfile,
        spec: StrategySpec,
    ) -> str:
        lines = [
            "## Dataset profile (from Profiler)",
            f"- Modality: `{profile.modality.value}`",
            f"- Inferred task: `{profile.task_type.value}`",
            f"- Samples: {profile.n_rows:,}",
        ]
        if profile.modality == Modality.TABULAR:
            lines += [
                f"- Columns: {profile.n_cols}",
                f"- Target: `{profile.target_column}`",
            ]
            if profile.columns:
                lines.append("- Column details:")
                for c in profile.columns[:20]:
                    lines.append(
                        f"  - `{c.name}` ({c.dtype}, missing={c.missing_pct:.1%})"
                    )
        else:
            lines += [
                f"- Classes: {profile.n_classes}",
                f"- Channels: {profile.image_channels}",
                f"- Sample resolutions: {profile.image_resolutions[:5]}",
                f"- Formats: {profile.image_formats}",
            ]
        if profile.class_balance:
            lines.append(
                "- Class balance: "
                + ", ".join(f"{k}={v:.0%}" for k, v in profile.class_balance.items())
            )
        if profile.warnings:
            lines.append("- Profiler warnings:")
            for w in profile.warnings:
                lines.append(f"  - {w}")
        lines += [
            "",
            "## Researcher recommendation (top architecture)",
        ]
        if spec.candidate_architectures:
            arch = spec.candidate_architectures[0]
            lines += [
                f"- Architecture: `{arch.name}` ({arch.family} / `{arch.library}`)",
                f"- Rationale: {arch.rationale}",
            ]
        else:
            lines.append("- (no architectures specified)")
        lines += [
            f"- Success metric: `{spec.success_metric}` ≥ {spec.success_threshold:.2f}",
            "",
            "Plan the preparation steps. Stick to the eight enum operations. "
            "Do NOT add training, evaluation, or model steps.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Programmatic backstop — guarantees the Trainer/Evaluator get a split.
    # ------------------------------------------------------------------
    def _ensure_split(
        self,
        current_path: Path,
        profile: DatasetProfile,
        artifact_dir: Path,
    ) -> tuple[Path, str]:
        """If no split op was applied, run one ourselves with defaults.

        This is the Preparer equivalent of the Researcher's arXiv backstop:
        the LLM is supposed to include a split, but if it forgets we add
        one so downstream agents don't crash. INFO-level event so the
        human sees it in the dashboard.
        """
        if profile.modality == Modality.IMAGE:
            out_dir = artifact_dir / "split"
            self.emit_event(
                EventType.INFO,
                message="auto-split backstop: LLM omitted split → applying default 80/20",
            )
            try:
                result = prep.split_image_dir(
                    source_dir=current_path,
                    test_size=0.2,
                    output_dir=out_dir,
                )
                return (
                    Path(result["output_dir"]),
                    f"train={result['train_count']}, test={result['test_count']}",
                )
            except Exception as exc:  # noqa: BLE001
                self.emit_event(
                    EventType.WARNING,
                    message=f"auto-split backstop FAILED: {type(exc).__name__}: {exc}",
                )
                return current_path, ""

        if profile.modality == Modality.TABULAR:
            out_dir = artifact_dir / "split"
            self.emit_event(
                EventType.INFO,
                message="auto-split backstop: LLM omitted split → applying default 80/20",
            )
            try:
                result = prep.split_train_test_csv(
                    source_path=current_path,
                    test_size=0.2,
                    stratify_by=profile.target_column,
                    output_dir=out_dir,
                )
                return (
                    Path(result["output_dir"]),
                    f"train={result['train_count']}, test={result['test_count']}",
                )
            except Exception as exc:  # noqa: BLE001
                self.emit_event(
                    EventType.WARNING,
                    message=f"auto-split backstop FAILED: {type(exc).__name__}: {exc}",
                )
                return current_path, ""

        return current_path, ""

    # ------------------------------------------------------------------
    def _dispatch_op(
        self,
        op: _PrepOperation,
        current_path: Path,
        profile: DatasetProfile,
        artifact_dir: Path,
    ) -> tuple[Path | None, str]:
        """Apply one operation. Returns (new_path | None, human-readable note).

        `new_path` is None for config-only ops (normalization, augmentation,
        feature_scaling). Op names are already canonical (enum-enforced),
        so no aliasing needed.
        """
        name = op.name
        args = op.args

        # === Image ops ===
        if name == "resize_images":
            target_h = int(args.get("target_h", 28))
            target_w = int(args.get("target_w", 28))
            out_dir = artifact_dir / f"resized_{target_h}x{target_w}"
            result = prep.resize_images_dir(
                source_dir=current_path,
                target_h=target_h, target_w=target_w,
                output_dir=out_dir,
            )
            return (
                Path(result["output_dir"]),
                f"resized {result['resized_count']} images to "
                f"{target_h}×{target_w}",
            )

        if name == "train_test_split_images":
            test_size = float(args.get("test_size", 0.2))
            out_dir = artifact_dir / "split"
            result = prep.split_image_dir(
                source_dir=current_path,
                test_size=test_size,
                output_dir=out_dir,
            )
            return (
                Path(result["output_dir"]),
                f"train={result['train_count']}, test={result['test_count']}",
            )

        if name == "set_normalization":
            mean = list(args.get("mean", []))
            std = list(args.get("std", []))
            prep.record_normalization(mean=mean, std=std)
            return None, f"mean={mean}, std={std}"

        if name == "set_augmentation":
            transforms = list(args.get("transforms", []))
            prep.record_augmentation(transforms=transforms)
            return None, f"transforms={transforms}"

        # === CSV ops ===
        if name == "impute_missing":
            strategy = str(args.get("strategy", "median"))
            columns = args.get("columns") or None
            out_path = artifact_dir / (current_path.stem + "_imputed.csv")
            result = prep.impute_missing_csv(
                source_path=current_path,
                strategy=strategy,
                columns=columns,
                output_path=out_path,
            )
            return (
                Path(result["output_path"]),
                f"strategy={strategy}, columns={result['imputed_columns']}",
            )

        if name == "encode_categoricals":
            method = str(args.get("method", "onehot"))
            columns = list(args.get("columns", []))
            out_path = artifact_dir / (current_path.stem + "_encoded.csv")
            result = prep.encode_categoricals_csv(
                source_path=current_path,
                method=method,
                columns=columns,
                output_path=out_path,
            )
            return (
                Path(result["output_path"]),
                f"method={method}, columns={result['encoded_columns']}",
            )

        if name == "train_test_split_csv":
            test_size = float(args.get("test_size", 0.2))
            stratify_by = args.get("stratify_by") or profile.target_column
            out_dir = artifact_dir / "split"
            result = prep.split_train_test_csv(
                source_path=current_path,
                test_size=test_size,
                stratify_by=stratify_by,
                output_dir=out_dir,
            )
            return (
                Path(result["output_dir"]),
                f"train={result['train_count']}, test={result['test_count']}",
            )

        if name == "set_feature_scaling":
            method = str(args.get("method", "standard"))
            columns = list(args.get("columns", []))
            prep.record_feature_scaling(method=method, columns=columns)
            return None, f"method={method}, columns={columns}"

        # Unreachable: Literal enum guarantees one of the above.
        raise ValueError(f"Unhandled op (enum drift?): {name!r}")
