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

from config import ARTIFACTS_DIR, WORKER_MODEL
from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    DatasetProfile,
    PreparationReport,
    StrategySpec,
)

from agents._llm_client import NemotronClient
from agents.base_agent import BaseAgent
from tools import preparation_tools as prep


# Canonical op names. Strict json_schema with Literal[...] forces the LLM to
# pick from exactly these five values — no aliases, no drift. Image ops were
# removed when AutoForge narrowed to tabular-sklearn-only.
PrepOpName = Literal[
    "drop_columns",
    "impute_missing",
    "encode_categoricals",
    "train_test_split_csv",
    "set_feature_scaling",
]


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
        "You ONLY do tabular data preparation for sklearn. You DO NOT:\n"
        "  - Define the model architecture (that's the Trainer's job)\n"
        "  - Choose loss functions, optimizers, learning rates, or epochs\n"
        "  - Plan training / evaluation / optimization steps\n"
        "  - Touch image data — AutoForge is CSV/sklearn-only\n"
        "Stick to the four operations below.\n\n"
        "## The ONLY operations you may emit (TABULAR CSV)\n"
        "Your output schema enforces these names — anything else is rejected.\n\n"
        "  - `drop_columns`              args: columns (list[str])\n"
        "  - `impute_missing`            args: strategy (str), columns (list[str])\n"
        "  - `encode_categoricals`       args: method (str), columns (list[str])\n"
        "  - `train_test_split_csv`      args: test_size (float), stratify_by (str | null)\n"
        "  - `set_feature_scaling`       args: method (str), columns (list[str])\n\n"
        "## Argument details\n"
        "- `drop_columns`: USE THIS FIRST. List columns to remove from the "
        "feature set. Drop:\n"
        "    (a) ID-like columns: anything ending in `_id`, `Id`, "
        "        named `id`, `customer_id`, `user_id`, `uuid`, `PassengerId`. "
        "        These leak or behave like row indices.\n"
        "    (b) High-cardinality string columns (>50 unique values), e.g. "
        "        `Name`, `Ticket`, `Address`. One-hot would explode column count.\n"
        "    (c) Columns with >50% missing values (e.g. `Cabin`). Imputing "
        "        them adds more noise than signal.\n"
        "  NEVER drop the target column.\n"
        "- `impute_missing` strategy: one of `median`, `mean`, `mode`, `drop`.\n"
        "- `encode_categoricals` method: one of `onehot`, `label`. Prefer "
        "`onehot` for low-cardinality features (<10 unique values). DO NOT "
        "encode the target column. DO NOT encode columns already in "
        "`drop_columns`.\n"
        "- `set_feature_scaling` method: one of `standard`, `minmax`, `robust`.\n\n"
        "## Required ordering\n"
        "- Order: `drop_columns` → `impute_missing` → `encode_categoricals` → "
        "  `train_test_split_csv` → (optional `set_feature_scaling`).\n"
        "- **ALWAYS include `train_test_split_csv`** — the Evaluator needs "
        "held-out data.\n"
        "- Emit EACH op AT MOST ONCE. Do not repeat ops.\n"
        "- **If the dataset is already clean (no NaN, all numeric, no IDs), "
        "just emit `train_test_split_csv` (and optionally "
        "`set_feature_scaling`). Do NOT pad with no-op ops.**\n"
        "- Output 1-5 operations TOTAL. Anything beyond 5 is over-engineering.\n\n"
        "## Worked example output (Titanic-style binary classification)\n"
        "Note how `drop_columns` removes PassengerId (ID), Name (high-card "
        "string), Ticket (high-card), and Cabin (>70% NaN) BEFORE any other op.\n"
        "```json\n"
        "{\n"
        '  "operations": [\n'
        "    {\n"
        '      "name": "drop_columns",\n'
        '      "args": {"columns": ["PassengerId", "Name", "Ticket", "Cabin"]},\n'
        '      "rationale": "PassengerId is an ID; Name/Ticket are high-cardinality strings; Cabin is >70% missing."\n'
        "    },\n"
        "    {\n"
        '      "name": "impute_missing",\n'
        '      "args": {"strategy": "median", "columns": ["Age"]},\n'
        '      "rationale": "Age has missing values; median is robust to outliers."\n'
        "    },\n"
        "    {\n"
        '      "name": "encode_categoricals",\n'
        '      "args": {"method": "onehot", "columns": ["Sex", "Embarked"]},\n'
        '      "rationale": "Two low-cardinality categoricals; one-hot preserves the categorical structure."\n'
        "    },\n"
        "    {\n"
        '      "name": "train_test_split_csv",\n'
        '      "args": {"test_size": 0.2, "stratify_by": "Survived"},\n'
        '      "rationale": "Stratified 80/20 keeps class balance."\n'
        "    },\n"
        "    {\n"
        '      "name": "set_feature_scaling",\n'
        '      "args": {"method": "standard", "columns": ["Age", "Fare", "SibSp", "Parch"]},\n'
        '      "rationale": "Standardize continuous features."\n'
        "    }\n"
        "  ],\n"
        '  "summary": "Drop ID/text/missing columns, impute age, one-hot categoricals, 80/20 stratified split, standardize numerics."\n'
        "}\n"
        "```\n"
    )

    def __init__(self, store, run_id: str) -> None:
        super().__init__(store=store, run_id=run_id)
        # 49B for prep planning — judgment-heavy; the 9B sometimes picks
        # operations outside the supported list (the enum stops that, but the
        # 49B also writes better rationales).
        # 9B nano for the prep plan — enum-locked schema means the model
        # can't drift to unsupported op names, and the 9B is ~3-5× faster
        # than 49B for this kind of short structured output. The 49B used
        # to wedge here generating 300+ lines of redundant ops.
        self.llm = NemotronClient(model=WORKER_MODEL)

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
                max_tokens=2000,  # cap plan size; 5 ops × ~150 tokens each
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
            # Accumulates the dicts returned by record_normalization /
            # record_augmentation / record_feature_scaling. Written to disk at
            # end of run() so the Trainer's generated train.py can read it.
            prep_config: dict[str, Any] = {}

            # Defense in depth — the Preparer LLM sometimes emits the same op
            # multiple times (e.g. `encode_categoricals` 7 times in a row),
            # which compounds disasters (encoding the target column away,
            # blowing up cardinality on ID columns). Dedupe by op name so each
            # operation runs at most once per plan.
            seen_op_names: set[str] = set()

            for op in plan.operations:
                if op.name in seen_op_names:
                    self.emit_event(
                        EventType.WARNING,
                        message=(
                            f"skipping duplicate `{op.name}` "
                            "(already applied earlier in the plan)"
                        ),
                    )
                    continue
                seen_op_names.add(op.name)

                # Target-column protection — strip target from any
                # columns-list arg so we never encode/scale the y vector.
                target_col = dataset_profile.target_column
                if target_col and "columns" in op.args:
                    cols = op.args.get("columns") or []
                    if isinstance(cols, list) and target_col in cols:
                        op.args["columns"] = [c for c in cols if c != target_col]
                        self.emit_event(
                            EventType.WARNING,
                            message=(
                                f"stripped target `{target_col}` from "
                                f"`{op.name}` columns arg (target stays raw)"
                            ),
                        )
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
                        prep_config,
                    )
                    applied_ops.append(
                        f"{op.name}({json.dumps(op.args, default=str)})"
                    )
                    if op.name == "train_test_split_csv":
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

            # Persist accumulated config so the Trainer's train.py can read it.
            prep_config_path: str | None = None
            if prep_config:
                config_path = artifact_dir / "prep_config.json"
                config_path.write_text(
                    json.dumps(prep_config, indent=2, default=str),
                    encoding="utf-8",
                )
                prep_config_path = str(config_path)
                self.emit_event(
                    EventType.INFO,
                    message=(
                        f"wrote prep_config.json with "
                        f"{', '.join(prep_config.keys())}"
                    ),
                    payload={"prep_config_path": prep_config_path},
                )

            report = PreparationReport(
                original_dataset_path=str(source_path),
                prepared_dataset_path=prepared_path,
                operations=applied_ops,
                prep_config_path=prep_config_path,
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
            f"- Modality: `{profile.modality.value}` (tabular CSV)",
            f"- Inferred task: `{profile.task_type.value}`",
            f"- Samples: {profile.n_rows:,}",
            f"- Columns: {profile.n_cols}",
            f"- Target: `{profile.target_column}`",
        ]
        if profile.columns:
            lines.append("- Column details:")
            for c in profile.columns[:20]:
                lines.append(
                    f"  - `{c.name}` ({c.dtype}, missing={c.missing_pct:.1%})"
                )
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
        out_dir = artifact_dir / "split"
        self.emit_event(
            EventType.INFO,
            message="auto-split backstop: LLM omitted split → applying default 80/20",
        )
        # Don't stratify on regression targets — they're continuous so
        # sklearn's stratified split would fail "too few members per class".
        is_regression = profile.task_type.value == "regression"
        stratify_target = None if is_regression else profile.target_column
        try:
            result = prep.split_train_test_csv(
                source_path=current_path,
                test_size=0.2,
                stratify_by=stratify_target,
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

    # ------------------------------------------------------------------
    def _dispatch_op(
        self,
        op: _PrepOperation,
        current_path: Path,
        profile: DatasetProfile,
        artifact_dir: Path,
        prep_config: dict[str, Any],
    ) -> tuple[Path | None, str]:
        """Apply one operation. Returns (new_path | None, human-readable note).

        `new_path` is None for config-only ops (feature_scaling). Those
        mutate `prep_config` in place so the agent can persist the
        accumulated config at end of run().
        """
        name = op.name
        args = op.args

        # LLM frequently emits `column` (singular string) instead of `columns`
        # (plural list). Coerce so downstream filtering sees a list.
        if "column" in args and "columns" not in args:
            single = args.pop("column")
            if isinstance(single, str):
                args["columns"] = [single]
            elif isinstance(single, list):
                args["columns"] = single
            else:
                args["columns"] = []
        # Also normalize `strategy` vs `method` mix-ups for encode op.
        if name == "encode_categoricals" and "strategy" in args and "method" not in args:
            args["method"] = args.pop("strategy")

        # === CSV ops (the only ops AutoForge supports — sklearn-tabular only) ===
        if name == "drop_columns":
            cols_to_drop = list(args.get("columns") or [])
            # Target-column protection happens up-front in run() already,
            # so the target is never in this list. Defense in depth:
            target_col = profile.target_column
            if target_col and target_col in cols_to_drop:
                cols_to_drop = [c for c in cols_to_drop if c != target_col]
            if not cols_to_drop:
                return None, "no columns to drop"
            import pandas as pd
            df = pd.read_csv(current_path)
            existing = [c for c in cols_to_drop if c in df.columns]
            missing = [c for c in cols_to_drop if c not in df.columns]
            if missing:
                self.emit_event(
                    EventType.WARNING,
                    message=(
                        f"drop_columns: skipping non-existent columns "
                        f"{missing!r}"
                    ),
                )
            if not existing:
                return None, "no columns to drop (all missing from CSV)"
            df = df.drop(columns=existing)
            out_path = artifact_dir / (current_path.stem + "_dropped.csv")
            df.to_csv(out_path, index=False)
            return Path(out_path), f"dropped {existing!r}"

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
            # Guard against high-cardinality ID columns and the target.
            # One-hot encoding `customer_id` on a 500-row dataset would
            # explode into 500 columns and trash downstream Trainer code.
            target_col = profile.target_column
            skipped = []
            safe_columns: list[str] = []
            for col in columns:
                if target_col and col == target_col:
                    skipped.append(col)
                    continue
                if any(suffix in col.lower() for suffix in (
                    "_id", "customer_id", "user_id", "id_", "uuid",
                )) or col.lower() == "id":
                    skipped.append(col)
                    continue
                safe_columns.append(col)
            if skipped:
                self.emit_event(
                    EventType.WARNING,
                    message=(
                        f"encode_categoricals skipped high-risk columns "
                        f"{skipped} (target / ID-like / high-cardinality)"
                    ),
                )
            if not safe_columns:
                self.emit_event(
                    EventType.INFO,
                    message=(
                        "encode_categoricals: no safe columns to encode "
                        "after filtering — skipping op"
                    ),
                )
                return None, "no columns to encode (all filtered)"
            out_path = artifact_dir / (current_path.stem + "_encoded.csv")
            result = prep.encode_categoricals_csv(
                source_path=current_path,
                method=method,
                columns=safe_columns,
                output_path=out_path,
            )
            return (
                Path(result["output_path"]),
                f"method={method}, columns={result['encoded_columns']}",
            )

        if name == "train_test_split_csv":
            test_size = float(args.get("test_size", 0.2))
            stratify_by = args.get("stratify_by") or profile.target_column
            # Disable stratification for regression targets (continuous → fails).
            if profile.task_type.value == "regression":
                stratify_by = None
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
            prep_config.update(prep.record_feature_scaling(
                method=method, columns=columns,
            ))
            return None, f"method={method}, columns={columns}"

        # Unreachable: Literal enum guarantees one of the above.
        raise ValueError(f"Unhandled op (enum drift?): {name!r}")
