"""Strategy Agent — the Researcher.

Real Nemotron-driven research agent. Consumes the `DatasetProfile` from the
Profiler + the user's objective, then:

  1. **Parallel pre-fetch** — kicks off Tavily web search + arXiv search
     concurrently with queries seeded from the dataset profile + task type.
     Old design: multi-turn tool-calling loop (LLM → tool → LLM → …, ~2min
     wall-clock). New design: fire all searches in one thread pool, ~10s.
  2. **Single LLM compose** — Nemotron-49B reads the pre-fetched search
     results and the profile, then emits a structured `StrategySpec` in one
     `/no_think` call (~10-20s). No tool loop, no multi-turn drift.
  3. Writes a Markdown deliverable to `data/artifacts/<run_id>/research.md`.
     This file is the input contract for the Trainer.

Total Researcher wall-clock: ~30s (was ~2min). The CoT chat feed still
shows the model's reasoning — it streams `/think` content for the
composition call when enabled — and the parallel tool calls each emit a
TOOL_CALL event the dashboard renders.
"""
from __future__ import annotations

import concurrent.futures as _futures
import json
from typing import Any, ClassVar

from config import ARTIFACTS_DIR, COORDINATOR_MODEL
from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    Citation,
    DatasetProfile,
    StrategySpec,
)

from agents._llm_client import NemotronClient
from agents.base_agent import BaseAgent
from tools.research_tools import arxiv_search, tavily_search


# How many parallel search calls to fire. Two queries per source × two sources
# = four parallel calls. Each takes ~3-5s, all together ~5-8s in the pool.
_MAX_PARALLEL_SEARCHES = 4
_RESULTS_PER_QUERY = 3


class StrategyAgent(BaseAgent):
    """Researcher — proposes 2-3 candidate architectures with citations."""

    name: ClassVar[AgentName] = AgentName.STRATEGY

    SYSTEM_PROMPT = (
        "You are the Researcher, the second agent in the AutoForge pipeline. "
        "The Profiler has handed you a dataset description; the user has given "
        "you an objective. Your job is to compose a structured `StrategySpec` "
        "with 2-3 candidate architectures the Trainer should consider, backed "
        "by real citations.\n\n"
        "## Inputs you'll receive\n"
        "- The profile (modality, task, sample count, class info).\n"
        "- The user's objective text.\n"
        "- **Pre-fetched search results** — Tavily web hits + arXiv papers — "
        "  selected by parallel queries seeded from the profile. Use these as "
        "  your evidence base. Do NOT call tools yourself; you don't have "
        "  tool access on this turn. The searches already ran.\n\n"
        "## HARD REQUIREMENTS for the output\n"
        "- **candidate_architectures: at least 2 entries.** Always include a "
        "  simple baseline (e.g., logistic regression, MLP) alongside the "
        "  strong recommendation. Variety helps the human pick a fit for the "
        "  budget. Each candidate needs name + family + library + a 1-line "
        "  rationale + hyperparameter_space dict.\n"
        "- **citations: at least 1 arXiv URL** (starts with `https://arxiv.org/`). "
        "  Use the pre-fetched arXiv results — they're real papers.\n"
        "- **success_metric: lowercase canonical form.** Use `f1`, `accuracy`, "
        "  `auc`, `rmse`, `mae`, `mape`. Never `F1_score`, `Accuracy`, etc.\n"
        "- **success_threshold:** parse from the user's objective if explicit "
        "  ('F1 >= 0.85' → 0.85). Otherwise pick a sensible default for the task.\n"
        "- **task_type:** must match what the Profiler determined.\n\n"
        "## Style\n"
        "- Hyperparameter_space values can be lists (for ranges) or single "
        "  values. Keep it small — the Trainer picks one set, not a grid.\n"
        "- Cite the pre-fetched papers by their actual title + URL. Don't "
        "  fabricate references.\n"
        "- Rationale per candidate: ONE sentence on why this fits the dataset."
    )

    # Canonical-form alias map. Defense in depth — schema asks for lowercase
    # but models drift to `F1_score`, `ROC-AUC`, etc.
    _METRIC_ALIASES: ClassVar[dict[str, str]] = {
        "f1_score": "f1", "f1score": "f1", "f1-score": "f1", "f1": "f1",
        "accuracy_score": "accuracy", "acc": "accuracy", "accuracy": "accuracy",
        "roc_auc": "auc", "roc-auc": "auc", "auc_roc": "auc",
        "auc_score": "auc", "auc": "auc",
        "rmse_score": "rmse", "root_mean_squared_error": "rmse", "rmse": "rmse",
        "mae_score": "mae", "mean_absolute_error": "mae", "mae": "mae",
        "mape": "mape",
        "precision_score": "precision", "precision": "precision",
        "recall_score": "recall", "recall": "recall",
    }

    def __init__(self, store, run_id: str) -> None:
        super().__init__(store=store, run_id=run_id)
        # 49B for composition — single call, but it's the StrategySpec which
        # is judgment-heavy; the 9B drops fields ~half the time.
        self.llm = NemotronClient(model=COORDINATOR_MODEL)

    @classmethod
    def _normalize_metric(cls, raw: str) -> str:
        key = raw.strip().lower().replace("-", "_").replace(" ", "_")
        return cls._METRIC_ALIASES.get(key, key)

    # ------------------------------------------------------------------
    def run(  # type: ignore[override]
        self,
        objective: str,
        dataset_profile: DatasetProfile,
    ) -> StrategySpec:
        with self._lifecycle(f"research: '{objective[:60]}'"):
            # --- 1. Parallel pre-fetch ---
            queries = self._build_queries(objective, dataset_profile)
            self.emit_event(
                EventType.INFO,
                message=f"firing {len(queries) * 2} parallel searches "
                        f"(Tavily + arXiv on {len(queries)} queries)",
                payload={"queries": queries},
            )
            tavily_results, arxiv_results = self._parallel_prefetch(queries)

            # --- 2. Compose ---
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"nemotron.compose StrategySpec (model={self.llm.model}, /no_think)",
            )
            user_prompt = self._build_user_prompt(
                objective=objective,
                profile=dataset_profile,
                tavily=tavily_results,
                arxiv=arxiv_results,
            )
            spec: StrategySpec = self.llm.think_and_answer_structured(
                system=self.SYSTEM_PROMPT,
                user=user_prompt,
                schema=StrategySpec,
                on_thinking=lambda p: self.emit_event(
                    EventType.THINKING, message=p,
                ),
                no_think=True,
            )

            # --- 3. Defense in depth ---
            normalized = self._normalize_metric(spec.success_metric)
            if normalized != spec.success_metric:
                self.emit_event(
                    EventType.INFO,
                    message=(
                        f"normalized success_metric "
                        f"`{spec.success_metric}` → `{normalized}`"
                    ),
                )
                spec = spec.model_copy(update={"success_metric": normalized})

            if len(spec.candidate_architectures) < 2:
                self.emit_event(
                    EventType.WARNING,
                    message=(
                        f"LLM returned only {len(spec.candidate_architectures)} "
                        "candidate architecture(s); prompt asked for ≥2"
                    ),
                )

            # arXiv citation backstop — even though we already prefetched
            # arxiv, the LLM might have ignored those citations.
            spec = self._ensure_arxiv_citation(spec, arxiv_results, dataset_profile)

            # --- 4. Write deliverable ---
            md_path = self._write_research_md(
                spec=spec, profile=dataset_profile, objective=objective,
            )
            self.emit_event(
                EventType.INFO,
                message=f"wrote research.md → {md_path}",
                payload={"research_md_path": str(md_path)},
            )
            self.emit_event(
                EventType.INFO,
                message=(
                    f"strategy ready: {len(spec.candidate_architectures)} "
                    f"candidate(s), target {spec.success_metric.upper()} ≥ "
                    f"{spec.success_threshold:.2f}"
                ),
            )

        return spec

    # ------------------------------------------------------------------
    # Query construction — seeded from profile + objective
    # ------------------------------------------------------------------
    def _build_queries(
        self, objective: str, profile: DatasetProfile,
    ) -> list[str]:
        """Build 2 focused search queries to fire on both Tavily + arXiv.

        We deliberately keep it to 2 queries (× 2 sources = 4 calls). More
        and the LLM gets buried in results.
        """
        task = profile.task_type.value.replace("_", " ")
        modality = profile.modality.value

        if modality == "image":
            extra = "deep learning architecture"
            if profile.n_classes:
                extra = f"{profile.n_classes}-class {extra}"
            return [
                f"{task} {modality} {extra}",
                f"sklearn vs pytorch small dataset {task}",
            ]

        if modality == "tabular":
            n_feats = max(profile.n_cols, 0)
            return [
                f"{task} tabular {n_feats} features model comparison",
                f"gradient boosting vs neural network {task}",
            ]

        # Unknown modality fallback
        return [f"{task} {modality}", objective[:80]]

    # ------------------------------------------------------------------
    def _parallel_prefetch(
        self, queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run Tavily + arXiv on each query in parallel. Returns flat lists."""
        tavily_out: list[dict[str, Any]] = []
        arxiv_out: list[dict[str, Any]] = []

        def _wrap(fn, q: str):
            return fn(query=q, max_results=_RESULTS_PER_QUERY)

        with _futures.ThreadPoolExecutor(max_workers=_MAX_PARALLEL_SEARCHES) as pool:
            fut_to_kind: dict[_futures.Future, str] = {}
            for q in queries:
                fut_to_kind[pool.submit(_wrap, tavily_search, q)] = "tavily"
                fut_to_kind[pool.submit(_wrap, arxiv_search, q)] = "arxiv"

            for fut in _futures.as_completed(fut_to_kind):
                kind = fut_to_kind[fut]
                try:
                    result = fut.result(timeout=30)
                except Exception as exc:  # noqa: BLE001
                    self.emit_event(
                        EventType.WARNING,
                        message=f"{kind} search raised: {type(exc).__name__}: {exc}",
                    )
                    continue

                if "error" in result:
                    self.emit_event(
                        EventType.WARNING,
                        message=f"{kind} search failed: {result['error']}",
                    )
                    continue

                rows = result.get("results") or []
                self.emit_event(
                    EventType.TOOL_CALL,
                    message=f"{kind}_search({result.get('query', '')!r}) → {len(rows)} result(s)",
                    payload={"tool": f"{kind}_search", "n_results": len(rows)},
                )
                if kind == "tavily":
                    tavily_out.extend(rows)
                else:
                    arxiv_out.extend(rows)

        # Dedup tavily by URL; dedup arxiv by entry_id
        tavily_out = _dedup_by_key(tavily_out, "url")
        arxiv_out = _dedup_by_key(arxiv_out, "url")
        return tavily_out, arxiv_out

    # ------------------------------------------------------------------
    def _ensure_arxiv_citation(
        self,
        spec: StrategySpec,
        arxiv_results: list[dict[str, Any]],
        profile: DatasetProfile,
    ) -> StrategySpec:
        """If LLM didn't include an arXiv citation, append the top arxiv hit."""
        has_arxiv = any(
            "arxiv.org/" in (c.url or "").lower()
            for c in spec.citations
        )
        if has_arxiv:
            return spec

        if not arxiv_results:
            self.emit_event(
                EventType.WARNING,
                message="no arXiv citation in output and no pre-fetched arXiv results — "
                        "research.md will ship without arXiv citation",
            )
            return spec

        top = arxiv_results[0]
        url = top.get("url", "") or ""
        if url.startswith("http://arxiv.org/"):
            url = "https://" + url[len("http://"):]
        backstop = Citation(
            title=top.get("title", "Untitled"),
            url=url or None,
            source="arxiv",
            snippet=(top.get("abstract") or "")[:200],
        )
        self.emit_event(
            EventType.INFO,
            message=(
                f"arXiv backstop appended: {backstop.title[:60]}"
                + ("…" if len(backstop.title) > 60 else "")
            ),
        )
        return spec.model_copy(update={
            "citations": list(spec.citations) + [backstop]
        })

    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        objective: str,
        profile: DatasetProfile,
        tavily: list[dict[str, Any]],
        arxiv: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"## User objective",
            objective,
            "",
            f"## Dataset (from Profiler)",
            f"- Modality: `{profile.modality.value}`",
            f"- Inferred task: `{profile.task_type.value}`",
            f"- Samples: {profile.n_rows:,}",
        ]
        if profile.modality.value == "tabular":
            lines += [
                f"- Columns: {profile.n_cols}",
                f"- Target column: `{profile.target_column}`",
            ]
        else:
            lines += [
                f"- Classes: {profile.n_classes}",
                f"- Channels: {profile.image_channels}",
                f"- Sample resolutions: {profile.image_resolutions[:3]}",
                f"- Formats: {profile.image_formats}",
            ]
        if profile.class_balance:
            balance_str = ", ".join(
                f"{k}: {v:.0%}" for k, v in profile.class_balance.items()
            )
            lines.append(f"- Class balance: {balance_str}")
        if profile.warnings:
            lines.append("- Profiler warnings:")
            for w in profile.warnings:
                lines.append(f"  - {w}")
        lines += [
            "",
            f"## Profile summary",
            profile.profile_summary or "(no summary)",
            "",
        ]

        # --- Pre-fetched evidence ---
        lines.append("## Pre-fetched arXiv papers (use these for citations)")
        if not arxiv:
            lines.append("(no arXiv results)")
        for i, p in enumerate(arxiv[:6], 1):
            title = p.get("title", "")[:120]
            url = p.get("url", "")
            abstract = (p.get("abstract") or "")[:240]
            lines.append(f"{i}. **{title}** — {url}")
            if abstract:
                lines.append(f"   > {abstract}")
        lines.append("")

        lines.append("## Pre-fetched web pages (Tavily)")
        if not tavily:
            lines.append("(no web results)")
        for i, w in enumerate(tavily[:6], 1):
            title = w.get("title", "")[:120]
            url = w.get("url", "")
            snippet = (w.get("snippet") or "")[:200]
            lines.append(f"{i}. **{title}** — {url}")
            if snippet:
                lines.append(f"   > {snippet}")
        lines.append("")

        lines.append(
            "Compose the final structured StrategySpec now. Include ≥2 "
            "candidate architectures (a strong choice + a simple baseline) "
            "and ≥1 arXiv citation drawn from the pre-fetched papers above."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _write_research_md(
        self,
        spec: StrategySpec,
        profile: DatasetProfile,
        objective: str,
    ) -> str:
        artifact_dir = ARTIFACTS_DIR / self.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "research.md"

        lines: list[str] = []
        lines.append(f"# Research output — run {self.run_id}")
        lines.append("")
        lines.append(f"_Generated by Researcher (Nemotron-Super-49B)._")
        lines.append("")

        lines.append("## Objective")
        lines.append(objective)
        lines.append("")

        lines.append("## Dataset (from Profiler)")
        lines.append(f"- **Modality:** `{profile.modality.value}`")
        lines.append(f"- **Inferred task:** `{profile.task_type.value}`")
        lines.append(f"- **Samples:** {profile.n_rows:,}")
        if profile.modality.value == "tabular":
            lines.append(f"- **Columns:** {profile.n_cols}")
            lines.append(f"- **Target:** `{profile.target_column}`")
        else:
            lines.append(f"- **Classes:** {profile.n_classes}")
            lines.append(f"- **Channels:** {profile.image_channels}")
            lines.append(
                f"- **Formats:** {', '.join(profile.image_formats) or '—'}"
            )
        if profile.class_balance:
            balance_str = ", ".join(
                f"{k}={v:.0%}" for k, v in profile.class_balance.items()
            )
            lines.append(f"- **Class balance:** {balance_str}")
        if profile.warnings:
            lines.append("- **Profiler warnings (for Preparer):**")
            for w in profile.warnings:
                lines.append(f"  - {w}")
        lines.append("")
        if profile.profile_summary:
            lines.append(f"> {profile.profile_summary}")
            lines.append("")

        lines.append("## Success criteria")
        lines.append(f"- **Hard:** `{spec.success_metric}` ≥ "
                     f"{spec.success_threshold:.3f}")
        lines.append(
            f"- **Oracle sanity:** trained model must beat the sklearn "
            f"baseline by ≥5 points (per the agentic-pipeline pattern)."
        )
        lines.append("")

        lines.append(f"## Candidate architectures "
                     f"({len(spec.candidate_architectures)})")
        for i, arch in enumerate(spec.candidate_architectures, 1):
            lines.append(f"")
            lines.append(f"### {i}. `{arch.name}`")
            lines.append(f"- **Family:** {arch.family}")
            lines.append(f"- **Library:** `{arch.library}`")
            if arch.rationale:
                lines.append(f"- **Rationale:** {arch.rationale}")
            if arch.hyperparameter_space:
                lines.append("- **Hyperparameter space:**")
                lines.append("  ```json")
                for line in json.dumps(
                    arch.hyperparameter_space, indent=2,
                ).splitlines():
                    lines.append(f"  {line}")
                lines.append("  ```")

        lines.append("")
        if spec.research_summary:
            lines.append("## Research summary")
            lines.append(spec.research_summary)
            lines.append("")

        if spec.citations:
            lines.append(f"## Citations ({len(spec.citations)})")
            for i, c in enumerate(spec.citations, 1):
                title_md = f"[{c.title}]({c.url})" if c.url else c.title
                lines.append(f"{i}. {title_md}")
                if c.source:
                    lines.append(f"   _source: `{c.source}`_")
                if c.snippet:
                    lines.append(f"   > {c.snippet[:200]}")
            lines.append("")

        lines.append("---")
        lines.append(
            "_This document is the input contract for the Trainer agent. "
            "When the Trainer is built (mirroring the user's `agentic-pipeline` "
            "pattern), it reads this file as its `requirements.md` equivalent._"
        )

        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)


# ---------------------------------------------------------------------------
def _dedup_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """First-wins dedup by `row[key]`. Preserves order. Empty keys pass through."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        v = r.get(key) or ""
        if v and v in seen:
            continue
        if v:
            seen.add(v)
        out.append(r)
    return out
