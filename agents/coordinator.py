"""Coordinator — orchestrates six agents with a HITL gate after each.

New pipeline order (post-refactor):

    Profiler    → [gate] →
    Researcher  → [gate] →
    Preparer    → [gate] →
    Trainer     → [gate] →
    Evaluator   → [gate] →
    Optimizer   → (done)

The Profiler runs first and emits TWO outputs — DatasetProfile + Training-
Envelope — each persisted as its own store row but bundled into a single
gate so the human approves "observation complete, hand to Researcher" once.

Phase 4 / wave 2 still uses hardcoded sequencing. The agentic-coordinator
rewrite (LLM-driven decision points + feedback loops) is deliberately
deferred per the user's chosen path.
"""
from __future__ import annotations

import uuid
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel

from contracts.messages import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    EventType,
)
from contracts.schemas import (
    AgentName,
    BenchmarkReport,
    DatasetProfile,
    DeploymentArtifact,
    PipelineRun,
    PipelineStatus,
    PreparationReport,
    StrategySpec,
    TrainingEnvelope,
    TrainingResult,
)
from memory.store import MemoryStore

from agents.base_agent import BaseAgent
from agents.benchmark_agent import BenchmarkAgent
from agents.dataset_agent import DatasetAgent
from agents.hardware_agent import HardwareAgent
from agents.profiler_agent import ProfilerAgent
from agents.strategy_agent import StrategyAgent
from agents.training_agent import TrainingAgent


# Display names — kept in sync with dashboard.agent_identity but duplicated
# here so Coordinator doesn't depend on the dashboard package (it runs in a
# non-Streamlit subprocess).
_AGENT_DISPLAY_NAME: dict[AgentName, str] = {
    AgentName.PROFILER:    "Profiler",
    AgentName.STRATEGY:    "Researcher",
    AgentName.DATASET:     "Data Preparer",
    AgentName.HARDWARE:    "Optimizer",
    AgentName.TRAINING:    "Trainer",
    AgentName.BENCHMARK:   "Evaluator",
    AgentName.COORDINATOR: "Director",
}


def _display(agent: AgentName) -> str:
    return _AGENT_DISPLAY_NAME.get(agent, agent.value)


class HITLService(Protocol):
    def request_and_wait(
        self, request: ApprovalRequest, timeout: float = ...,
    ) -> ApprovalResponse: ...

    def notify(self, run_id: str, message: str) -> None: ...


class PipelineRejected(Exception):
    """Raised when a human rejects an approval gate."""


# ---------------------------------------------------------------------------
# Summarizers — one short sentence per output type
# ---------------------------------------------------------------------------
def _summarize(output: BaseModel) -> str:
    if isinstance(output, DatasetProfile):
        balance = ""
        if output.class_balance:
            balance = " · class balance " + ", ".join(
                f"{k}: {v:.0%}" for k, v in output.class_balance.items()
            )
        return (
            f"Observed {output.n_rows:,} rows × {output.n_cols} columns. "
            f"Target `{output.target_column}` · task `{output.task_type.value}`{balance}."
        )
    if isinstance(output, TrainingEnvelope):
        return (
            f"Hardware envelope: {output.gpu_name or '—'} "
            f"({output.gpu_memory_gb or 0:.0f}GB). "
            f"Max {output.max_trials} trials in ≤{output.max_train_minutes:.1f}min."
        )
    if isinstance(output, StrategySpec):
        archs = ", ".join(f"`{a.name}`" for a in output.candidate_architectures)
        return (
            f"Recommended {len(output.candidate_architectures)} architecture(s): "
            f"{archs}. Target {output.success_metric.upper()} ≥ "
            f"{output.success_threshold:.2f}."
        )
    if isinstance(output, PreparationReport):
        return (
            f"Applied {len(output.operations)} operation(s). "
            f"Prepared dataset: `{output.prepared_dataset_path or '—'}`."
        )
    if isinstance(output, TrainingResult):
        return (
            f"Best {output.metric_name.upper()}: **{output.best_score:.3f}** from "
            f"{output.trials_completed}/{output.total_trials} trials. "
            f"Library: `{output.library}` · model `{output.best_model_id}`."
        )
    if isinstance(output, BenchmarkReport):
        verdict = "PASS" if output.passed_threshold else "FAIL"
        return (
            f"**{verdict}** · {output.accuracy_metric.upper()}="
            f"{output.accuracy_value:.3f} · "
            f"p50={output.latency.p50_ms:.1f}ms · "
            f"throughput={output.throughput_qps:.0f}QPS."
        )
    if isinstance(output, DeploymentArtifact):
        return (
            f"Exported {output.format.upper()} "
            f"({output.quantization or 'no quant'}, {output.size_mb:.1f}MB)."
        )
    return str(output)[:200]


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class Coordinator(BaseAgent):
    """Orchestrates the six agents end-to-end with a gate after each."""

    name = AgentName.COORDINATOR

    def __init__(
        self,
        store: MemoryStore,
        hitl: HITLService,
        run_id: str | None = None,
    ) -> None:
        run_id = run_id or str(uuid.uuid4())
        super().__init__(store=store, run_id=run_id)
        self.hitl = hitl

    def run(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("Use Coordinator.execute() instead")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def execute(self, dataset_path: str, objective: str) -> PipelineRun:
        self.store.create_run(self.run_id, objective, dataset_path)
        self.store.update_run_status(self.run_id, PipelineStatus.RUNNING)
        self.emit_event(
            EventType.STARTED,
            message=f"pipeline start: {objective}",
            payload={"dataset_path": dataset_path},
        )

        try:
            # --- Stage 1: Profiler (produces DatasetProfile + Envelope) ---
            profile, envelope = self._run_profiler(dataset_path, objective)
            # Gate combines profile + envelope info into one summary
            combined_summary = (
                _summarize(profile)
                + "\n\n_Envelope:_ " + _summarize(envelope)
            )
            self._gate_custom(
                AgentName.PROFILER, profile,
                _display(AgentName.STRATEGY),
                summary_override=combined_summary,
            )

            # --- Stage 2: Researcher ---
            spec = self._run_strategy(objective, profile)
            # Specialized gate: human PICKS one of the proposed candidates.
            # Trims spec.candidate_architectures down to the chosen one so
            # the Preparer + Trainer see a single committed direction.
            spec = self._gate_candidate_pick(spec)

            # --- Stage 3: Data Preparer ---
            prep = self._run_preparer(profile, spec)
            self._gate(
                AgentName.DATASET, prep, _display(AgentName.TRAINING),
            )

            # --- Stage 4: Trainer ---
            training = self._run_training(spec, envelope, profile, prep)
            self._gate(
                AgentName.TRAINING, training, _display(AgentName.BENCHMARK),
            )

            # --- Stage 5: Evaluator ---
            benchmark = self._run_benchmark(training, spec, profile, prep)
            self._gate(
                AgentName.BENCHMARK, benchmark, _display(AgentName.HARDWARE),
            )

            # --- Stage 6: Optimizer (post-training only, no gate after) ---
            self._run_optimizer(training)

            self.store.update_run_status(self.run_id, PipelineStatus.COMPLETED)
            self.emit_event(
                EventType.COMPLETED,
                message=(
                    f"pipeline complete: "
                    f"{benchmark.accuracy_metric}={benchmark.accuracy_value:.3f}, "
                    f"passed={benchmark.passed_threshold}"
                ),
            )
            self.hitl.notify(
                self.run_id,
                f"AutoForge run complete — {benchmark.accuracy_metric}="
                f"{benchmark.accuracy_value:.3f} "
                f"({'PASS' if benchmark.passed_threshold else 'FAIL'})",
            )

        except PipelineRejected as exc:
            self.store.update_run_status(
                self.run_id, PipelineStatus.CANCELLED, error=str(exc),
            )
            self.emit_event(EventType.WARNING, message=f"pipeline cancelled: {exc}")
            self.hitl.notify(self.run_id, f"Run cancelled by reviewer: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline failed")
            self.store.update_run_status(
                self.run_id, PipelineStatus.FAILED, error=str(exc),
            )
            self.emit_event(
                EventType.ERROR,
                message=f"pipeline failed: {type(exc).__name__}: {exc}",
            )
            self.hitl.notify(self.run_id, f"Run FAILED: {exc}")
            raise

        result = self.store.get_run(self.run_id)
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------
    def _run_profiler(
        self, dataset_path: str, objective: str,
    ) -> tuple[DatasetProfile, TrainingEnvelope]:
        agent = ProfilerAgent(self.store, self.run_id)
        profile = agent.run(dataset_path=dataset_path, objective=objective)
        self.store.save_agent_output(
            self.run_id, AgentName.PROFILER, "dataset_profile", profile,
        )
        envelope = agent.run_envelope()
        self.store.save_agent_output(
            self.run_id, AgentName.PROFILER, "training_envelope", envelope,
        )
        return profile, envelope

    def _run_strategy(self, objective: str, profile: DatasetProfile) -> StrategySpec:
        agent = StrategyAgent(self.store, self.run_id)
        spec = agent.run(objective=objective, dataset_profile=profile)
        self.store.save_agent_output(
            self.run_id, AgentName.STRATEGY, "strategy_spec", spec,
        )
        return spec

    def _run_preparer(
        self, profile: DatasetProfile, spec: StrategySpec,
    ) -> PreparationReport:
        agent = DatasetAgent(self.store, self.run_id)
        prep = agent.run(dataset_profile=profile, strategy_spec=spec)
        self.store.save_agent_output(
            self.run_id, AgentName.DATASET, "preparation_report", prep,
        )
        return prep

    def _run_training(
        self,
        spec: StrategySpec,
        envelope: TrainingEnvelope,
        profile: DatasetProfile,
        prep: PreparationReport,
    ) -> TrainingResult:
        # Trainer runs its own internal HITL gate for design.md, so we hand it
        # the HITL service. The post-training "Trainer → Evaluator" gate still
        # fires from the Coordinator below.
        agent = TrainingAgent(self.store, self.run_id, hitl=self.hitl)
        result = agent.run(
            strategy_spec=spec,
            training_envelope=envelope,
            dataset_profile=profile,
            preparation_report=prep,
        )
        self.store.save_agent_output(
            self.run_id, AgentName.TRAINING, "training_result", result,
        )
        return result

    def _run_benchmark(
        self,
        training: TrainingResult,
        spec: StrategySpec,
        profile: DatasetProfile,
        prep: PreparationReport,
    ) -> BenchmarkReport:
        agent = BenchmarkAgent(self.store, self.run_id)
        report = agent.run(
            training_result=training,
            strategy_spec=spec,
            dataset_profile=profile,
            preparation_report=prep,
        )
        self.store.save_agent_output(
            self.run_id, AgentName.BENCHMARK, "benchmark_report", report,
        )
        return report

    def _run_optimizer(self, training: TrainingResult) -> DeploymentArtifact:
        agent = HardwareAgent(self.store, self.run_id)
        artifact = agent.run_post_training(training_result=training)
        self.store.save_agent_output(
            self.run_id, AgentName.HARDWARE, "deployment_artifact", artifact,
        )
        return artifact

    # ------------------------------------------------------------------
    # Specialized HITL gate: pick one candidate architecture
    # ------------------------------------------------------------------
    def _gate_candidate_pick(self, spec: StrategySpec) -> StrategySpec:
        """Show the human a radio-button list of Researcher candidates.

        Returns the StrategySpec trimmed to the chosen candidate (single entry
        in `candidate_architectures`). If the human rejects, raises
        PipelineRejected. If they approve without picking (or the payload is
        malformed), defaults to candidate #0.
        """
        from_agent = AgentName.STRATEGY
        from_display = _display(from_agent)
        next_display = _display(AgentName.DATASET)

        candidates = [
            {
                "index": i,
                "name": arch.name,
                "family": arch.family,
                "library": arch.library,
                "rationale": arch.rationale,
                "hyperparameter_space": arch.hyperparameter_space,
            }
            for i, arch in enumerate(spec.candidate_architectures)
        ]

        # Build a one-line summary listing all candidates.
        if candidates:
            names = " · ".join(f"`{c['name']}`" for c in candidates)
            summary = (
                f"Researcher proposed {len(candidates)} candidate(s): {names}. "
                f"Target {spec.success_metric.upper()} ≥ {spec.success_threshold:.2f}. "
                f"Pick one to proceed."
            )
        else:
            summary = "Researcher returned no candidates."

        request = ApprovalRequest(
            run_id=self.run_id,
            agent=from_agent,
            title=f"{from_display}: pick a candidate → {next_display}",
            description=summary,
            payload={
                "kind": "candidate_pick",  # dashboard switches to radio UI
                "summary": summary,
                "next_agent": next_display,
                "candidates": candidates,
                "default_index": 0,
                # Full spec dump available for "show details" expander.
                "agent_output": spec.model_dump(mode="json"),
            },
        )
        self.store.update_run_status(self.run_id, PipelineStatus.AWAITING_APPROVAL)
        self.emit_event(
            EventType.APPROVAL_REQUESTED,
            message=f"Handoff: {from_display} → {next_display} (pick a candidate)",
            payload={
                "summary": summary,
                "next_agent": next_display,
                "from_agent": from_agent.value,
                "request_id": request.request_id,
            },
        )
        response = self.hitl.request_and_wait(request)

        if response.decision is ApprovalDecision.REJECTED:
            self.emit_event(
                EventType.APPROVAL_RECEIVED,
                message=f"rejected by {response.responder or 'unknown'}"
                + (f" — {response.comment}" if response.comment else ""),
                payload={"request_id": response.request_id, "decision": "rejected"},
            )
            raise PipelineRejected(
                response.comment or "Researcher candidates rejected"
            )

        # Find the chosen index. Default to 0 if nothing came back.
        chosen_idx = 0
        if response.response_payload:
            raw = response.response_payload.get("selected_index", 0)
            try:
                chosen_idx = int(raw)
            except (TypeError, ValueError):
                chosen_idx = 0
        chosen_idx = max(0, min(chosen_idx, len(spec.candidate_architectures) - 1))

        chosen = spec.candidate_architectures[chosen_idx]
        self.emit_event(
            EventType.APPROVAL_RECEIVED,
            message=(
                f"approved by {response.responder or 'unknown'} — "
                f"picked candidate #{chosen_idx + 1}: `{chosen.name}`"
            ),
            payload={
                "request_id": response.request_id,
                "decision": "approved",
                "selected_index": chosen_idx,
                "selected_name": chosen.name,
            },
        )

        # Trim to the chosen candidate so Preparer + Trainer commit to one.
        trimmed = spec.model_copy(update={
            "candidate_architectures": [chosen],
        })
        self.store.save_agent_output(
            self.run_id, from_agent, "strategy_spec", trimmed,
        )
        self.store.update_run_status(self.run_id, PipelineStatus.RUNNING)
        self.emit_event(
            EventType.INFO,
            message=f"committed to `{chosen.name}` ({chosen.family} / `{chosen.library}`)",
        )
        return trimmed

    # ------------------------------------------------------------------
    # Generic HITL gate (used by every other stage)
    # ------------------------------------------------------------------
    def _gate(
        self,
        from_agent: AgentName,
        output: BaseModel,
        next_agent_display: str,
        edit_model: type[BaseModel] | None = None,
    ) -> BaseModel | None:
        return self._gate_custom(
            from_agent=from_agent,
            output=output,
            next_agent_display=next_agent_display,
            summary_override=None,
            edit_model=edit_model,
        )

    def _gate_custom(
        self,
        from_agent: AgentName,
        output: BaseModel,
        next_agent_display: str,
        summary_override: str | None = None,
        edit_model: type[BaseModel] | None = None,
    ) -> BaseModel | None:
        """Block until the human approves. `summary_override` lets stages
        with multi-output handoffs (e.g. Profiler) supply a richer summary
        than `_summarize(output)` produces alone."""
        summary = summary_override or _summarize(output)
        from_display = _display(from_agent)
        request = ApprovalRequest(
            run_id=self.run_id,
            agent=from_agent,
            title=f"{from_display} → {next_agent_display}",
            description=summary,
            payload={
                "summary": summary,
                "next_agent": next_agent_display,
                "agent_output": output.model_dump(mode="json"),
            },
        )
        self.store.update_run_status(self.run_id, PipelineStatus.AWAITING_APPROVAL)
        self.emit_event(
            EventType.APPROVAL_REQUESTED,
            message=f"Handoff: {from_display} → {next_agent_display}",
            payload={
                "summary": summary,
                "next_agent": next_agent_display,
                "from_agent": from_agent.value,
                "request_id": request.request_id,
            },
        )

        response = self.hitl.request_and_wait(request)

        self.emit_event(
            EventType.APPROVAL_RECEIVED,
            message=(
                f"{response.decision.value} by {response.responder or 'unknown'}"
                + (f" — {response.comment}" if response.comment else "")
            ),
            payload={
                "request_id": response.request_id,
                "decision": response.decision.value,
            },
        )

        if response.decision is ApprovalDecision.REJECTED:
            raise PipelineRejected(
                response.comment or f"{from_display} output rejected"
            )

        self.store.update_run_status(self.run_id, PipelineStatus.RUNNING)

        if (
            edit_model is not None
            and response.decision is ApprovalDecision.EDITED
            and response.response_payload is not None
        ):
            raw = response.response_payload.get(
                "agent_output", response.response_payload,
            )
            try:
                edited = edit_model.model_validate(raw)
                # Only Strategy currently supports semantic edits.
                self.store.save_agent_output(
                    self.run_id, from_agent, "strategy_spec", edited,
                )
                self.emit_event(
                    EventType.INFO,
                    message=f"{from_display} output edited by reviewer — using edited version",
                )
                return edited
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to apply edit: {}", exc)

        return None
