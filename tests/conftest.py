"""Shared pytest fixtures.

The skeleton tests exercise Coordinator behavior (sequencing, gates,
rejection) — they shouldn't care whether agents make real LLM calls.
Real ProfilerAgent + StrategyAgent now hit the Nemotron API and would
burn ~2-3 minutes + thousands of tokens per pytest run.

The autouse fixtures below monkey-patch both agents' `run` methods to
return deterministic stubs without file I/O or network. Tests stay fast
and offline; the real implementations are exercised via:
  scripts/_smoke_profiler.py
  scripts/_smoke_researcher.py
"""
from __future__ import annotations

import pytest

from contracts.messages import EventType
from contracts.schemas import (
    BenchmarkReport,
    CandidateArchitecture,
    Citation,
    ColumnProfile,
    DatasetProfile,
    DeploymentArtifact,
    LatencyStats,
    Modality,
    ParetoPoint,
    PreparationReport,
    StrategySpec,
    TaskType,
    TrainingEnvelope,
    TrainingResult,
    TrialResult,
)


@pytest.fixture(autouse=True)
def _stub_profiler(monkeypatch):
    """Replace ProfilerAgent's LLM-backed methods with deterministic stubs."""
    from agents.profiler_agent import ProfilerAgent

    def fake_run(self, dataset_path: str, objective: str) -> DatasetProfile:
        with self._lifecycle(f"observe {dataset_path} [test stub]"):
            self.emit_event(
                EventType.INFO,
                message="test stub: skipping pandas/PIL + LLM",
            )
            return DatasetProfile(
                dataset_path=dataset_path,
                modality=Modality.TABULAR,
                n_rows=1000,
                n_cols=8,
                columns=[
                    ColumnProfile(name="customer_id", dtype="int64", missing_pct=0.0),
                    ColumnProfile(name="age", dtype="int64", missing_pct=0.01),
                    ColumnProfile(name="churn", dtype="int64", missing_pct=0.0),
                ],
                target_column="churn",
                task_type=TaskType.BINARY_CLASSIFICATION,
                class_balance={"0": 0.85, "1": 0.15},
                warnings=["test stub warning"],
                profile_summary="Test-stub dataset profile.",
            )

    def fake_envelope(self) -> TrainingEnvelope:
        with self._lifecycle("envelope [test stub]"):
            return TrainingEnvelope(
                gpu_available=False,
                gpu_name="test-cpu",
                gpu_memory_gb=0.0,
                cpu_count=4,
                system_memory_gb=8.0,
                max_train_minutes=1.0,
                max_trials=5,
                batch_size_range=(16, 64),
                allowed_libraries=["xgboost", "sklearn"],
                notes="test-stub envelope",
            )

    monkeypatch.setattr(ProfilerAgent, "run", fake_run)
    monkeypatch.setattr(ProfilerAgent, "run_envelope", fake_envelope)


@pytest.fixture(autouse=True)
def _stub_researcher(monkeypatch):
    """Replace StrategyAgent's LLM + tools with a deterministic stub."""
    from agents.strategy_agent import StrategyAgent

    def fake_run(self, objective: str, dataset_profile: DatasetProfile) -> StrategySpec:
        with self._lifecycle(f"research: '{objective[:60]}' [test stub]"):
            self.emit_event(
                EventType.INFO,
                message="test stub: skipping Nemotron + Tavily + arXiv",
            )
            return StrategySpec(
                objective=objective,
                task_type=dataset_profile.task_type,
                success_metric="f1",
                success_threshold=0.85,
                candidate_architectures=[
                    CandidateArchitecture(
                        name="xgboost-stub",
                        family="gradient_boost",
                        library="xgboost",
                        hyperparameter_space={
                            "max_depth": [3, 6, 9],
                            "learning_rate": [0.01, 0.1],
                        },
                        rationale="test-stub rationale",
                    ),
                    CandidateArchitecture(
                        name="logreg-baseline-stub",
                        family="linear",
                        library="sklearn",
                        hyperparameter_space={"C": [0.1, 1.0]},
                        rationale="simple baseline stub",
                    ),
                ],
                research_summary="Test-stub research summary.",
                citations=[
                    Citation(title="Test paper", url="https://arxiv.org/abs/0000.0000",
                             source="test", snippet=""),
                ],
            )

    monkeypatch.setattr(StrategyAgent, "run", fake_run)


@pytest.fixture(autouse=True)
def _stub_preparer(monkeypatch):
    """Replace DatasetAgent's LLM-driven planner + dispatch with a stub."""
    from agents.dataset_agent import DatasetAgent

    def fake_run(self, dataset_profile, strategy_spec) -> PreparationReport:
        with self._lifecycle(f"prepare {dataset_profile.dataset_path} [test stub]"):
            self.emit_event(
                EventType.INFO,
                message="test stub: skipping Nemotron + tool dispatch",
            )
            return PreparationReport(
                original_dataset_path=dataset_profile.dataset_path,
                prepared_dataset_path=None,
                operations=[
                    "impute_missing(strategy='median', columns=['x'])",
                    "train_test_split_csv(test_size=0.2)",
                ],
                summary="Test-stub preparation report.",
                notes="Stub — no real preparation applied.",
            )

    monkeypatch.setattr(DatasetAgent, "run", fake_run)


@pytest.fixture(autouse=True)
def _stub_trainer(monkeypatch):
    """Replace TrainingAgent.run with a fast deterministic stub (no Optuna, no fit)."""
    from agents.training_agent import TrainingAgent

    def fake_run(
        self, strategy_spec, training_envelope, dataset_profile,
        preparation_report=None, previous_feedback=None, attempt_num=1,
    ) -> TrainingResult:
        with self._lifecycle("HPO [test stub]"):
            self.emit_event(EventType.INFO, message="test stub: skipping Optuna + fit")
            return TrainingResult(
                best_model_id="stub_model",
                metric_name="accuracy",
                best_score=0.96,
                best_params={"hidden_layer_sizes": [64]},
                trials_completed=3,
                total_trials=3,
                training_time_seconds=0.1,
                artifact_path="data/artifacts/stub_model.pkl",
                library="sklearn",
                all_trials=[
                    TrialResult(trial_id=i, params={}, score=0.90 + 0.02 * i,
                                duration_seconds=0.01, status="completed")
                    for i in range(3)
                ],
                notes="stub",
            )

    monkeypatch.setattr(TrainingAgent, "run", fake_run)


@pytest.fixture(autouse=True)
def _stub_evaluator(monkeypatch):
    """Replace BenchmarkAgent.run with a deterministic stub."""
    from agents.benchmark_agent import BenchmarkAgent

    def fake_run(
        self, training_result, strategy_spec,
        dataset_profile=None, preparation_report=None,
    ) -> BenchmarkReport:
        with self._lifecycle("benchmark [test stub]"):
            self.emit_event(EventType.INFO, message="test stub: skipping load+eval")
            return BenchmarkReport(
                model_id=training_result.best_model_id,
                accuracy_metric=strategy_spec.success_metric,
                accuracy_value=training_result.best_score,
                latency=LatencyStats(p50_ms=1.0, p95_ms=2.0, p99_ms=3.0, mean_ms=1.2),
                throughput_qps=500.0,
                memory_mb=0.0,
                passed_threshold=training_result.best_score >= strategy_spec.success_threshold,
                pareto_frontier=[
                    ParetoPoint(config_id="cfg_0", accuracy=training_result.best_score,
                                latency_ms=1.0)
                ],
                feedback_to_training=None,
                notes="stub",
            )

    monkeypatch.setattr(BenchmarkAgent, "run", fake_run)


@pytest.fixture(autouse=True)
def _stub_optimizer(monkeypatch):
    """Replace HardwareAgent.run_post_training with a deterministic stub."""
    from agents.hardware_agent import HardwareAgent

    def fake_post(self, training_result) -> DeploymentArtifact:
        with self._lifecycle("optimize [test stub]"):
            self.emit_event(EventType.INFO, message="test stub: skipping compression")
            return DeploymentArtifact(
                artifact_path="data/artifacts/stub_deploy.pkl.gz",
                format="joblib",
                quantization=None,
                size_mb=0.5,
                notes="stub",
            )

    monkeypatch.setattr(HardwareAgent, "run_post_training", fake_post)
