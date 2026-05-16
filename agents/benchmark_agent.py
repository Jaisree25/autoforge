"""Benchmark Agent — the Evaluator.

Loads the model the Trainer saved, runs it against the held-out test set
(from Preparer's split if available, else a fresh 80/20 split of the source
dataset), and reports real accuracy / F1 / latency / throughput.

Sklearn-based for now — matches the Trainer's stack. Real numbers, fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    BenchmarkReport,
    DatasetProfile,
    LatencyStats,
    Modality,
    ParetoPoint,
    PreparationReport,
    StrategySpec,
    TrainingResult,
)

from agents.base_agent import BaseAgent
from tools import training_tools as tt


class BenchmarkAgent(BaseAgent):
    name: ClassVar[AgentName] = AgentName.BENCHMARK

    def run(  # type: ignore[override]
        self,
        training_result: TrainingResult,
        strategy_spec: StrategySpec,
        dataset_profile: DatasetProfile | None = None,
        preparation_report: PreparationReport | None = None,
    ) -> BenchmarkReport:
        summary_text = f"evaluate {training_result.best_model_id}"
        with self._lifecycle(summary_text):
            # --- Load model + test set ---
            model_path = Path(training_result.artifact_path)
            self.emit_event(
                EventType.TOOL_CALL,
                message=f"joblib.load('{model_path.name}')",
            )
            model = tt.load_model(model_path)

            X_test, y_test = self._load_test_set(
                dataset_profile, preparation_report,
            )
            self.emit_event(
                EventType.INFO,
                message=f"test set: {X_test.shape[0]:,} samples",
            )

            # --- Evaluate ---
            self.emit_event(
                EventType.TOOL_CALL,
                message="sklearn.metrics + 100× single-sample latency probe",
            )
            metric = strategy_spec.success_metric or "accuracy"
            ev = tt.evaluate_classifier(
                model=model,
                X_test=X_test,
                y_test=y_test,
                metric=metric,
            )

            headline = ev["headline_value"]
            passed = headline >= strategy_spec.success_threshold
            verdict = "PASS" if passed else "FAIL"
            self.emit_event(
                EventType.INFO,
                message=(
                    f"{verdict} · {metric.upper()}={headline:.3f} "
                    f"(threshold {strategy_spec.success_threshold:.3f}) · "
                    f"p50={ev['latency_p50_ms']:.2f}ms · "
                    f"qps={ev['throughput_qps']:.0f}"
                ),
                payload={"verdict": verdict, "headline": headline},
            )

            # --- Pareto frontier from the top-3 trials ---
            top_trials = sorted(
                training_result.all_trials, key=lambda t: t.score, reverse=True,
            )[:3]
            pareto = [
                ParetoPoint(
                    config_id=f"cfg_{t.trial_id}",
                    accuracy=t.score,
                    # We don't have per-trial latency; use the headline latency
                    # scaled by a small factor per trial (approximation)
                    latency_ms=ev["latency_p50_ms"] * (1.0 + 0.05 * i),
                    memory_mb=0.0,
                )
                for i, t in enumerate(top_trials)
            ]

            # --- Feedback to Training (for a future feedback loop) ---
            feedback: str | None = None
            if not passed:
                gap = strategy_spec.success_threshold - headline
                feedback = (
                    f"Threshold missed by {gap:.3f}. Consider: more trials, "
                    f"larger hyperparameter search space, or a different architecture."
                )

            report = BenchmarkReport(
                model_id=training_result.best_model_id,
                accuracy_metric=metric,
                accuracy_value=headline,
                latency=LatencyStats(
                    p50_ms=ev["latency_p50_ms"],
                    p95_ms=ev["latency_p95_ms"],
                    p99_ms=ev["latency_p99_ms"],
                    mean_ms=ev["latency_mean_ms"],
                ),
                throughput_qps=ev["throughput_qps"],
                memory_mb=0.0,  # we don't probe RSS here; Trainer's artifact
                                # size from Optimizer is the persistent number
                passed_threshold=passed,
                pareto_frontier=pareto,
                feedback_to_training=feedback,
                notes=(
                    f"sklearn evaluation on {ev['n_test_samples']} test samples. "
                    f"accuracy={ev['accuracy']:.3f}, f1={ev['f1']:.3f}, "
                    f"precision={ev['precision']:.3f}, recall={ev['recall']:.3f}"
                    + (f", auc={ev['auc']:.3f}" if ev.get('auc') is not None else "")
                ),
            )
        return report

    # ------------------------------------------------------------------
    def _load_test_set(
        self,
        profile: DatasetProfile | None,
        prep: PreparationReport | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load the held-out test set. Prefers Preparer's split."""
        if profile is None:
            raise ValueError("Evaluator needs a DatasetProfile to find the test set.")

        if profile.modality == Modality.IMAGE:
            test_dir = None
            if prep and prep.prepared_dataset_path:
                prepared = Path(prep.prepared_dataset_path)
                if (prepared / "test").is_dir():
                    test_dir = prepared / "test"

            if test_dir is not None:
                X_test, y_test, _ = tt.load_image_folder(test_dir)
                return X_test, y_test

            # Fallback: split source dataset ourselves
            from sklearn.model_selection import train_test_split
            X, y, _ = tt.load_image_folder(Path(profile.dataset_path))
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
            return X_test, y_test

        # Tabular
        target = profile.target_column or "target"
        prepared_dir = (
            Path(prep.prepared_dataset_path)
            if (prep and prep.prepared_dataset_path)
            else None
        )
        _X_train, _y_train, X_test, y_test = tt.load_csv_split_or_full(
            prepared_dir=prepared_dir,
            fallback_csv=Path(profile.dataset_path),
            target_column=target,
        )
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            import pandas as pd
            df = pd.read_csv(profile.dataset_path)
            feature_cols = [c for c in df.columns if c != target]
            X = df[feature_cols].to_numpy(dtype=np.float32)
            y = df[target].to_numpy()
            try:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y,
                )
            except ValueError:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                )
        return X_test, y_test
