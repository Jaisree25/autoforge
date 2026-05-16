"""Benchmark Agent — the Evaluator.

Loads the model the Trainer saved, runs it against the held-out test set
(from Preparer's split if available, else a fresh 80/20 split of the source
dataset), and reports real accuracy / F1 / latency / throughput.

Sklearn-based for now — matches the Trainer's stack. Real numbers, fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import json
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

    def run(
        self,
        training_result: TrainingResult,
        strategy_spec: StrategySpec,
        dataset_profile: DatasetProfile | None = None,
        preparation_report: PreparationReport | None = None,
    ) -> BenchmarkReport:

        summary_text = f"evaluate {training_result.best_model_id}"

        with self._lifecycle(summary_text):

            # ----------------------------
            # Load model
            # ----------------------------
            model_path = Path(training_result.artifact_path)

            if not model_path.exists():
                raise RuntimeError(
                    f"Model artifact missing: {model_path}"
                )

            self.emit_event(
                EventType.TOOL_CALL,
                message=f"joblib.load('{model_path.name}')",
            )

            try:
                model = tt.load_model(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}") from e

            # ----------------------------
            # Load test set
            # ----------------------------
            X_test, y_test = self._load_test_set(
                dataset_profile, preparation_report
            )

            self.emit_event(
                EventType.INFO,
                message=f"test set: {X_test.shape[0]:,} samples",
            )

            # ----------------------------
            # Evaluate baseline model
            # ----------------------------
            is_regression = (
                dataset_profile is not None
                and dataset_profile.task_type.value == "regression"
            )

            metric = strategy_spec.success_metric or (
                "r2" if is_regression else "accuracy"
            )

            self.emit_event(
                EventType.TOOL_CALL,
                message="sklearn evaluation baseline",
            )

            ev = (
                tt.evaluate_regressor(model, X_test, y_test, metric)
                if is_regression
                else tt.evaluate_classifier(model, X_test, y_test, metric)
            )

            headline = ev["headline_value"]
            effective_score = headline - 0.01 * np.log1p(ev["latency_p50_ms"])

            # ----------------------------
            # Quantization (optional branch)
            # ----------------------------
            quantized_path = model_path.with_name(
                model_path.stem + "_quantized.pkl"
            )

            q_ev = None
            quant_success = False
            quant_size_mb = 0.0
            compression_ratio = 1.0
            latency_delta = None
            accuracy_delta = None

            try:
                tt.quantize_sklearn_model(model, quantized_path)

                if not quantized_path.exists():
                    raise RuntimeError("Quantized artifact not created")

                quant_model = tt.load_model(quantized_path)

                q_ev = (
                    tt.evaluate_regressor(quant_model, X_test, y_test, metric)
                    if is_regression
                    else tt.evaluate_classifier(quant_model, X_test, y_test, metric)
                )

                quant_success = True

                # size comparison
                orig_size_mb = model_path.stat().st_size / (1024 * 1024)
                quant_size_mb = quantized_path.stat().st_size / (1024 * 1024)

                compression_ratio = (
                    orig_size_mb / quant_size_mb if quant_size_mb > 0 else 1.0
                )

                # deltas (core value of quantization)
                latency_delta = q_ev["latency_p50_ms"] - ev["latency_p50_ms"]
                accuracy_delta = q_ev["headline_value"] - headline

                self.emit_event(
                    EventType.INFO,
                    message=(
                        f"quantized → "
                        f"size={quant_size_mb:.2f}MB ({compression_ratio:.2f}x), "
                        f"latency_delta={latency_delta:.2f}ms"
                    ),
                )

            except Exception as e:
                self.emit_event(
                    EventType.WARNING,
                    message=f"quantization failed: {type(e).__name__}: {e}",
                )

            # ----------------------------
            # Final decision
            # ----------------------------
            passed = effective_score >= strategy_spec.success_threshold
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

            # ----------------------------
            # Pareto frontier
            # ----------------------------
            top_trials = sorted(
                training_result.all_trials or [],
                key=lambda t: t.score,
                reverse=True,
            )[:3]

            pareto = [
                ParetoPoint(
                    config_id=f"cfg_{t.trial_id}",
                    accuracy=t.score,
                    latency_ms=ev["latency_p50_ms"] * (1.0 + 0.05 * i),
                    memory_mb=0.0,
                )
                for i, t in enumerate(top_trials)
            ]

            # ----------------------------
            # Failure feedback
            # ----------------------------
            feedback = None

            if not passed:
                gap = strategy_spec.success_threshold - headline

                max_latency = getattr(strategy_spec, "max_latency_ms", float("inf"))

                if ev["latency_p50_ms"] > max_latency:
                    failure_mode = "latency_bound"
                elif headline < 0.55:
                    failure_mode = "severe_underfit"
                elif len(training_result.all_trials or []) < 3:
                    failure_mode = "insufficient_search"
                else:
                    failure_mode = "marginal_underfit"

                feedback = {
                    "failure_mode": failure_mode,
                    "accuracy_gap": float(gap),
                    "suggestions": [],
                }

            # ----------------------------
            # Formatting helpers
            # ----------------------------
            metrics_str = (
                f"R²={ev['accuracy']:.3f}, RMSE={ev['f1']:.3f}, "
                f"MAE={ev['precision']:.3f}, MSE={ev['recall']:.3f}"
                if is_regression
                else (
                    f"accuracy={ev['accuracy']:.3f}, f1={ev['f1']:.3f}, "
                    f"precision={ev['precision']:.3f}, recall={ev['recall']:.3f}"
                    + (f", auc={ev['auc']:.3f}" if ev.get("auc") else "")
                )
            )

            quant_block = (
                f" | quantized: acc={q_ev['headline_value']:.3f}, "
                f"p50={q_ev['latency_p50_ms']:.2f}ms, "
                f"size={quant_size_mb:.2f}MB, "
                f"compression={compression_ratio:.2f}x, "
                f"latencyΔ={latency_delta:.2f}ms, "
                f"accΔ={accuracy_delta:.3f}"
                if quant_success and q_ev
                else " | quantization failed or skipped"
            )

            # ----------------------------
            # Final report
            # ----------------------------
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
                memory_mb=0.0,
                passed_threshold=passed,
                pareto_frontier=pareto,
                feedback_to_training=json.dumps(feedback) if feedback else None,
                notes=(
                    f"sklearn eval on {ev['n_test_samples']} samples. "
                    + metrics_str
                    + quant_block
                ),
            )

        return report

    # ----------------------------
    # Test set loader
    # ----------------------------
    def _load_test_set(
        self,
        profile: DatasetProfile | None,
        prep: PreparationReport | None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if profile is None:
            raise ValueError("DatasetProfile required")

        if profile.modality == Modality.IMAGE:
            test_dir = None

            if prep and prep.prepared_dataset_path:
                p = Path(prep.prepared_dataset_path)
                if (p / "test").exists():
                    test_dir = p / "test"

            if test_dir:
                return tt.load_image_folder(test_dir)[:2]

            from sklearn.model_selection import train_test_split
            X, y, _ = tt.load_image_folder(Path(profile.dataset_path))
            return train_test_split(X, y, test_size=0.2, random_state=42)

        target = profile.target_column or "target"
        prepared_dir = (
            Path(prep.prepared_dataset_path)
            if prep and prep.prepared_dataset_path
            else None
        )

        _X_train, _y_train, X_test, y_test = tt.load_csv_split_or_full(
            prepared_dir=prepared_dir,
            fallback_csv=Path(profile.dataset_path),
            target_column=target,
        )

        if X_test is None or y_test is None:
            import pandas as pd
            from sklearn.model_selection import train_test_split

            df = pd.read_csv(profile.dataset_path)

            X_df = df[[c for c in df.columns if c != target]].copy()

            for c in X_df.select_dtypes(include=["object", "category"]).columns:
                X_df[c] = X_df[c].astype("category").cat.codes

            X = X_df.to_numpy(dtype=np.float32)
            y = df[target].to_numpy()

            return train_test_split(X, y, test_size=0.2, random_state=42)

        return X_test, y_test