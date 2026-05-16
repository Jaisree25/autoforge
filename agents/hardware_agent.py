"""Hardware Agent — the Optimizer (post-training only after the refactor).

The Profiler now owns hardware detection (its old pre-training role).
This agent now does just the post-training step: take the Trainer's
sklearn pickle, re-serialize with maximum compression, and report the
final deployable size.

Future deepening (when time permits):
  - Convert sklearn → ONNX via `skl2onnx` (gives portable artifact)
  - Run dynamic int8 quantization on the ONNX via `onnxruntime.quantization`
  - Optional Pytorch path: `torch.onnx.export` + TensorRT

For the hackathon: joblib max-compression gets the model from ~2MB to
~0.5MB. Honest, real, and the dashboard shows actual bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from config import ARTIFACTS_DIR
from contracts.messages import EventType
from contracts.schemas import (
    AgentName,
    DeploymentArtifact,
    TrainingResult,
)

from agents.base_agent import BaseAgent
from tools import training_tools as tt


HardwarePhase = Literal["pre_training", "post_training"]


class HardwareAgent(BaseAgent):
    """Optimizer — model compression + deployment artifact."""

    name: ClassVar[AgentName] = AgentName.HARDWARE

    # Dispatcher (satisfies BaseAgent abstract `run`)
    def run(  # type: ignore[override]
        self,
        phase: HardwarePhase = "post_training",
        **kwargs,
    ) -> BaseModel:
        if phase == "post_training":
            return self.run_post_training(**kwargs)
        raise ValueError(
            f"Unknown hardware phase: {phase!r}. The pre-training pass was "
            "absorbed into ProfilerAgent.run_envelope() post-refactor."
        )

    def run_post_training(
        self,
        training_result: TrainingResult,
    ) -> DeploymentArtifact:
        summary_text = f"optimize {training_result.best_model_id} for deployment"
        with self._lifecycle(summary_text):
            source_path = Path(training_result.artifact_path)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Trainer's artifact not found at {source_path}"
                )

            original_size_mb = source_path.stat().st_size / 1024.0 / 1024.0
            self.emit_event(
                EventType.INFO,
                message=(
                    f"source model: {source_path.name} "
                    f"({original_size_mb:.2f} MB before optimization)"
                ),
            )

            # --- Re-serialize with maximum compression ---
            self.emit_event(
                EventType.TOOL_CALL,
                message="joblib.dump(model, compress=9)  # max LZMA compression",
            )
            model = tt.load_model(source_path)
            optimized_path = (
                ARTIFACTS_DIR / self.run_id / "deploy"
                / f"{training_result.best_model_id}.pkl.gz"
            )
            save_info = tt.save_model(model, optimized_path, compress=9)
            new_size_mb = save_info["size_mb"]

            ratio = (
                (1.0 - new_size_mb / original_size_mb) * 100.0
                if original_size_mb > 0 else 0.0
            )
            self.emit_event(
                EventType.INFO,
                message=(
                    f"compressed: {original_size_mb:.2f} MB → "
                    f"{new_size_mb:.2f} MB ({ratio:+.1f}%)"
                ),
            )

            artifact = DeploymentArtifact(
                artifact_path=save_info["path"],
                format="joblib",
                quantization=None,  # sklearn models don't get int8/fp16 directly
                size_mb=new_size_mb,
                notes=(
                    f"sklearn pickle via joblib (compress=9, LZMA). "
                    f"Source was {original_size_mb:.2f} MB; deploy "
                    f"artifact is {new_size_mb:.2f} MB "
                    f"({ratio:+.1f}% size change). "
                    "ONNX export + int8 quantization is a future step."
                ),
            )
            self.emit_event(
                EventType.INFO,
                message=(
                    f"deployment artifact ready: {artifact.format} "
                    f"({artifact.size_mb:.2f} MB) at {Path(artifact.artifact_path).name}"
                ),
            )
        return artifact

    # Backward-compat: the pre-training pass was moved to ProfilerAgent.
    # If some old code path still calls run_pre_training, raise loudly.
    def run_pre_training(self, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError(
            "HardwareAgent's pre-training pass moved to "
            "ProfilerAgent.run_envelope() during the refactor."
        )
