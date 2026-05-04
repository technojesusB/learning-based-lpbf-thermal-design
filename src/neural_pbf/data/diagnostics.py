"""Per-step timing and diagnostics for the offline dataset generator."""

from __future__ import annotations

import csv
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_CSV_FIELDS = [
    "step_idx",
    "t_sim",
    "t_transfer",
    "t_io",
    "substeps",
    "peak_temperature",
    "is_sample",
]


@dataclass(frozen=True)
class StepTiming:
    """Immutable record of timing and physics metrics for one exposure step."""

    step_idx: int
    t_sim: float
    t_transfer: float
    t_io: float
    substeps: int
    peak_temperature: float
    is_sample: bool


class StepProfiler:
    """Measures wall-clock time for named phases of one exposure step."""

    def __init__(self, sync_cuda: bool = False) -> None:
        self._sync_cuda = sync_cuda
        self._timings: dict[str, float] = {}

    def _sync(self) -> None:
        if self._sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time the enclosed block and store the duration under *name*."""
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._timings[name] = time.perf_counter() - t0

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the recorded duration for *name*, or *default* if not measured."""
        return self._timings.get(name, default)

    def build(
        self,
        *,
        step_idx: int,
        substeps: int,
        peak_temperature: float,
        is_sample: bool,
    ) -> StepTiming:
        """Assemble a StepTiming from measured phases and provided metadata."""
        return StepTiming(
            step_idx=step_idx,
            t_sim=self.get("sim"),
            t_transfer=self.get("transfer"),
            t_io=self.get("io"),
            substeps=substeps,
            peak_temperature=peak_temperature,
            is_sample=is_sample,
        )


def format_diag_line(timing: StepTiming) -> str:
    """Render a one-line diagnostic summary for *timing*.

    Example output:
        Step 54 | Sim: 1.2s | Xfer: 0.8s | IO: 2.1s | Steps: 140 | Tmax: 1850K
    """
    return (
        f"Step {timing.step_idx} | "
        f"Sim: {timing.t_sim:.1f}s | "
        f"Xfer: {timing.t_transfer:.1f}s | "
        f"IO: {timing.t_io:.1f}s | "
        f"Steps: {timing.substeps} | "
        f"Tmax: {timing.peak_temperature:.0f}K"
    )


class DiagnosticsCsvWriter:
    """Append-mode CSV sidecar that records one row per exposure step."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._initialized = path.exists() and path.stat().st_size > 0

    def write(self, timing: StepTiming) -> None:
        """Append *timing* as one row; writes the header on the very first call."""
        mode = "a" if self._initialized else "w"
        with open(self._path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(
                {
                    "step_idx": timing.step_idx,
                    "t_sim": timing.t_sim,
                    "t_transfer": timing.t_transfer,
                    "t_io": timing.t_io,
                    "substeps": timing.substeps,
                    "peak_temperature": timing.peak_temperature,
                    "is_sample": timing.is_sample,
                }
            )


class MLflowStepLogger:
    """Logs StepTiming metrics to the active MLflow run. Silent no-op when disabled."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def log(self, timing: StepTiming, global_step: int) -> None:
        """Log per-step metrics at *global_step*. Silently absorbs all exceptions."""
        if not self._enabled:
            return
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "t_sim": timing.t_sim,
                    "t_transfer": timing.t_transfer,
                    "t_io": timing.t_io,
                    "substeps": float(timing.substeps),
                    "peak_temperature": timing.peak_temperature,
                },
                step=global_step,
            )
        except Exception:
            pass
