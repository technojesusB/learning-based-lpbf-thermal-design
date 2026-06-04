"""GPU telemetry, step-timing, and profiler helpers for active MLflow runs."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def log_gpu_telemetry(step: int) -> None:
    """Log GPU temperature, VRAM usage, and fan speed to the active MLflow run.

    Queries device 0 via pynvml. Silent no-op on any failure (no GPU,
    no driver, pynvml not initialised, etc.). Calls nvmlShutdown after each
    query to respect the pynvml lifecycle contract.

    Args:
        step: Current training step (epoch) for the MLflow time-series.
    """
    try:
        import mlflow
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        finally:
            pynvml.nvmlShutdown()
        mlflow.log_metrics(
            {
                "GPU/Temp_C": float(temp),
                "GPU/VRAM_MB": mem.used / 1e6,
                "GPU/Fan_Pct": float(fan),
            },
            step=step,
        )
    except Exception:
        logger.debug("GPU telemetry unavailable", exc_info=True)


def log_step_timing(step: int, secs_per_iter: float) -> None:
    """Log training throughput to the active MLflow run.

    Args:
        step:          Current training step (epoch) for the MLflow time-series.
        secs_per_iter: Average wall-clock seconds per training iteration.
    """
    try:
        import mlflow

        mlflow.log_metric("Perf/steps_per_sec", 1.0 / max(secs_per_iter, 1e-9), step=step)
    except Exception:
        logger.debug("Step timing log unavailable", exc_info=True)


@contextmanager
def epoch0_profiler(loader_len: int, run_name: str) -> Generator:
    """Context manager that wraps epoch-0 training in a PyTorch profiler.

    Yields the profiler so callers can call ``profiler.step()`` inside the
    training batch loop.  The profiler is always stopped (even if an exception
    is raised inside the ``with`` block).  The Chrome trace is exported and
    logged to MLflow **only on clean exit** — if an exception propagates out
    of the ``with`` block, the trace is discarded silently.

    Args:
        loader_len: Number of batches in the training loader (used to warn
                    when the profiler schedule cannot collect any data).
        run_name:   Stable run identifier used as the MLflow artifact
                    sub-directory (e.g. ``"v3_rope_patches_4"``).

    Example::

        if epoch == 0:
            with epoch0_profiler(len(train_loader), _RUN_NAME) as prof:
                _run_train_epoch(..., profiler=prof)
        else:
            _run_train_epoch(...)
    """
    import torch
    from torch.profiler import ProfilerActivity

    if loader_len < 5:
        logger.warning(
            "Profiler schedule requires >=5 batches per epoch but epoch 0 has %d. The exported trace will be empty.",
            loader_len,
        )

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        with_stack=False,
    )
    prof.start()
    try:
        yield prof
    finally:
        prof.stop()

    # Export trace — only reached on clean exit from the with block.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        trace_path = tmp.name
    try:
        import mlflow

        prof.export_chrome_trace(trace_path)
        mlflow.log_artifact(trace_path, artifact_path=f"profiler/{run_name}")
        logger.info("Profiler trace logged to MLflow (epoch 0).")
    except Exception:
        logger.debug("Failed to export profiler trace", exc_info=True)
    finally:
        if os.path.exists(trace_path):
            os.remove(trace_path)
