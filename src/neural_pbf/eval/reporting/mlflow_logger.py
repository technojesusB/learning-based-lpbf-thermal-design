"""Log RolloutResult metrics and artifacts to an active MLflow run."""

from __future__ import annotations

from pathlib import Path

from neural_pbf.eval.rollout.autoregressive import ar_summary
from neural_pbf.eval.rollout.engine import RolloutResult


def log_rollout_to_mlflow(
    result: RolloutResult,
    artifact_dir: Path | str | None = None,
) -> None:
    """Log all metrics from *result* to the currently active MLflow run.

    Logs summary scalars once, per-step scalars as time-series (using MLflow
    ``step``), and optionally uploads a directory of artifact files.

    Args:
        result:       Completed :class:`RolloutResult`.
        artifact_dir: If provided, all files in this directory are uploaded to
                      MLflow under ``eval/<stepper_name>/``.
    """
    import mlflow

    summary = ar_summary(result)
    prefix = f"eval/{result.stepper_name}/{result.mode}"
    n = len(result.latencies_s)

    mlflow.log_metrics(
        {
            f"{prefix}/mean_mae_K": summary.get("mean_mae", float("nan")),
            f"{prefix}/final_mae_K": summary.get("final_mae", float("nan")),
            f"{prefix}/peak_mae_K": summary.get("peak_mae", float("nan")),
            f"{prefix}/vram_peak_MB": result.vram_peak_bytes / 1e6,
            f"{prefix}/mean_latency_ms": 1000.0 * sum(result.latencies_s) / n
            if n > 0
            else float("nan"),
            f"{prefix}/diverged": float(result.diverged_at_step is not None),
        }
    )

    for i, m in enumerate(result.per_step_metrics):
        mlflow.log_metrics(
            {
                f"{prefix}/step/mae_global": m["mae_global"],
                f"{prefix}/step/iou_melt": m["iou_melt"],
                f"{prefix}/step/max_error": m["max_error"],
            },
            step=i,
        )

    if artifact_dir is not None:
        art = Path(artifact_dir)
        if art.exists():
            mlflow.log_artifacts(str(art), artifact_path=f"eval/{result.stepper_name}")
