"""Fetch training statistics from MLflow runs (pure — no plotting)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from mlflow.client import MlflowClient

from neural_pbf.eval.benchmark.constants import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

_SCHEMA = [
    "Version",
    "Duration [h]",
    "s/sample",
    "GPU Util [%]",
    "GPU Mem [MB]",
    "Final Loss",
    "Start_Time",
    "End_Time",
    "Inf GPU Util [%]",
    "Inf GPU Mem [MB]",
]


def fetch_mlflow_data(
    run_ids: dict[str, str],
    tracking_uri: str = MLFLOW_TRACKING_URI,
    bench_run_id: str | None = None,
) -> pd.DataFrame:
    """Fetch training stats for each run ID and return a tidy DataFrame.

    Args:
        run_ids:      Mapping of version label → MLflow training run ID.
        tracking_uri: MLflow tracking server URI.
        bench_run_id: Optional ID of the current benchmark run (for nested inf stats).

    Returns:
        DataFrame with columns: Version, Duration [h], s/sample,
        GPU Util [%], GPU Mem [MB], Final Loss, Start_Time, End_Time,
        Inf GPU Util [%], Inf GPU Mem [MB].
    """
    if not run_ids:
        return pd.DataFrame(columns=_SCHEMA)  # type: ignore[call-overload]

    client = MlflowClient(tracking_uri=tracking_uri)
    rows: list[dict[str, Any]] = []

    # Resolve benchmark experiment ID once if needed
    bench_exp_id = None
    if bench_run_id:
        try:
            bench_exp_id = client.get_run(bench_run_id).info.experiment_id
        except Exception as exc:
            logger.warning("Could not resolve benchmark experiment ID: %s", exc)

    for version, run_id in run_ids.items():
        row: dict[str, Any] = {k: float("nan") for k in _SCHEMA}
        row["Version"] = version
        try:
            # 1. Fetch Training Stats
            run = client.get_run(run_id)
            start_ms = run.info.start_time or 0
            end_ms = run.info.end_time

            loss_hist = client.get_metric_history(run_id, "train_loss")
            if end_ms is None and loss_hist:
                end_ms = loss_hist[-1].timestamp
                logger.info(
                    "Interrupted run detected for %s; using last metric timestamp.",
                    version,
                )

            eff_end_ms = end_ms or start_ms
            row["Duration [h]"] = round((eff_end_ms - start_ms) / 3_600_000.0, 3)
            row["Start_Time"] = start_ms
            row["End_Time"] = eff_end_ms

            util_hist = client.get_metric_history(run_id, "system/gpu_0_utilization_percentage")
            mem_hist = client.get_metric_history(run_id, "system/gpu_0_memory_usage_megabytes")

            if util_hist:
                row["GPU Util [%]"] = round(float(np.mean([m.value for m in util_hist])), 1)
            if mem_hist:
                row["GPU Mem [MB]"] = round(float(np.mean([m.value for m in mem_hist])), 1)

            if len(loss_hist) > 1:
                total_ms = loss_hist[-1].timestamp - loss_hist[0].timestamp
                total_steps = loss_hist[-1].step - loss_hist[0].step
                if total_steps > 0:
                    s_per_it = (total_ms / 1000.0) / total_steps
                    batch_size = int(run.data.params.get("batch_size", 4))
                    # s/sample = (seconds_per_batch / batch_size)
                    row["s/sample"] = round(s_per_it / batch_size, 4)

            row["Final Loss"] = run.data.metrics.get("val_loss", run.data.metrics.get("train_loss", float("nan")))

            # 2. Fetch Inference Stats (from the BENCHMARK experiment)
            if bench_exp_id and bench_run_id:
                nested_runs = client.search_runs(
                    experiment_ids=[bench_exp_id],
                    filter_string=f"tags.mlflow.parentRunId = '{bench_run_id}'",
                )
                for nr in nested_runs:
                    if nr.data.tags.get("model_version") == version:
                        i_util = client.get_metric_history(nr.info.run_id, "system/gpu_0_utilization_percentage")
                        i_mem = client.get_metric_history(nr.info.run_id, "system/gpu_0_memory_usage_megabytes")
                        if i_util:
                            row["Inf GPU Util [%]"] = round(float(np.mean([m.value for m in i_util])), 1)
                        if i_mem:
                            row["Inf GPU Mem [MB]"] = round(float(np.max([m.value for m in i_mem])), 1)

        except Exception as exc:
            logger.warning("Could not fetch MLflow data for %s: %s", version, exc)

        rows.append(row)

    return pd.DataFrame(rows, columns=_SCHEMA)  # type: ignore[call-overload]
