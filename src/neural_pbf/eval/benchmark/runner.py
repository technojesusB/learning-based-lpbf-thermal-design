"""Pure physics sweep runner — no plotting, no MLflow side effects.

Public API:
    run_physics_sweep  -- evaluate a list of (name, model, cond_enc, type) tuples
                          over a dataset, returning metrics + collected volumes
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from neural_pbf.eval.benchmark.constants import N_EULER_STEPS
from neural_pbf.eval.benchmark.metrics_physics import compute_physics_metrics
from neural_pbf.eval.benchmark.rollout import run_euler_rollout

logger = logging.getLogger(__name__)

# Type alias: (name, model, cond_enc, model_type, optional grid_attrs)
ModelEntry = tuple[str, nn.Module, nn.Module, str] | tuple[str, nn.Module, nn.Module, str, dict[str, Any]]


def run_physics_sweep(
    models: list[ModelEntry],
    dataset: Dataset,
    device: torch.device,
    test_indices: list[int] | None = None,
    collect_indices: list[int] | None = None,
    n_steps: int = N_EULER_STEPS,
) -> tuple[pd.DataFrame, dict[str, dict[int, torch.Tensor]], dict[str, dict[str, float]]]:
    """Evaluate models over *test_indices* of *dataset*, computing physics metrics.

    Pure function — does NOT write files, does NOT call MLflow.

    Args:
        models:         List of (name, model, cond_enc, model_type) or
                        (name, model, cond_enc, model_type, grid_attrs).
        dataset:        Indexed dataset compatible with the model_type.
        device:         Inference device.
        test_indices:   Sample indices to evaluate. Defaults to last 10%.
        collect_indices: Indices whose predicted volumes are stored and returned.
        n_steps:        Euler integration steps.

    Returns:
        (df, all_vols, throughput)
        - df:          Per-sample metrics DataFrame.
        - all_vols:    {model_name: {sample_idx: T_pred_tensor}}.
        - throughput:  {model_name: {"s_per_sample": float, ...}}.
    """
    if test_indices is None:
        n = len(dataset)  # type: ignore[arg-type]
        test_indices = list(range(int(n * 0.9), n))

    all_rows: list[dict[str, Any]] = []
    all_vols: dict[str, dict[int, torch.Tensor]] = {}
    throughput: dict[str, dict[str, float]] = {}

    for entry in models:
        name, model, cond_enc, model_type = entry[0], entry[1], entry[2], entry[3]
        grid_attrs: dict[str, Any] | None = entry[4] if len(entry) > 4 else None  # type: ignore[arg-type]

        model = model.to(device)
        cond_enc = cond_enc.to(device)
        model.eval()
        cond_enc.eval()

        rows: list[dict[str, Any]] = []
        vols: dict[int, torch.Tensor] = {}

        # Measure throughput on the first sample
        t0 = time.perf_counter()
        n_timed = 0

        for i in tqdm(test_indices, desc=f"Sweep {name}", leave=False):
            try:
                sample = dataset[i]
                batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                T_pred = run_euler_rollout(
                    model,
                    cond_enc,
                    batch,
                    model_type,
                    device,
                    n_steps=n_steps,
                    grid_attrs=grid_attrs,
                )
                T_tgt = batch["T_target"].squeeze(1)

                if collect_indices and i in collect_indices:
                    vols[i] = T_pred.squeeze().cpu()

                metrics = compute_physics_metrics(T_pred, T_tgt)
                metrics["Model"] = name  # type: ignore[assignment]
                metrics["Sample"] = f"s{i:04d}"  # type: ignore[assignment]
                rows.append(metrics)
                n_timed += 1

            except Exception as exc:
                logger.warning("Error on sample %d for model %s: %s", i, name, exc)

        elapsed = time.perf_counter() - t0
        s_per = elapsed / n_timed if n_timed > 0 else float("nan")
        throughput[name] = {
            "s_per_sample": s_per,
            "throughput_samples_per_sec": 1.0 / s_per if s_per > 0 else float("inf"),
            "n_samples": n_timed,
        }

        all_rows.extend(rows)
        all_vols[name] = vols

        del model, cond_enc
        torch.cuda.empty_cache()

    _COLS = [
        "Model",
        "Sample",
        "IoU",
        "Depth_GT",
        "Depth_Pred",
        "T_max_GT",
        "T_max_Pred",
        "T_max_Error",
        "Offset_vox",
    ]
    df = (
        pd.DataFrame(all_rows, columns=_COLS)  # type: ignore[call-overload]
        if all_rows
        else pd.DataFrame(columns=_COLS)  # type: ignore[call-overload]
    )
    return df, all_vols, throughput
