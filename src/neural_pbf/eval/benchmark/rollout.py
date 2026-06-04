"""Euler-integration rollout for all model types.

Public API:
    run_euler_rollout         -- single-batch rollout, all model types
    measure_inference_throughput -- time n warm-up + timed passes, return s/sample
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from neural_pbf.eval.benchmark.constants import N_EULER_STEPS

logger = logging.getLogger(__name__)


def run_euler_rollout(
    model: nn.Module,
    cond_enc: nn.Module,
    batch: dict[str, Any],
    model_type: str,
    device: torch.device,
    n_steps: int = N_EULER_STEPS,
    grid_attrs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run Euler integration to produce T_pred for one batch.

    Args:
        model:       Velocity / DiT model.
        cond_enc:    Conditioning encoder.
        batch:       Dict with tensors on *device*.
        model_type:  ``"net"``, ``"dit"``, ``"rope"``, or ``"triton"``.
        device:      Compute device.
        n_steps:     Euler steps (default: N_EULER_STEPS = 25).
        grid_attrs:  Physical grid spacing — required for rope / triton;
                     raises ValueError if None for those types.

    Returns:
        T_pred tensor, same shape as batch["T_in"].
    """
    if model_type in ("rope", "triton"):
        if grid_attrs is None:
            raise ValueError(
                f"grid_attrs must be provided for model_type={model_type!r}. "
                "Pass e.g. grid_attrs={'dx_m': 1.5e-5, 'dy_m': 1.5e-5, 'dz_m': 1.5e-5}."
            )
        from neural_pbf.integrator.fm_stepper import euler_rollout_rope

        ps = getattr(model, "patch_size", 4)
        return euler_rollout_rope(
            model,  # type: ignore[arg-type]
            cond_enc,
            batch,
            n_steps,
            device,
            grid_attrs,
            ps,
        )

    # --- net / dit path ---
    with torch.no_grad():
        T_in = batch["T_in"]
        if T_in.ndim > 5:
            T_in = T_in.squeeze(2)
        mask = batch["mask"]
        if mask.ndim > 5:
            mask = mask.squeeze(2)
        Q = batch["Q"]
        if Q.ndim > 5:
            Q = Q.squeeze(2)

        cond = batch["conditioning"]
        cond_emb = cond_enc(cond)

        B = T_in.shape[0]
        x = torch.randn_like(T_in)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=device)
            inp = torch.cat([x, mask, Q], dim=1)
            v = model(inp, tau, cond_emb)
            x = x + v * dt

    return x


def measure_inference_throughput(
    model: nn.Module,
    cond_enc: nn.Module,
    batch: dict[str, Any],
    model_type: str,
    device: torch.device,
    n_warmup: int = 2,
    n_timed: int = 5,
    grid_attrs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Measure inference latency in seconds-per-sample.

    Returns:
        Dict with keys: s_per_sample, throughput_samples_per_sec.
    """
    for _ in range(n_warmup):
        run_euler_rollout(model, cond_enc, batch, model_type, device, grid_attrs=grid_attrs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_timed):
        run_euler_rollout(model, cond_enc, batch, model_type, device, grid_attrs=grid_attrs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    s_per_sample = elapsed / n_timed

    return {
        "s_per_sample": s_per_sample,
        "throughput_samples_per_sec": 1.0 / s_per_sample if s_per_sample > 0 else float("inf"),
    }
