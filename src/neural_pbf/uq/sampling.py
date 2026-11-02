"""K-draw sampling ensemble through the existing rollout dispatch (model-agnostic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neural_pbf.eval.benchmark.constants import N_EULER_STEPS
from neural_pbf.eval.benchmark.rollout import run_euler_rollout
from neural_pbf.eval.metrics.geometry import melt_pool_extent, melt_pool_volume_vox


def sample_seed(base_seed: int, sample_index: int) -> int:
    """Independent per-input seed, so chunks can be resumed in any order."""
    return int(np.random.SeedSequence([base_seed, sample_index]).generate_state(1)[0])


def sample_k(
    model: nn.Module,
    cond_enc: nn.Module,
    batch: dict[str, Any],
    model_type: str,
    K: int,
    base_seed: int,
    device: torch.device,
    n_steps: int = N_EULER_STEPS,
    grid_attrs: dict[str, Any] | None = None,
    draw_batch: int = 4,
) -> torch.Tensor:
    """Draw *K* independent predictions for one input.

    Args:
        batch: One input (batch dimension 1), as produced by the patch datasets.
        draw_batch: Draws evaluated per forward pass. Smaller is safer on the eGPU.

    Returns:
        ``(K, D, H, W)`` normalised temperature on CPU.

    Raises:
        FloatingPointError: A draw contains NaN or inf.
    """
    if batch["T_in"].shape[0] != 1:
        raise ValueError(f"sample_k expects one input per call, got batch size {batch['T_in'].shape[0]}")
    generator = torch.Generator(device=device).manual_seed(base_seed)
    draws: list[torch.Tensor] = []
    for start in range(0, K, draw_batch):
        n = min(draw_batch, K - start)
        repeated = {
            k: (v.to(device).repeat(n, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor) else v * n)
            for k, v in batch.items()
        }
        pred = run_euler_rollout(
            model, cond_enc, repeated, model_type, device, n_steps=n_steps, grid_attrs=grid_attrs, generator=generator
        )
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"non-finite prediction from model_type={model_type!r} seed={base_seed}")
        draws.append(pred[:, 0].cpu())
    return torch.cat(draws)


def scalar_functionals(T_phys: torch.Tensor, T_l: float) -> tuple[float, float, float]:
    """``(depth_vox, T_max_K, volume_vox)`` of a ``(D, H, W)`` field, using this sample's liquidus."""
    field = T_phys[None, None]
    return melt_pool_extent(field, T_l)["D"], float(T_phys.max().item()), melt_pool_volume_vox(field, T_l)
