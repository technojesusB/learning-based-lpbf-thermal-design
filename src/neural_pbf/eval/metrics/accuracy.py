"""Pure accuracy metrics: MAE, max error, per-step aggregation.

All functions are side-effect-free (no I/O, no MLflow logging).
NaN/Inf in predictions are treated as 0-error to avoid metric explosion;
the rollout engine handles divergence detection separately.
"""
from __future__ import annotations

import torch


def _safe_abs(T_pred: torch.Tensor, T_gt: torch.Tensor) -> torch.Tensor:
    diff = (T_pred - T_gt).abs()
    return torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))


def mae_global(T_pred: torch.Tensor, T_gt: torch.Tensor) -> float:
    """Mean absolute error over the full domain [K]."""
    return _safe_abs(T_pred, T_gt).mean().item()


def mae_melt_pool(T_pred: torch.Tensor, T_gt: torch.Tensor, T_solidus: float) -> float:
    """MAE restricted to voxels that are molten in the GT field [K].

    Returns ``float("nan")`` when no voxel in GT exceeds *T_solidus*.
    """
    mask = T_gt > T_solidus
    if not mask.any():
        return float("nan")
    return _safe_abs(T_pred, T_gt)[mask].mean().item()


def max_error(T_pred: torch.Tensor, T_gt: torch.Tensor) -> float:
    """Peak absolute temperature deviation [K]."""
    return _safe_abs(T_pred, T_gt).max().item()


def per_step_mae(
    pred_T: list[torch.Tensor],
    gt_T: list[torch.Tensor],
) -> list[float]:
    """Global MAE for each step in a trajectory."""
    return [mae_global(p, g) for p, g in zip(pred_T, gt_T, strict=False)]
