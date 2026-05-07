"""Melt-pool geometric metrics: extent (W/L/D) and volumetric IoU.

All functions are pure (no I/O, no MLflow).
Extents are returned in *voxels*; multiply by dx/dy/dz for SI metres.
"""
from __future__ import annotations

import torch


def _molten(T: torch.Tensor, T_liquidus: float) -> torch.Tensor:
    return T_liquidus < T


def melt_pool_extent(T: torch.Tensor, T_liquidus: float) -> dict[str, float]:
    """Compute melt-pool bounding-box extents in voxel units.

    Returns a dict with keys ``"W"`` (Y-extent), ``"L"`` (X-extent),
    ``"D"`` (Z-extent).  Returns zeros when no voxel exceeds *T_liquidus*.

    Args:
        T:          Temperature field ``(1, 1, [Nz,] Ny, Nx)`` [K].
        T_liquidus: Liquidus temperature threshold [K].
    """
    mask = _molten(T, T_liquidus)
    if not mask.any():
        return {"W": 0.0, "L": 0.0, "D": 0.0}

    is_3d = T.ndim == 5
    if is_3d:
        m = mask.squeeze(0).squeeze(0)  # (Nz, Ny, Nx)
        W = float(m.any(dim=(0, 2)).sum())  # Y extent
        L = float(m.any(dim=(0, 1)).sum())  # X extent
        D = float(m.any(dim=(1, 2)).sum())  # Z extent
    else:
        m = mask.squeeze(0).squeeze(0)  # (Ny, Nx)
        W = float(m.any(dim=1).sum())  # Y extent
        L = float(m.any(dim=0).sum())  # X extent
        D = 1.0

    return {"W": W, "L": L, "D": D}


def iou_melt_volumes(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    T_liquidus: float,
) -> float:
    """Intersection-over-Union of the two molten volumes (threshold = T_liquidus).

    Returns ``1.0`` when both volumes are empty (perfect agreement).
    """
    pred_mask = _molten(T_pred, T_liquidus)
    gt_mask = _molten(T_gt, T_liquidus)

    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()

    if union.item() == 0:
        return 1.0
    return (intersection / union).item()
