"""Melt-pool geometric metrics: extent (W/L/D), volumetric IoU, and hotspot offset.

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


def hotspot_offset_vox(T_pred: torch.Tensor, T_tgt: torch.Tensor) -> float:
    """L2 distance between the peak-temperature voxel in *T_pred* and *T_tgt*.

    Operates on the first channel and averages across the batch dimension.
    Works for both 2D ``(B, C, Ny, Nx)`` and 3D ``(B, C, Nz, Ny, Nx)`` inputs.

    Returns:
        Mean L2 offset in voxel units across the batch.
    """
    B = T_pred.shape[0]
    spatial_shape = T_pred.shape[2:]  # skip B, C

    sp = T_pred[:, 0].reshape(B, -1)
    st = T_tgt[:, 0].reshape(B, -1)

    total = 0.0
    for b in range(B):
        ip = int(sp[b].argmax().item())
        it = int(st[b].argmax().item())
        # Unravel flat index → per-axis voxel coordinates (last axis first)
        cp: list[int] = []
        ct: list[int] = []
        for dim in reversed(spatial_shape):
            cp.append(ip % dim)
            ct.append(it % dim)
            ip //= dim
            it //= dim
        total += float(sum((a - c) ** 2 for a, c in zip(cp, ct, strict=False)) ** 0.5)
    return total / B


def evaluate_physical_metrics(
    T_pred: torch.Tensor,
    T_tgt: torch.Tensor,
    T_liquidus: float,
) -> dict[str, float]:
    """Compute the standard physical-fidelity triad for a single prediction/GT pair.

    Intended for the final test phase only (rollout-based T_pred required).
    Expects batch size 1; ``melt_pool_extent`` squeezes batch and channel dims.

    Args:
        T_pred:     Predicted temperature field ``(1, 1, [Nz,] Ny, Nx)``.
        T_tgt:      Ground-truth field, same shape.
        T_liquidus: Liquidus threshold in the same normalisation as the fields.

    Returns:
        Dict with keys ``Physical/Meltpool_IoU``, ``Physical/Depth_Error_Vox``,
        and ``Physical/Hotspot_Offset_Vox``.

    Raises:
        ValueError: If ``T_pred`` batch size is not 1 (``melt_pool_extent``
                    squeezes batch and channel dims and silently corrupts results
                    for B > 1).
    """
    if T_pred.shape[0] != 1:
        raise ValueError(
            f"evaluate_physical_metrics expects batch size 1, got {T_pred.shape[0]}. "
            "Call per-sample inside your test loop."
        )
    return {
        "Physical/Meltpool_IoU": iou_melt_volumes(T_pred, T_tgt, T_liquidus),
        "Physical/Depth_Error_Vox": abs(
            melt_pool_extent(T_pred, T_liquidus)["D"]
            - melt_pool_extent(T_tgt, T_liquidus)["D"]
        ),
        "Physical/Hotspot_Offset_Vox": hotspot_offset_vox(T_pred, T_tgt),
    }
