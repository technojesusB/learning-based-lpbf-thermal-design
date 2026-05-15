"""Per-sample physical fidelity metrics.

All inputs are normalised temperatures (T_norm). This module applies
the canonical de-normalisation (T_phys = T_norm * T_REF + T_AMBIENT)
and computes IoU / depth / T_max_error / hotspot_offset using the
standardised liquidus threshold T_LIQUIDUS = 1600 K.
"""

from __future__ import annotations

import torch

from neural_pbf.eval.benchmark.constants import T_AMBIENT, T_LIQUIDUS, T_REF
from neural_pbf.eval.metrics.geometry import (
    hotspot_offset_vox,
    iou_melt_volumes,
    melt_pool_extent,
)


def compute_physics_metrics(
    T_pred_norm: torch.Tensor,
    T_tgt_norm: torch.Tensor,
) -> dict[str, float]:
    """Compute physical fidelity metrics from normalised temperature volumes.

    Args:
        T_pred_norm: Predicted temperature, normalised. Any shape accepted;
                     squeezed to 3D (D, H, W) or kept as (1, 1, D, H, W) for
                     geometry helpers that require 5D.
        T_tgt_norm:  Ground-truth temperature, same shape as T_pred_norm.

    Returns:
        Dict with keys: IoU, Depth_GT, Depth_Pred, T_max_Error, Offset_vox.
    """

    def _to_5d(t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze()  # → (D, H, W) or scalar
        if t.ndim == 0:
            t = t.unsqueeze(0)
        while t.ndim < 3:
            t = t.unsqueeze(0)
        if t.ndim == 3:
            t = t.unsqueeze(0).unsqueeze(0)  # → (1, 1, D, H, W)
        elif t.ndim == 4:
            t = t.unsqueeze(0)  # → (1, C, D, H, W)
        return t

    pred_5d = _to_5d(T_pred_norm.detach())
    tgt_5d = _to_5d(T_tgt_norm.detach())

    pred_phys = pred_5d * T_REF + T_AMBIENT
    tgt_phys = tgt_5d * T_REF + T_AMBIENT

    iou = iou_melt_volumes(pred_phys, tgt_phys, T_LIQUIDUS)
    d_pred = melt_pool_extent(pred_phys, T_LIQUIDUS)["D"]
    d_gt = melt_pool_extent(tgt_phys, T_LIQUIDUS)["D"]
    offset = hotspot_offset_vox(pred_5d, tgt_5d)

    t_max_pred = float(pred_phys.max().item())
    t_max_gt = float(tgt_phys.max().item())
    t_max_err = float(abs(t_max_pred - t_max_gt))

    return {
        "IoU": iou,
        "Depth_GT": d_gt,
        "Depth_Pred": d_pred,
        "T_max_GT": t_max_gt,
        "T_max_Pred": t_max_pred,
        "T_max_Error": t_max_err,
        "Offset_vox": offset,
    }
