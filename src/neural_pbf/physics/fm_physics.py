"""Physics helpers for Flow Matching LPBF surrogate training."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from neural_pbf.physics.ops import div_k_grad


def denorm_cond_batch(
    cond_normalized: Tensor,
    ds_cfg: Any,
    key: str,
) -> Tensor:
    """De-normalise one conditioning key from z-score to physical units.

    Args:
        cond_normalized: (B, D_cond) z-score normalised conditioning tensor.
        ds_cfg:          Dataset config with conditioning_keys, cond_means, cond_stds.
        key:             Conditioning key (e.g. "rho", "cp", "k_s").

    Returns:
        (B, 1, 1, 1, 1) float32 in physical units, broadcast-ready for 5-D fields.
    """
    idx = list(ds_cfg.conditioning_keys).index(key)
    mean: float = ds_cfg.cond_means[key]
    std: float = ds_cfg.cond_stds[key]
    vals = cond_normalized[:, idx] * std + mean  # (B,)
    return vals.view(-1, 1, 1, 1, 1)


def physics_heat_residual(
    v_pred: Tensor,
    x_tau_detached: Tensor,
    tau: Tensor,
    T_in: Tensor,
    Q: Tensor,
    rho: Tensor,
    cp: Tensor,
    k: Tensor,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    ds_cfg: Any,
    dt_s: float = 5e-6,
) -> Tensor:
    """Heat-equation PDE residual in SI units.

    Reconstructs T_pred = x_tau + (1-tau)*v_pred (OT-FM identity), de-normalises
    to SI [K], and evaluates ρ·cp·(T_pred - T_in)/dt_s ≈ ∇·(k·∇T_pred) + Q.

    x_tau_detached must be detached before calling so gradients flow only
    through v_pred (avoids expensive second-order backprop).

    Args:
        v_pred:           (B, 1, D, H, W) predicted velocity, with grad.
        x_tau_detached:   (B, 1, D, H, W) interpolated noisy field, detached.
        tau:              (B,) flow time ∈ [0, 1].
        T_in:             (B, 1, D, H, W) normalised initial temperature.
        Q:                (B, 1, D, H, W) normalised heat source.
        rho:              (B, 1, 1, 1, 1) density [kg/m³].
        cp:               (B, 1, 1, 1, 1) specific heat [J/(kg·K)].
        k:                (B, 1, 1, 1, 1) conductivity [W/(m·K)].
        dx_m, dy_m, dz_m: Grid spacings [m].
        ds_cfg:           Dataset config carrying T_ref, T_ambient, Q_ref.
        dt_s:             Physical time step for finite-difference dT/dt [s].

    Returns:
        Scalar MSE residual (SI units: [W/m³]², scaled by Q_ref²).
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")

    T_pred_norm = x_tau_detached + (1.0 - tau.view(-1, 1, 1, 1, 1)) * v_pred

    T_pred_SI = T_pred_norm * ds_cfg.T_ref + ds_cfg.T_ambient
    T_in_SI = T_in * ds_cfg.T_ref + ds_cfg.T_ambient
    Q_SI = Q * ds_cfg.Q_ref

    dT_dt_SI = (T_pred_SI - T_in_SI) / dt_s

    lhs = rho * cp * dT_dt_SI
    k_field = k.expand_as(T_pred_SI)
    rhs = div_k_grad(T_pred_SI, k_field, dx_m, dy_m, dz_m) + Q_SI

    # Scale by Q_ref to prevent float overflow when squaring (values ~1e15 W/m³)
    lhs_scaled = lhs / ds_cfg.Q_ref
    rhs_scaled = rhs / ds_cfg.Q_ref

    return F.mse_loss(lhs_scaled, rhs_scaled)
