"""Physics helpers for Flow Matching LPBF surrogate training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff, sigmoid_step
from neural_pbf.physics.material_vector import exposure_time_of, material_from_vector
from neural_pbf.physics.ops import div_k_grad

_STORAGE = ("enthalpy", "cp_in")
_DIFFUSION = ("explicit", "implicit", "trapezoid")


def _sensible_enthalpy(T: Tensor, mat: MaterialConfig) -> Tensor:
    """Integral of the sensible cp(T) used by cp_eff, up to a constant [J/kg]."""
    if not mat.use_T_dep:
        return mat.cp_base * T
    if not (mat.use_lut and mat.T_lut is not None and mat.cp_lut is not None):
        dT = T - mat.T_ref
        return mat.cp_base * (dT + 0.5 * mat.cp_T_coeff * dT**2)
    t = torch.tensor(mat.T_lut, dtype=T.dtype, device=T.device)
    v = torch.tensor(mat.cp_lut, dtype=T.dtype, device=T.device)
    # Exact integral of the piecewise-linear LUT; cp is held constant outside it, as cp_eff does.
    H_nodes = F.pad(torch.cumsum(0.5 * (v[1:] + v[:-1]) * t.diff(), 0), (1, 0))
    i = (torch.searchsorted(t, T.clamp(t[0], t[-1]).contiguous(), right=True) - 1).clamp(0, len(t) - 2)
    s = T.clamp(t[0], t[-1]) - t[i]
    slope = (v[i + 1] - v[i]) / (t[i + 1] - t[i])
    inside = H_nodes[i] + v[i] * s + 0.5 * slope * s**2
    return inside + v[0] * (T - t[0]).clamp(max=0.0) + v[-1] * (T - t[-1]).clamp(min=0.0)


def enthalpy(T: Tensor, mat: MaterialConfig) -> Tensor:
    """Specific enthalpy H(T) with dH/dT == cp_eff(T), up to a constant [J/kg]."""
    H = _sensible_enthalpy(T, mat)
    if mat.latent_heat_L <= 1e-9:
        return H
    width = mat.T_liquidus - mat.T_solidus  # same mid and width as cp_eff, so dH/dT matches it exactly
    width = width if width >= 1e-6 else 1.0
    s = (T - 0.5 * (mat.T_solidus + mat.T_liquidus)) / (0.5 * width)
    return H + mat.latent_heat_L * sigmoid_step(s, mat.transition_sharpness)


def _div_flux(T: Tensor, mask: Tensor, mat: MaterialConfig, dx: float, dy: float, dz: float) -> Tensor:
    solid = (mask.to(torch.bool) | (mat.T_solidus <= T)).to(T.dtype)  # the Triton kernel promotes per substep
    return div_k_grad(T, k_eff(T, mat, solid), dx, dy, dz)


def heat_residual_si(
    T_in: Tensor,
    T_out: Tensor,
    Q: Tensor,
    mask_in: Tensor,
    mat: MaterialConfig,
    dt_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    storage: str = "enthalpy",
    diffusion: str = "trapezoid",
    mask_out: Tensor | None = None,
) -> Tensor:
    """Residual of ρ·dH/dt = ∇·(k(T)∇T) + Q over one step of length dt_s, in W/m³.

    Uses the solver's discretisation (harmonic face k, k_eff with powder, cp_eff with latent heat).
    One voxel is dropped at every face, because a cropped patch has artificial zero-flux faces.
    The solver's linear cooling term (SimulationConfig.loss_h) is left out; the dataset generator keeps it at 0.

    Args:
        T_in, T_out: (B, 1, D, H, W) temperatures [K] before and after the step.
        Q:           (B, 1, D, H, W) volumetric source [W/m³], constant over the step.
        mask_in:     (B, 1, D, H, W) 0 = powder, 1 = consolidated, before the step.
        storage:     "enthalpy" (ρ·ΔH/dt, exact across melting) or "cp_in" (ρ·cp_eff(T_in)·ΔT/dt,
                     the solver's own explicit form).
        diffusion:   where ∇·k∇T is evaluated: "explicit" (T_in), "implicit" (T_out), or "trapezoid".
        mask_out:    mask for the T_out term; defaults to mask_in (promotion above T_solidus applies anyway).

    Returns:
        (B, 1, D-2, H-2, W-2) residual field [W/m³].
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")
    if storage not in _STORAGE:
        raise ValueError(f"storage must be one of {_STORAGE}, got {storage!r}")
    if diffusion not in _DIFFUSION:
        raise ValueError(f"diffusion must be one of {_DIFFUSION}, got {diffusion!r}")
    if not (T_in.shape == T_out.shape == Q.shape == mask_in.shape) or T_in.ndim != 5:
        raise ValueError(f"field shapes differ or are not 5-D: {T_in.shape}, {T_out.shape}, {Q.shape}, {mask_in.shape}")

    if storage == "enthalpy":
        lhs = mat.rho * (enthalpy(T_out, mat) - enthalpy(T_in, mat)) / dt_s
    else:
        lhs = mat.rho * cp_eff(T_in, mat) * (T_out - T_in) / dt_s

    explicit = _div_flux(T_in, mask_in, mat, dx_m, dy_m, dz_m) if diffusion != "implicit" else 0.0
    m_out = mask_in if mask_out is None else mask_out
    implicit = _div_flux(T_out, m_out, mat, dx_m, dy_m, dz_m) if diffusion != "explicit" else 0.0
    weight = 0.5 if diffusion == "trapezoid" else 1.0
    R = lhs - weight * (explicit + implicit) - Q
    return R[..., 1:-1, 1:-1, 1:-1]


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


PHYSICS_FORMS = ("coupled", "tin")


def physics_loss_reference(
    T_pred: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor, material: Tensor, form: str,
    dx_m: float, dy_m: float, dz_m: float,
) -> Tensor:  # fmt: skip
    """Physics loss in plain PyTorch: Σ R² / Σ (ρ·(H(T_target) − H(T_in))/dt)² over the interiors of the batch.

    Both forms use enthalpy storage. ``coupled`` evaluates ∇·(k∇T) at the prediction, ``tin`` at T_in. Each
    sample uses its own material and exposure_time from ``material``. The
    denominator comes from the ground truth without gradient, pooled over the batch so a patch that barely heats
    cannot blow the ratio up. Only ``T_pred`` receives a gradient.

    Args:
        T_pred, T_in, T_target, Q: (B, 1, D, H, W) in SI units [K], [W/m³].
        mask_in:  (B, 1, D, H, W) 0 = powder, 1 = consolidated, before the step.
        material: (B, MATERIAL_DIM) per-sample material vectors (``physics.material_vector``).
    """
    if form not in PHYSICS_FORMS:
        raise ValueError(f"physics form must be one of {PHYSICS_FORMS}, got {form!r}")
    diffusion = "implicit" if form == "coupled" else "explicit"
    T_in, T_target, Q = T_in.detach(), T_target.detach(), Q.detach()
    num = T_pred.new_zeros(())
    den = T_pred.new_zeros(())
    for i in range(T_pred.shape[0]):
        mat, dt = material_from_vector(material[i]), exposure_time_of(material[i])
        sl = slice(i, i + 1)
        R = heat_residual_si(T_in[sl], T_pred[sl], Q[sl], mask_in[sl], mat, dt, dx_m, dy_m, dz_m, "enthalpy", diffusion)
        num = num + R.pow(2).sum()
        with torch.no_grad():
            stored = mat.rho * (enthalpy(T_target[sl], mat) - enthalpy(T_in[sl], mat)) / dt
            den = den + stored[..., 1:-1, 1:-1, 1:-1].pow(2).sum()
    if not float(den) > 0.0:
        raise ValueError("no stored heat in the batch: every patch interior has T_target == T_in")
    return num / den


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
