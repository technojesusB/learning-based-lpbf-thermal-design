"""Physics helpers for Flow Matching LPBF surrogate training."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff, sigmoid_step
from neural_pbf.physics.material_vector import exposure_time_of, material_from_vector
from neural_pbf.physics.ops import div_k_grad
from neural_pbf.physics.triton_pde_loss import _TRITON_AVAILABLE, PDEResidualLoss
from neural_pbf.physics.triton_pde_loss import residual_sums as triton_residual_sums

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


PHYSICS_FORMS = ("coupled", "tin")
PHYSICS_BACKENDS = ("reference", "triton")


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
    num, den = _reference_sums(T_pred, T_in, T_target, Q, mask_in, material, form, dx_m, dy_m, dz_m)
    return _ratio(num.sum(), den.sum())


def _ratio(num: Tensor, den: Tensor) -> Tensor:
    if not float(den) > 0.0:
        raise ValueError("no stored heat in the batch: every patch interior has T_target == T_in")
    return num / den


def _reference_sums(
    T_pred: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor, material: Tensor, form: str,
    dx_m: float, dy_m: float, dz_m: float,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """Per-sample Σ R² (with gradient to T_pred) and Σ S² (without), SI inputs."""
    if form not in PHYSICS_FORMS:
        raise ValueError(f"physics form must be one of {PHYSICS_FORMS}, got {form!r}")
    diffusion = "implicit" if form == "coupled" else "explicit"
    T_in, T_target, Q = T_in.detach(), T_target.detach(), Q.detach()
    nums, dens = [], []
    for i in range(T_pred.shape[0]):
        mat, dt = material_from_vector(material[i]), exposure_time_of(material[i])
        sl = slice(i, i + 1)
        R = heat_residual_si(T_in[sl], T_pred[sl], Q[sl], mask_in[sl], mat, dt, dx_m, dy_m, dz_m, "enthalpy", diffusion)
        nums.append(R.pow(2).sum())
        with torch.no_grad():
            stored = mat.rho * (enthalpy(T_target[sl], mat) - enthalpy(T_in[sl], mat)) / dt
            dens.append(stored[..., 1:-1, 1:-1, 1:-1].pow(2).sum())
    return torch.stack(nums), torch.stack(dens)


def _check_backend(backend: str, field: Tensor) -> None:
    if backend not in PHYSICS_BACKENDS:
        raise ValueError(f"physics backend must be one of {PHYSICS_BACKENDS}, got {backend!r}")
    if backend == "triton" and not (field.is_cuda and _TRITON_AVAILABLE):
        raise ValueError("the triton backend needs CUDA tensors and an installed Triton")


def physics_loss(
    v_pred: Tensor, x_tau: Tensor, tau: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor,
    material: Tensor, form: str, T_ref: float, T_ambient: float, Q_ref: float, dx_m: float, dy_m: float, dz_m: float,
    *, backend: str,
) -> Tensor:  # fmt: skip
    """Physics loss for flow-matching training, in the dataset's normalized units; replaces physics_heat_residual.

    T_pred = x_tau + (1 − τ)·v_pred (OT-FM identity), with x_tau detached, so only v_pred gets a gradient. Fields are
    (B, 1, D, H, W) with T normalized as (T_SI − T_ambient)/T_ref and Q as Q_SI/Q_ref. Returns the batch-level ratio
    of ``physics_loss_reference``. ``backend`` is ``"reference"`` (plain PyTorch, any device) or ``"triton"`` (the
    kernels of ``triton_pde_loss``, CUDA only), which match the reference in value and gradient. There is no
    automatic choice: the script's --physics_backend fixes it.

    The function this replaced (tag ``v0-random-split-eval``) used dt fixed at 5 µs, one scalar k per sample,
    constant-cp storage and diffusion at the prediction.
    """
    if form not in PHYSICS_FORMS:
        raise ValueError(f"physics form must be one of {PHYSICS_FORMS}, got {form!r}")
    _check_backend(backend, v_pred)
    x_tau = x_tau.detach()
    if backend == "triton":
        loss = PDEResidualLoss.apply(
            v_pred, x_tau, tau, T_in, T_target, Q, mask_in, material, form, T_ref, T_ambient, Q_ref, dx_m, dy_m, dz_m
        )
        return cast(Tensor, loss)
    T_pred = (x_tau + (1.0 - tau.view(-1, 1, 1, 1, 1)) * v_pred) * T_ref + T_ambient
    return physics_loss_reference(
        T_pred, T_in * T_ref + T_ambient, T_target * T_ref + T_ambient, Q * Q_ref, mask_in, material, form,
        dx_m, dy_m, dz_m,
    )  # fmt: skip


@torch.no_grad()
def physics_residual_sums(
    T_pred: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor, material: Tensor, form: str,
    T_ref: float, T_ambient: float, Q_ref: float, dx_m: float, dy_m: float, dz_m: float, *, backend: str,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """Per-sample (Σ R², Σ S²) of the physics loss for a predicted temperature (normalized units), no gradient.

    Pooled sums give the loss's normalization over any set of samples: Σ num / Σ den over the val set, or over the
    melt-regime samples only.
    """
    if form not in PHYSICS_FORMS:
        raise ValueError(f"physics form must be one of {PHYSICS_FORMS}, got {form!r}")
    _check_backend(backend, T_pred)
    if backend == "triton":
        return triton_residual_sums(
            T_pred, T_in, T_target, Q, mask_in, material, form, T_ref, T_ambient, Q_ref, dx_m, dy_m, dz_m
        )

    def si(t: Tensor) -> Tensor:
        return t * T_ref + T_ambient

    return _reference_sums(si(T_pred), si(T_in), si(T_target), Q * Q_ref, mask_in, material, form, dx_m, dy_m, dz_m)
