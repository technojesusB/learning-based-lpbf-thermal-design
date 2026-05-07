"""Physical auditing metrics: PDE residual, energy conservation, cooling rate.

Conductivity is computed with the *same* harmonic-mean finite-difference
operator (``div_k_grad``) used by the Triton solver so that GT trajectories
yield near-zero residuals and false positives are avoided.
"""
from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff
from neural_pbf.physics.ops import div_k_grad


def pde_residual_l2(
    T_t: torch.Tensor,
    T_t1: torch.Tensor,
    dt: float,
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    Q_ext: torch.Tensor | None = None,
    material_mask: torch.Tensor | None = None,
) -> float:
    """L2 norm of the heat-equation residual [W²·m³]^(1/2).

    Residual:
        R = ρ·cp(T_t)·(T_{t+1} - T_t)/dt - ∇·(k(T_t)·∇T_t) - Q

    Uses harmonic-mean conductivity at cell faces to match the solver exactly.
    """
    k = k_eff(T_t, mat_cfg, material_mask)
    cp = cp_eff(T_t, mat_cfg)
    rho = mat_cfg.rho

    dz = sim_cfg.dz if sim_cfg.is_3d else None
    diff_term = div_k_grad(T_t, k, sim_cfg.dx, sim_cfg.dy, dz)

    Q = Q_ext if Q_ext is not None else torch.zeros_like(T_t)
    # Match 2D unit conversion from stepper: surface flux → volumetric
    if not sim_cfg.is_3d and Q_ext is not None:
        Q = Q / sim_cfg.dz

    dT_dt = (T_t1 - T_t) / dt
    residual = rho * cp * dT_dt - diff_term - Q

    dV = sim_cfg.dx * sim_cfg.dy * (sim_cfg.dz if sim_cfg.is_3d else 1.0)
    return (residual.pow(2).sum() * dV).sqrt().item()


def energy_conservation_error(
    T_t: torch.Tensor,
    T_t1: torch.Tensor,
    dt: float,
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    Q_ext: torch.Tensor | None = None,
    material_mask: torch.Tensor | None = None,
) -> float:
    """Relative energy conservation error.

    Compares sensible + latent heat change (ΔE_thermal) against net injected
    energy (Q_in - convective loss).

    Returns ``|ΔE_thermal - E_net| / (|E_net| + eps)``.
    """
    cp = cp_eff(T_t, mat_cfg)
    rho = mat_cfg.rho
    dV = sim_cfg.dx * sim_cfg.dy * (sim_cfg.dz if sim_cfg.is_3d else 1.0)

    delta_E = (rho * cp * (T_t1 - T_t) * dV).sum()

    Q = Q_ext if Q_ext is not None else torch.zeros_like(T_t)
    if not sim_cfg.is_3d and Q_ext is not None:
        Q = Q / sim_cfg.dz
    E_in = (Q * dt * dV).sum()

    # Convective loss term (linear cooling)
    E_loss = (sim_cfg.loss_h * (T_t - sim_cfg.T_ambient) * dt * dV).sum()
    E_net = E_in - E_loss

    return ((delta_E - E_net).abs() / (E_net.abs() + 1e-12)).item()


def cooling_rate_at_solidification(
    T_t: torch.Tensor,
    T_t1: torch.Tensor,
    dt: float,
    T_solidus: float,
) -> torch.Tensor:
    """Instantaneous cooling rate [K/s] at voxels crossing T_solidus downward.

    Returns zero everywhere except at voxels that solidified during this step.
    """
    crossing = (T_t > T_solidus) & (T_t1 <= T_solidus)
    cr = (T_t - T_t1) / dt
    return torch.where(crossing, cr, torch.zeros_like(cr))
