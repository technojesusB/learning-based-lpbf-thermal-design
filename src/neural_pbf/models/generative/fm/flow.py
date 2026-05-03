"""Flow Matching helpers: interpolation, velocity target, loss, physics residual."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neural_pbf.core.config import SimulationConfig
from neural_pbf.physics.material import MaterialConfig


def sample_noise(x_target: Tensor) -> Tensor:
    """Sample standard Gaussian noise with the same shape as x_target."""
    return torch.randn_like(x_target)


def interpolate(noise: Tensor, x_target: Tensor, tau: Tensor) -> Tensor:
    """Linear interpolation: x_τ = (1-τ)·noise + τ·x_target.

    Args:
        noise:    (B, C, ...) Gaussian noise.
        x_target: (B, C, ...) clean target field.
        tau:      (B,) flow time ∈ [0, 1].

    Returns:
        (B, C, ...) interpolated field.
    """
    # Broadcast τ to spatial dimensions
    ndim_extra = noise.ndim - 1  # all dims except batch
    tau_b = tau.float()
    for _ in range(ndim_extra):
        tau_b = tau_b.unsqueeze(-1)
    return (1.0 - tau_b) * noise + tau_b * x_target


def target_velocity(noise: Tensor, x_target: Tensor) -> Tensor:
    """Optimal FM velocity: v* = x_target - noise (constant along each path)."""
    return x_target - noise


def fm_loss(v_pred: Tensor, noise: Tensor, x_target: Tensor) -> Tensor:
    """Mean-squared error between predicted and target velocity.

    Args:
        v_pred:   (B, C, ...) model output.
        noise:    (B, C, ...) noise used for interpolation.
        x_target: (B, C, ...) clean target field.

    Returns:
        Scalar loss tensor.
    """
    v_target = target_velocity(noise, x_target)
    return F.mse_loss(v_pred, v_target)


def compute_physics_residuum(
    v_pred: Tensor,
    x_tau: Tensor,
    tau: Tensor,
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
) -> Tensor:
    """Physics residual placeholder for future PINN-FM integration.

    Currently returns a zero scalar.  Future implementation should approximate:
        ρ·cp(T)·v_pred ≈ ∇·(k(T)·∇T) + Q
    using the finite-difference operators already available in neural_pbf.physics.ops.

    TODO: implement PINN-FM residual using ρ·cp(T)·v ≈ ∇·(k∇T) + Q
    """
    return torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
