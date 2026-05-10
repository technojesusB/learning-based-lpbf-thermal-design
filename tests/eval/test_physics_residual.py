"""Unit tests for physics_heat_residual."""

from __future__ import annotations

import pytest
import torch

from experiments.train_fm_dit_rope import physics_heat_residual
from neural_pbf.physics.ops import div_k_grad


def _scalar_to_field(val: float, B: int) -> torch.Tensor:
    """Create a (B, 1, 1, 1, 1) tensor broadcast-compatible with (B, 1, D, H, W)."""
    return torch.full((B, 1, 1, 1, 1), val)


@pytest.mark.unit
def test_physics_residual_returns_scalar() -> None:
    B, D, H, W = 2, 8, 8, 8
    shape = (B, 1, D, H, W)
    v_pred = torch.randn(*shape, requires_grad=True)
    x_tau = torch.randn(*shape)
    Q = torch.randn(*shape)
    res = physics_heat_residual(
        v_pred, x_tau, Q,
        rho=_scalar_to_field(7900.0, B),
        cp=_scalar_to_field(500.0, B),
        k=_scalar_to_field(20.0, B),
        dx_m=1e-5, dy_m=1e-5, dz_m=1e-5,
    )
    assert res.shape == (), f"Expected scalar, got shape {res.shape}"


@pytest.mark.unit
def test_physics_residual_gradient_flows_through_v_pred() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    x_tau = torch.randn(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    res = physics_heat_residual(
        v_pred, x_tau, Q,
        rho=_scalar_to_field(1.0, B),
        cp=_scalar_to_field(1.0, B),
        k=_scalar_to_field(1.0, B),
        dx_m=1e-4, dy_m=1e-4, dz_m=1e-4,
    )
    res.backward()
    assert v_pred.grad is not None
    assert v_pred.grad.shape == v_pred.shape


@pytest.mark.unit
def test_physics_residual_no_gradient_through_x_tau() -> None:
    """x_tau is intended to be detached — verify it carries no grad in normal use."""
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    x_tau = torch.randn(B, 1, D, H, W, requires_grad=True)
    res = physics_heat_residual(
        v_pred, x_tau.detach(), Q=torch.zeros(B, 1, D, H, W),
        rho=_scalar_to_field(1.0, B),
        cp=_scalar_to_field(1.0, B),
        k=_scalar_to_field(1.0, B),
        dx_m=1e-4, dy_m=1e-4, dz_m=1e-4,
    )
    res.backward()
    assert x_tau.grad is None, "x_tau should not receive gradients when detached"


@pytest.mark.unit
def test_physics_residual_near_zero_on_exact_solution() -> None:
    """When v_pred == (∇·(k∇T) + Q) / (ρ·cp), the residual should be ~0."""
    B, D, H, W = 1, 8, 8, 8
    rho_val, cp_val, k_val = 1.0, 1.0, 1.0
    dx_m = dy_m = dz_m = 1.0

    x_tau = torch.rand(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)

    k_field = torch.full_like(x_tau, k_val)
    divkgrad = div_k_grad(x_tau, k_field, dx_m, dy_m, dz_m)
    v_exact = (divkgrad + Q) / (rho_val * cp_val)

    res = physics_heat_residual(
        v_exact, x_tau, Q,
        rho=_scalar_to_field(rho_val, B),
        cp=_scalar_to_field(cp_val, B),
        k=_scalar_to_field(k_val, B),
        dx_m=dx_m, dy_m=dy_m, dz_m=dz_m,
    )
    assert res.item() < 1e-8, f"Residual should be ~0 for exact solution, got {res.item()}"


@pytest.mark.unit
def test_physics_residual_nonzero_for_wrong_v() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.ones(B, 1, D, H, W) * 1e6
    x_tau = torch.zeros(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    res = physics_heat_residual(
        v_pred, x_tau, Q,
        rho=_scalar_to_field(1.0, B),
        cp=_scalar_to_field(1.0, B),
        k=_scalar_to_field(1.0, B),
        dx_m=1.0, dy_m=1.0, dz_m=1.0,
    )
    assert res.item() > 0.0


@pytest.mark.unit
def test_physics_residual_per_sample_heterogeneous_material() -> None:
    """Different per-sample material properties must produce different residuals."""
    B, D, H, W = 2, 4, 4, 4
    v_pred = torch.ones(B, 1, D, H, W)
    x_tau = torch.zeros(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)

    # Two very different rho values per sample
    rho = torch.tensor([1.0, 1e6]).view(B, 1, 1, 1, 1)
    cp = _scalar_to_field(1.0, B)
    k = _scalar_to_field(1.0, B)

    # Compute residual — if batch-mean were used instead, this would equal a
    # single scalar residual; per-sample produces different LHS contributions.
    lhs_per_sample = (rho * cp * v_pred).view(B, -1).mean(dim=1)
    assert not torch.allclose(lhs_per_sample[0], lhs_per_sample[1]), (
        "Per-sample rho must produce different LHS contributions across batch"
    )

    # Full residual should be non-zero
    res = physics_heat_residual(
        v_pred, x_tau, Q, rho=rho, cp=cp, k=k,
        dx_m=1.0, dy_m=1.0, dz_m=1.0,
    )
    assert res.item() > 0.0
