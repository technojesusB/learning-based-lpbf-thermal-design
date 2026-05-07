"""Unit tests for physics_heat_residual."""

from __future__ import annotations

import pytest
import torch

from experiments.train_fm_dit_physics import physics_heat_residual
from neural_pbf.physics.ops import div_k_grad


@pytest.mark.unit
def test_physics_residual_returns_scalar() -> None:
    B, D, H, W = 2, 8, 8, 8
    shape = (B, 1, D, H, W)
    v_pred = torch.randn(*shape, requires_grad=True)
    x_tau = torch.randn(*shape)
    Q = torch.randn(*shape)
    res = physics_heat_residual(v_pred, x_tau, Q, rho=7900.0, cp=500.0, k=20.0,
                                dx_m=1e-5, dy_m=1e-5, dz_m=1e-5)
    assert res.shape == (), f"Expected scalar, got shape {res.shape}"


@pytest.mark.unit
def test_physics_residual_gradient_flows_through_v_pred() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    x_tau = torch.randn(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    res = physics_heat_residual(v_pred, x_tau, Q, rho=1.0, cp=1.0, k=1.0,
                                dx_m=1e-4, dy_m=1e-4, dz_m=1e-4)
    res.backward()
    assert v_pred.grad is not None
    assert v_pred.grad.shape == v_pred.shape


@pytest.mark.unit
def test_physics_residual_no_gradient_through_x_tau() -> None:
    """x_tau is intended to be detached — verify it carries no grad in normal use."""
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    x_tau = torch.randn(B, 1, D, H, W, requires_grad=True)
    # Pass detached version as the API expects
    res = physics_heat_residual(v_pred, x_tau.detach(), Q=torch.zeros(B, 1, D, H, W),
                                rho=1.0, cp=1.0, k=1.0,
                                dx_m=1e-4, dy_m=1e-4, dz_m=1e-4)
    res.backward()
    assert x_tau.grad is None, "x_tau should not receive gradients when detached"


@pytest.mark.unit
def test_physics_residual_near_zero_on_exact_solution() -> None:
    """When v_pred == (∇·(k∇T) + Q) / (ρ·cp), the residual should be ~0."""
    B, D, H, W = 1, 8, 8, 8
    rho, cp, k = 1.0, 1.0, 1.0
    dx_m = dy_m = dz_m = 1.0  # unit spacing keeps magnitudes tractable

    x_tau = torch.rand(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)

    # Exact RHS
    k_field = torch.full_like(x_tau, k)
    divkgrad = div_k_grad(x_tau, k_field, dx_m, dy_m, dz_m)
    v_exact = (divkgrad + Q) / (rho * cp)

    res = physics_heat_residual(v_exact, x_tau, Q, rho=rho, cp=cp, k=k,
                                dx_m=dx_m, dy_m=dy_m, dz_m=dz_m)
    assert res.item() < 1e-8, f"Residual should be ~0 for exact solution, got {res.item()}"


@pytest.mark.unit
def test_physics_residual_nonzero_for_wrong_v() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.ones(B, 1, D, H, W) * 1e6  # clearly wrong
    x_tau = torch.zeros(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    res = physics_heat_residual(v_pred, x_tau, Q, rho=1.0, cp=1.0, k=1.0,
                                dx_m=1.0, dy_m=1.0, dz_m=1.0)
    assert res.item() > 0.0
