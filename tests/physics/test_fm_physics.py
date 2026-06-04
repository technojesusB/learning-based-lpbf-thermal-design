"""Unit tests for fm_physics module (physics_heat_residual, denorm_cond_batch)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from neural_pbf.physics.fm_physics import denorm_cond_batch, physics_heat_residual
from neural_pbf.physics.ops import div_k_grad


def _scalar_field(val: float, B: int) -> torch.Tensor:
    return torch.full((B, 1, 1, 1, 1), val)


def _identity_cfg() -> SimpleNamespace:
    return SimpleNamespace(T_ref=1.0, T_ambient=0.0, Q_ref=1.0)


# ---------------------------------------------------------------------------
# physics_heat_residual
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_physics_residual_returns_scalar() -> None:
    B, D, H, W = 2, 8, 8, 8
    shape = (B, 1, D, H, W)
    v_pred = torch.randn(*shape, requires_grad=True)
    res = physics_heat_residual(
        v_pred, torch.randn(*shape), torch.zeros(B), torch.randn(*shape), torch.randn(*shape),
        rho=_scalar_field(7900.0, B), cp=_scalar_field(500.0, B), k=_scalar_field(20.0, B),
        dx_m=1e-5, dy_m=1e-5, dz_m=1e-5, ds_cfg=_identity_cfg(),
    )
    assert res.shape == ()


@pytest.mark.unit
def test_physics_residual_gradient_flows_through_v_pred() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    res = physics_heat_residual(
        v_pred, torch.randn(B, 1, D, H, W), torch.zeros(B),
        torch.randn(B, 1, D, H, W), torch.zeros(B, 1, D, H, W),
        rho=_scalar_field(1.0, B), cp=_scalar_field(1.0, B), k=_scalar_field(1.0, B),
        dx_m=1e-4, dy_m=1e-4, dz_m=1e-4, ds_cfg=_identity_cfg(),
    )
    res.backward()
    assert v_pred.grad is not None
    assert v_pred.grad.shape == v_pred.shape


@pytest.mark.unit
def test_physics_residual_no_gradient_through_detached_x_tau() -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.randn(B, 1, D, H, W, requires_grad=True)
    x_tau = torch.randn(B, 1, D, H, W, requires_grad=True)
    res = physics_heat_residual(
        v_pred, x_tau.detach(), torch.zeros(B),
        torch.randn(B, 1, D, H, W), torch.zeros(B, 1, D, H, W),
        rho=_scalar_field(1.0, B), cp=_scalar_field(1.0, B), k=_scalar_field(1.0, B),
        dx_m=1e-4, dy_m=1e-4, dz_m=1e-4, ds_cfg=_identity_cfg(),
    )
    res.backward()
    assert x_tau.grad is None


@pytest.mark.unit
def test_physics_residual_near_zero_on_exact_solution() -> None:
    """Residual ≈ 0 when lhs and rhs balance exactly."""
    B, D, H, W = 1, 8, 8, 8
    dx_m = dy_m = dz_m = 1.0
    x_tau = torch.rand(B, 1, D, H, W)
    k_field = torch.ones_like(x_tau)
    divkgrad = div_k_grad(x_tau, k_field, dx_m, dy_m, dz_m)
    Q = -divkgrad
    res = physics_heat_residual(
        v_pred=torch.zeros_like(x_tau), x_tau_detached=x_tau,
        tau=torch.zeros(B), T_in=x_tau, Q=Q,
        rho=_scalar_field(1.0, B), cp=_scalar_field(1.0, B), k=_scalar_field(1.0, B),
        dx_m=dx_m, dy_m=dy_m, dz_m=dz_m, ds_cfg=_identity_cfg(), dt_s=1.0,
    )
    assert res.item() < 1e-8, f"Residual should be ~0, got {res.item()}"


@pytest.mark.unit
def test_physics_residual_nonzero_for_wrong_v() -> None:
    B, D, H, W = 1, 4, 4, 4
    res = physics_heat_residual(
        v_pred=torch.ones(B, 1, D, H, W) * 1e6,
        x_tau_detached=torch.zeros(B, 1, D, H, W),
        tau=torch.zeros(B),
        T_in=torch.zeros(B, 1, D, H, W),
        Q=torch.zeros(B, 1, D, H, W),
        rho=_scalar_field(1.0, B), cp=_scalar_field(1.0, B), k=_scalar_field(1.0, B),
        dx_m=1.0, dy_m=1.0, dz_m=1.0, ds_cfg=_identity_cfg(),
    )
    assert res.item() > 0.0


# ---------------------------------------------------------------------------
# denorm_cond_batch
# ---------------------------------------------------------------------------


def _fake_ds_cfg() -> Any:
    """SimpleNamespace mimicking FMDatasetConfig normalisation attributes."""
    return SimpleNamespace(
        conditioning_keys=("rho", "cp", "k_s"),
        cond_means={"rho": 7900.0, "cp": 500.0, "k_s": 20.0},
        cond_stds={"rho": 500.0, "cp": 60.0, "k_s": 6.0},
    )


@pytest.mark.unit
def test_denorm_cond_batch_output_shape() -> None:
    B = 4
    cfg = _fake_ds_cfg()
    cond = torch.randn(B, 3)
    out = denorm_cond_batch(cond, cfg, "rho")
    assert out.shape == (B, 1, 1, 1, 1)


@pytest.mark.unit
def test_denorm_cond_batch_roundtrip() -> None:
    """De-normalising a z-score of 0 should return the mean."""
    B = 3
    cfg = _fake_ds_cfg()
    cond = torch.zeros(B, 3)  # all z-scores = 0
    out = denorm_cond_batch(cond, cfg, "rho")
    expected = 7900.0
    assert abs(out.mean().item() - expected) < 1e-3


@pytest.mark.unit
def test_denorm_cond_batch_correct_index_selected() -> None:
    B = 2
    cfg = _fake_ds_cfg()
    # Set only cp channel (idx=1) to z-score 1.0
    cond = torch.zeros(B, 3)
    cond[:, 1] = 1.0
    cp_out = denorm_cond_batch(cond, cfg, "cp")
    # z=1 → cp = 500 + 60*1 = 560
    assert abs(cp_out.mean().item() - 560.0) < 1e-2


@pytest.mark.unit
def test_denorm_cond_batch_broadcast_shape() -> None:
    B = 5
    cfg = _fake_ds_cfg()
    cond = torch.randn(B, 3)
    out = denorm_cond_batch(cond, cfg, "k_s")
    # Shape must be broadcast-compatible with (B, 1, D, H, W) tensors
    dummy_field = torch.randn(B, 1, 8, 8, 8)
    result = dummy_field * out
    assert result.shape == (B, 1, 8, 8, 8)
