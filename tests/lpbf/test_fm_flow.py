"""Tests for FM flow helpers (interpolate, target_velocity, fm_loss, residuum).

All run on CPU. No GPU required.
"""

from __future__ import annotations

import pytest
import torch

from neural_pbf.models.generative.fm.flow import (
    compute_physics_residuum,
    fm_loss,
    interpolate,
    sample_noise,
    target_velocity,
)


# ---------------------------------------------------------------------------
# sample_noise
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_noise_shape_matches_input() -> None:
    x = torch.randn(2, 1, 4, 8, 8)
    noise = sample_noise(x)
    assert noise.shape == x.shape


@pytest.mark.unit
def test_sample_noise_dtype_matches() -> None:
    x = torch.zeros(1, 1, 4, 8, 8, dtype=torch.float32)
    assert sample_noise(x).dtype == torch.float32


@pytest.mark.unit
def test_sample_noise_not_all_zeros() -> None:
    x = torch.zeros(2, 1, 4, 8, 8)
    assert sample_noise(x).abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interpolate_tau_zero_returns_noise() -> None:
    """At τ=0 the interpolated value should be pure noise."""
    noise = torch.randn(1, 1, 4, 8, 8)
    target = torch.randn(1, 1, 4, 8, 8)
    tau = torch.zeros(1)
    result = interpolate(noise, target, tau)
    assert torch.allclose(result, noise, atol=1e-6), "τ=0 should return noise"


@pytest.mark.unit
def test_interpolate_tau_one_returns_target() -> None:
    """At τ=1 the interpolated value should be the clean target."""
    noise = torch.randn(1, 1, 4, 8, 8)
    target = torch.randn(1, 1, 4, 8, 8)
    tau = torch.ones(1)
    result = interpolate(noise, target, tau)
    assert torch.allclose(result, target, atol=1e-6), "τ=1 should return target"


@pytest.mark.unit
def test_interpolate_tau_half() -> None:
    """At τ=0.5 the result should be the midpoint."""
    noise = torch.zeros(1, 1, 4, 8, 8)
    target = torch.ones(1, 1, 4, 8, 8) * 2.0
    tau = torch.full((1,), 0.5)
    result = interpolate(noise, target, tau)
    assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


@pytest.mark.unit
def test_interpolate_output_shape() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    tau = torch.rand(2)
    result = interpolate(noise, target, tau)
    assert result.shape == noise.shape


@pytest.mark.unit
def test_interpolate_batch_dimension() -> None:
    """Each batch element should use its own τ value."""
    noise = torch.zeros(2, 1, 4, 8, 8)
    target = torch.ones(2, 1, 4, 8, 8)
    tau = torch.tensor([0.0, 1.0])
    result = interpolate(noise, target, tau)
    assert torch.allclose(result[0], noise[0], atol=1e-6)
    assert torch.allclose(result[1], target[1], atol=1e-6)


# ---------------------------------------------------------------------------
# target_velocity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_velocity_shape() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v = target_velocity(noise, target)
    assert v.shape == noise.shape


@pytest.mark.unit
def test_target_velocity_is_target_minus_noise() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v = target_velocity(noise, target)
    expected = target - noise
    assert torch.allclose(v, expected, atol=1e-6)


@pytest.mark.unit
def test_target_velocity_zero_when_noise_equals_target() -> None:
    x = torch.randn(1, 1, 4, 8, 8)
    v = target_velocity(x, x)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


# ---------------------------------------------------------------------------
# fm_loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fm_loss_zero_when_v_pred_equals_v_target() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v_target = target_velocity(noise, target)
    loss = fm_loss(v_target, noise, target)
    assert loss.item() < 1e-10, f"Loss should be ~0 when v_pred == v_target, got {loss.item()}"


@pytest.mark.unit
def test_fm_loss_positive_for_wrong_prediction() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v_wrong = torch.zeros_like(noise)
    loss = fm_loss(v_wrong, noise, target)
    assert loss.item() > 0.0


@pytest.mark.unit
def test_fm_loss_returns_scalar() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v_pred = torch.randn_like(noise)
    loss = fm_loss(v_pred, noise, target)
    assert loss.ndim == 0, f"fm_loss should return a scalar, got shape {loss.shape}"


@pytest.mark.unit
def test_fm_loss_is_differentiable() -> None:
    noise = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8)
    v_pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    loss = fm_loss(v_pred, noise, target)
    loss.backward()
    assert v_pred.grad is not None


# ---------------------------------------------------------------------------
# compute_physics_residuum — placeholder contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_physics_residuum_returns_zero_tensor() -> None:
    from neural_pbf.core.config import SimulationConfig
    from neural_pbf.physics.material import MaterialConfig
    from neural_pbf.utils.units import LengthUnit

    sim_cfg = SimulationConfig(Lx=1.0, Ly=0.5, Lz=0.125, Nx=8, Ny=4, Nz=2, length_unit=LengthUnit.MILLIMETERS)
    mat_cfg = MaterialConfig.ss316l_preset()

    v_pred = torch.randn(1, 1, 2, 4, 8)
    x_tau = torch.randn(1, 3, 2, 4, 8)
    tau = torch.tensor([0.5])

    result = compute_physics_residuum(v_pred, x_tau, tau, sim_cfg, mat_cfg)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), (
        "compute_physics_residuum placeholder must return zeros"
    )


@pytest.mark.unit
def test_physics_residuum_is_scalar() -> None:
    from neural_pbf.core.config import SimulationConfig
    from neural_pbf.physics.material import MaterialConfig
    from neural_pbf.utils.units import LengthUnit

    sim_cfg = SimulationConfig(Lx=1.0, Ly=0.5, Lz=0.125, Nx=8, Ny=4, Nz=2, length_unit=LengthUnit.MILLIMETERS)
    mat_cfg = MaterialConfig.ss316l_preset()

    v_pred = torch.randn(1, 1, 2, 4, 8)
    x_tau = torch.randn(1, 3, 2, 4, 8)
    tau = torch.tensor([0.5])

    result = compute_physics_residuum(v_pred, x_tau, tau, sim_cfg, mat_cfg)
    assert result.ndim == 0, f"Expected scalar (0-D), got shape {result.shape}"


@pytest.mark.unit
def test_physics_residuum_respects_device() -> None:
    from neural_pbf.core.config import SimulationConfig
    from neural_pbf.physics.material import MaterialConfig
    from neural_pbf.utils.units import LengthUnit

    sim_cfg = SimulationConfig(Lx=1.0, Ly=0.5, Lz=0.125, Nx=8, Ny=4, Nz=2, length_unit=LengthUnit.MILLIMETERS)
    mat_cfg = MaterialConfig.ss316l_preset()

    v_pred = torch.randn(1, 1, 2, 4, 8)
    x_tau = torch.randn(1, 3, 2, 4, 8)
    tau = torch.tensor([0.5])

    result = compute_physics_residuum(v_pred, x_tau, tau, sim_cfg, mat_cfg)
    assert result.device == v_pred.device
