"""Physics audit tests: uniform T → zero residual, analytic energy balance."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.metrics.physics_audit import (
    cooling_rate_at_solidification,
    energy_conservation_error,
    pde_residual_l2,
)


@pytest.mark.unit
def test_pde_residual_uniform_T_is_zero(sim_cfg, mat_cfg):
    """A perfectly uniform temperature field evolving only due to a zero source
    should have near-zero PDE residual (diffusion and time derivative cancel)."""
    T_t = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 500.0)
    T_t1 = T_t.clone()  # no change
    residual = pde_residual_l2(T_t, T_t1, dt=1e-5, sim_cfg=sim_cfg, mat_cfg=mat_cfg)
    assert residual == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_pde_residual_returns_float(sim_cfg, mat_cfg):
    T_t = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx) * 300.0
    T_t1 = T_t + 1.0
    result = pde_residual_l2(T_t, T_t1, dt=1e-5, sim_cfg=sim_cfg, mat_cfg=mat_cfg)
    assert isinstance(result, float)


@pytest.mark.unit
def test_energy_conservation_no_source(sim_cfg, mat_cfg):
    """Without any heat source and no cooling loss, ΔE and E_net should both be
    zero for a static field, giving a well-defined (small) relative error."""
    T_t = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 300.0)
    T_t1 = T_t.clone()
    err = energy_conservation_error(
        T_t, T_t1, dt=1e-5, sim_cfg=sim_cfg, mat_cfg=mat_cfg
    )
    assert isinstance(err, float)
    # Both numerator and denominator near zero — just check it doesn't raise
    assert err >= 0.0


@pytest.mark.unit
def test_cooling_rate_at_solidification_identifies_crossing():
    T_solidus = 1653.0
    T_t = torch.tensor([[[[1700.0, 300.0], [300.0, 300.0]]]])   # one voxel above
    T_t1 = torch.tensor([[[[1600.0, 300.0], [300.0, 300.0]]]])  # that voxel crosses
    dt = 1e-5
    cr = cooling_rate_at_solidification(T_t, T_t1, dt=dt, T_solidus=T_solidus)
    # Only voxel [0,0,0,0] crosses — expected cooling rate = (1700-1600)/1e-5 = 1e7 K/s
    assert cr[0, 0, 0, 0].item() == pytest.approx(1e7, rel=1e-4)
    assert cr[0, 0, 0, 1].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_cooling_rate_no_crossing_is_zero():
    T_t = torch.full((1, 1, 4, 4), 300.0)
    T_t1 = T_t - 10.0  # cooling, but never crossed solidus
    cr = cooling_rate_at_solidification(T_t, T_t1, dt=1e-5, T_solidus=1653.0)
    assert (cr == 0.0).all()
