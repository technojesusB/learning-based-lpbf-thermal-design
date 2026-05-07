"""Adapter contract tests: IdentityAdapter, TritonAdapter."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.core.state import SimulationState
from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.adapters.triton_adapter import TritonAdapter


# ── IdentityAdapter ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_identity_returns_same_temperature(sim_cfg, device):
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 500.0, device=device)
    state = SimulationState(T=T)
    adapter = IdentityAdapter()
    out = adapter.step(state, Q_ext=None, dt=1e-5)
    assert torch.allclose(out.T, T)


@pytest.mark.unit
def test_identity_advances_time(sim_cfg, device):
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 300.0, device=device)
    state = SimulationState(T=T, t=0.1)
    adapter = IdentityAdapter()
    out = adapter.step(state, Q_ext=None, dt=0.01)
    assert abs(out.t - 0.11) < 1e-12


@pytest.mark.unit
def test_identity_does_not_mutate_input(sim_cfg, device):
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 300.0, device=device)
    state = SimulationState(T=T.clone(), t=0.0)
    adapter = IdentityAdapter()
    adapter.step(state, Q_ext=None, dt=1e-5)
    assert state.t == 0.0  # original untouched


@pytest.mark.unit
def test_identity_increments_step(sim_cfg, device):
    T = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx, device=device)
    state = SimulationState(T=T, step=3)
    adapter = IdentityAdapter()
    out = adapter.step(state, Q_ext=None, dt=1e-5)
    assert out.step == 4


# ── TritonAdapter ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_triton_adapter_returns_simulation_state(sim_cfg, mat_cfg, device):
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), sim_cfg.T_ambient, device=device)
    state = SimulationState(T=T)
    adapter = TritonAdapter(sim_cfg, mat_cfg, use_triton=False, name="pytorch_ref")
    out = adapter.step(state, Q_ext=None, dt=1e-5)
    assert isinstance(out, SimulationState)
    assert out.T.shape == T.shape


@pytest.mark.unit
def test_triton_adapter_does_not_mutate_input(sim_cfg, mat_cfg, device):
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), sim_cfg.T_ambient, device=device)
    state = SimulationState(T=T.clone(), t=0.0)
    adapter = TritonAdapter(sim_cfg, mat_cfg, use_triton=False)
    adapter.step(state, Q_ext=None, dt=1e-5)
    assert state.t == 0.0


@pytest.mark.unit
def test_triton_adapter_custom_name(sim_cfg, mat_cfg):
    adapter = TritonAdapter(sim_cfg, mat_cfg, use_triton=False, name="my_solver")
    assert adapter.name == "my_solver"


@pytest.mark.unit
def test_fm_adapter_raises_without_conditioning_key(sim_cfg, mat_cfg, device):
    """FMAdapter must raise ValueError when 'vector' key is missing."""
    from unittest.mock import MagicMock

    from neural_pbf.eval.adapters.fm_adapter import FMAdapter
    from neural_pbf.models.generative.fm.config import FMConfig

    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_fm_cfg = MagicMock(spec=FMConfig)
    mock_fm_cfg.n_inference_steps = 2
    mock_fm_cfg.T_ambient = sim_cfg.T_ambient
    mock_fm_cfg.T_ref = 1000.0

    adapter = FMAdapter(
        model=mock_model,
        cond_encoder=mock_encoder,
        sim_cfg=sim_cfg,
        fm_cfg=mock_fm_cfg,
        device=device,
        name="fm_test",
    )
    T = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), sim_cfg.T_ambient, device=device)
    state = SimulationState(T=T)

    with pytest.raises(ValueError, match="vector"):
        adapter.step(state, Q_ext=None, dt=1e-5, conditioning={"wrong_key": None})
