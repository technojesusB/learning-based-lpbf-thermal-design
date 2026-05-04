"""
Tests for last_n_sub field on SimulationState and its population in step_adaptive.

TDD RED phase: these tests must fail before state.py and stepper.py are updated.
"""
from __future__ import annotations

import math

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.utils.units import LengthUnit


def _small_2d_config() -> SimulationConfig:
    return SimulationConfig(
        Lx=0.1, Ly=0.05, Nx=8, Ny=4,
        length_unit=LengthUnit.MILLIMETERS,
    )


def _make_stepper(sim_cfg: SimulationConfig) -> TimeStepper:
    return TimeStepper(sim_cfg, MaterialConfig.ss316l_preset())


# ── SimulationState.last_n_sub field ──────────────────────────────────────────


@pytest.mark.unit
def test_last_n_sub_defaults_to_none():
    """last_n_sub must default to None on a freshly created state."""
    state = SimulationState.zeros(_small_2d_config(), device=torch.device("cpu"))
    assert state.last_n_sub is None


@pytest.mark.unit
def test_clone_preserves_none_last_n_sub():
    """clone() must carry forward last_n_sub=None."""
    state = SimulationState.zeros(_small_2d_config(), device=torch.device("cpu"))
    assert state.clone().last_n_sub is None


@pytest.mark.unit
def test_clone_preserves_set_last_n_sub():
    """clone() must carry forward a non-None last_n_sub value."""
    state = SimulationState.zeros(_small_2d_config(), device=torch.device("cpu"))
    state.last_n_sub = 42
    assert state.clone().last_n_sub == 42


# ── step_adaptive populates last_n_sub ────────────────────────────────────────


@pytest.mark.unit
def test_step_adaptive_sets_last_n_sub():
    """step_adaptive must set state.last_n_sub to a positive integer."""
    cfg = _small_2d_config()
    state = SimulationState.zeros(cfg, device=torch.device("cpu"))
    state = _make_stepper(cfg).step_adaptive(state, dt_target=1e-5)

    assert state.last_n_sub is not None
    assert isinstance(state.last_n_sub, int)
    assert state.last_n_sub >= 1


@pytest.mark.unit
def test_step_adaptive_last_n_sub_matches_expected():
    """last_n_sub must equal ceil(dt_target / (estimate_stability_dt * safety_factor))."""
    cfg = _small_2d_config()
    stepper = _make_stepper(cfg)
    state = SimulationState.zeros(cfg, device=torch.device("cpu"))

    dt_target = 1e-5
    safety_factor = 0.9
    dt_crit = stepper.estimate_stability_dt(state) * safety_factor
    expected = math.ceil(dt_target / dt_crit)

    state = stepper.step_adaptive(state, dt_target=dt_target, safety_factor=safety_factor)
    assert state.last_n_sub == expected


@pytest.mark.unit
def test_step_adaptive_large_dt_gives_multiple_substeps():
    """A dt_target much larger than dt_crit must produce last_n_sub > 1."""
    cfg = _small_2d_config()
    state = SimulationState.zeros(cfg, device=torch.device("cpu"))
    state = _make_stepper(cfg).step_adaptive(state, dt_target=1e-3)

    assert state.last_n_sub is not None
    assert state.last_n_sub > 1
