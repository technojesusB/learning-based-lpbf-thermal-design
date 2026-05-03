"""Tests for FMStepper.

CPU-only. Uses a freshly-initialized (near-zero output) VelocityNet so that
step() behaves predictably: T_out ≈ T_in (model predicts ~0 velocity).
"""

from __future__ import annotations

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.fm_stepper import FMStepper
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.utils.units import LengthUnit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NZ, NY, NX = 4, 4, 8


@pytest.fixture()
def sim_cfg() -> SimulationConfig:
    return SimulationConfig(
        Lx=1.0, Ly=0.5, Lz=0.125, Nx=NX, Ny=NY, Nz=NZ,
        length_unit=LengthUnit.MILLIMETERS,
    )


@pytest.fixture()
def fm_cfg() -> FMConfig:
    return FMConfig(
        base_channels=8,
        depth=2,
        cond_dim=12,
        cond_embed_dim=32,
        tau_embed_dim=32,
        n_inference_steps=4,
    )


@pytest.fixture()
def stepper(sim_cfg: SimulationConfig, fm_cfg: FMConfig) -> FMStepper:
    torch.manual_seed(42)
    model = VelocityNet(fm_cfg)
    encoder = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim)
    return FMStepper(
        model=model,
        cond_encoder=encoder,
        sim_cfg=sim_cfg,
        fm_cfg=fm_cfg,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def initial_state(sim_cfg: SimulationConfig) -> SimulationState:
    return SimulationState.zeros(sim_cfg, device=torch.device("cpu"), T_initial=400.0)


@pytest.fixture()
def conditioning() -> torch.Tensor:
    return torch.randn(12)


# ---------------------------------------------------------------------------
# Basic step contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_returns_simulation_state(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert isinstance(out, SimulationState)


@pytest.mark.unit
def test_step_returns_new_object(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert out is not initial_state, "step() must return a NEW SimulationState, not the input"


@pytest.mark.unit
def test_step_does_not_mutate_input_T(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    T_before = initial_state.T.clone()
    stepper.step(initial_state, conditioning)
    assert torch.allclose(initial_state.T, T_before), "step() must not mutate input state.T"


# ---------------------------------------------------------------------------
# Time and step counter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_increments_step_counter(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning, dt_target=1e-5)
    assert out.step == initial_state.step + 1


@pytest.mark.unit
def test_step_advances_time(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    dt = 5e-5
    out = stepper.step(initial_state, conditioning, dt_target=dt)
    assert abs(out.t - (initial_state.t + dt)) < 1e-12


# ---------------------------------------------------------------------------
# Output tensor shape and device
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_output_T_shape(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert out.T.shape == initial_state.T.shape, (
        f"Expected shape {initial_state.T.shape}, got {out.T.shape}"
    )


@pytest.mark.unit
def test_step_output_device_matches_input(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert out.T.device == initial_state.T.device


@pytest.mark.unit
def test_step_output_dtype_is_float32(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert out.T.dtype == torch.float32


# ---------------------------------------------------------------------------
# max_T is updated
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_updates_max_T(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert out.max_T is not None
    assert out.max_T.shape == out.T.shape
    # max_T must be >= T element-wise
    assert (out.max_T >= out.T).all()


# ---------------------------------------------------------------------------
# No NaN / Inf in output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_no_nan_inf(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    out = stepper.step(initial_state, conditioning)
    assert not torch.isnan(out.T).any(), "NaN in output T"
    assert not torch.isinf(out.T).any(), "Inf in output T"


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rollout_returns_list(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    n_macro = 3
    cond_seq = [conditioning for _ in range(n_macro)]
    out_seq = stepper.rollout(initial_state, cond_seq)
    assert isinstance(out_seq, list)
    assert len(out_seq) == n_macro


@pytest.mark.unit
def test_rollout_each_element_is_simulation_state(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    cond_seq = [conditioning] * 3
    out_seq = stepper.rollout(initial_state, cond_seq)
    for i, s in enumerate(out_seq):
        assert isinstance(s, SimulationState), f"Element {i} is not a SimulationState"


@pytest.mark.unit
def test_rollout_step_counter_increments(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    n_macro = 4
    cond_seq = [conditioning] * n_macro
    out_seq = stepper.rollout(initial_state, cond_seq)
    for i, s in enumerate(out_seq):
        assert s.step == initial_state.step + (i + 1)


@pytest.mark.unit
def test_rollout_does_not_mutate_initial_state(
    stepper: FMStepper,
    initial_state: SimulationState,
    conditioning: torch.Tensor,
) -> None:
    T_before = initial_state.T.clone()
    stepper.rollout(initial_state, [conditioning] * 3)
    assert torch.allclose(initial_state.T, T_before)
