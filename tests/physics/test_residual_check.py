"""Measurement logic for the pre-registered large-step residual check (docs/physics_residual_check.md)."""

from __future__ import annotations

import math

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.residual_check import (
    VARIANTS,
    exposure_residuals,
    relative_residual,
    select_variant,
    substep_check,
    summarize,
)
from neural_pbf.utils.units import LengthUnit

MAT = MaterialConfig.ss316l_preset()
SIM = SimulationConfig(Lx=0.032, Ly=0.024, Lz=0.016, Nx=17, Ny=13, Nz=9, length_unit=LengthUnit.MILLIMETERS)
SHAPE = (1, 1, SIM.Nz, SIM.Ny, SIM.Nx)


def _beam_case(T_hot: float = 1500.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A warm spot under a source on a powder bed with a solid floor."""
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in SHAPE[2:]), indexing="ij")
    r2 = (x - SIM.Nx // 2) ** 2 + (y - SIM.Ny // 2) ** 2 + (z - SIM.Nz + 1) ** 2
    T = (300.0 + (T_hot - 300.0) * torch.exp(-r2 / 8.0)).view(SHAPE)
    mask = (z < SIM.Nz // 2).to(torch.uint8).view(SHAPE)
    Q = (2e15 * torch.exp(-r2 / 4.0)).view(SHAPE)
    return T, mask, Q


def _exposure(T: torch.Tensor, mask: torch.Tensor, Q: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    state = SimulationState.zeros(SIM, device=torch.device("cpu"))
    state.T, state.material_mask = T.clone(), mask.clone()
    out = TimeStepper(SIM, MAT).step_adaptive(state, dt, Q)
    return out.T, out.material_mask


@pytest.mark.unit
def test_six_variants() -> None:
    assert len(VARIANTS) == 6 and len(set(VARIANTS)) == 6


@pytest.mark.unit
def test_relative_residual_is_zero_for_zero_residual() -> None:
    T, _, _ = _beam_case()
    R = torch.zeros(T[..., 1:-1, 1:-1, 1:-1].shape, dtype=torch.float64)
    assert relative_residual(R, T, T + 100.0, MAT, 1e-5) == 0.0


@pytest.mark.unit
def test_relative_residual_filter_ignores_small_change_voxels() -> None:
    """Residual only where |dT| <= 50 K must not count when the filter is on, and must count when it is off."""
    T_in = torch.full(SHAPE, 800.0, dtype=torch.float64)
    T_out = T_in.clone()
    T_out[..., 4, 6, 8] += 200.0
    R = torch.zeros(T_in[..., 1:-1, 1:-1, 1:-1].shape, dtype=torch.float64)
    R[..., 0, 0, 0] = 1e12  # a voxel that barely changed
    assert relative_residual(R, T_in, T_out, MAT, 1e-5, min_dT=50.0) == 0.0
    assert relative_residual(R, T_in, T_out, MAT, 1e-5) > 0.0


@pytest.mark.unit
def test_relative_residual_is_nan_without_eligible_voxels() -> None:
    T = torch.full(SHAPE, 800.0, dtype=torch.float64)
    R = torch.ones(T[..., 1:-1, 1:-1, 1:-1].shape, dtype=torch.float64)
    assert math.isnan(relative_residual(R, T, T + 1.0, MAT, 1e-5, min_dT=50.0))


@pytest.mark.unit
def test_substep_check_passes_for_the_solver_form() -> None:
    T, mask, Q = _beam_case()
    result = substep_check(T, mask, Q, MAT, SIM, exposure_time=40e-6)
    assert set(result) == set(VARIANTS)
    assert result[("cp_in", "explicit")] < 1e-3


@pytest.mark.unit
def test_substep_check_promotes_hot_powder_like_triton() -> None:
    """Powder above T_solidus must be solid inside the substep, or the solver form would not match."""
    T, _, Q = _beam_case(T_hot=2500.0)
    powder = torch.zeros(SHAPE, dtype=torch.uint8)
    assert substep_check(T, powder, Q, MAT, SIM, exposure_time=40e-6)[("cp_in", "explicit")] < 1e-3


@pytest.mark.unit
def test_exposure_residuals_reports_every_variant() -> None:
    T, mask, Q = _beam_case(T_hot=1200.0)
    T_out, mask_out = _exposure(T, mask, Q, 2e-6)
    result = exposure_residuals(T, T_out, Q, mask, mask_out, MAT, SIM, exposure_time=2e-6)
    assert set(result) == set(VARIANTS)
    assert all(v >= 0.0 for v in result.values())


@pytest.mark.unit
def test_summarize_gives_n_median_p90_per_regime_and_variant() -> None:
    v = ("enthalpy", "trapezoid")
    rows = [{"regime": "melt", v: x} for x in (0.1, 0.2, 0.3, 0.4, float("nan"))] + [{"regime": "solid", v: 0.05}]
    table = summarize(rows, [v])
    melt = table[("melt", v)]
    assert melt["n"] == 4 and melt["n_empty"] == 1
    assert melt["median"] == pytest.approx(0.25)
    assert melt["p90"] == pytest.approx(0.37)
    assert table[("solid", v)]["n"] == 1
    assert ("keyhole", v) not in table


@pytest.mark.unit
def test_select_variant_takes_lowest_eligible_melt_median() -> None:
    a, b, c = ("enthalpy", "trapezoid"), ("enthalpy", "implicit"), ("cp_in", "explicit")
    table = {("melt", a): {"median": 0.3}, ("melt", b): {"median": 0.1}, ("melt", c): {"median": 0.2}}
    assert select_variant(table, eligible={a, c}) == c
    assert select_variant(table, eligible=set()) is None


@pytest.mark.unit
def test_substep_check_in_float64_is_exact_to_machine_precision() -> None:
    T, mask, Q = _beam_case(T_hot=2500.0)
    result = substep_check(T, mask, Q, MAT, SIM, exposure_time=40e-6, dtype=torch.float64)
    assert result[("cp_in", "explicit")] < 1e-10
