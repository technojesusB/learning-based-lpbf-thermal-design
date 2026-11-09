"""Reference heat-equation residual in SI units, checked against the solver's own update rule."""

from __future__ import annotations

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.fm_physics import enthalpy, heat_residual_si
from neural_pbf.physics.material import MaterialConfig, cp_eff
from neural_pbf.utils.units import LengthUnit

MAT = MaterialConfig.ss316l_preset()
SIM = SimulationConfig(Lx=0.016, Ly=0.012, Lz=0.010, Nx=9, Ny=7, Nz=6, length_unit=LengthUnit.MILLIMETERS)
SHAPE = (1, 1, SIM.Nz, SIM.Ny, SIM.Nx)


def _fields(seed: int = 0, T_lo: float = 400.0, T_hi: float = 1500.0) -> tuple[torch.Tensor, ...]:
    """Temperatures below T_solidus, a random powder/solid mask, and a random source."""
    g = torch.Generator().manual_seed(seed)
    T = T_lo + (T_hi - T_lo) * torch.rand(SHAPE, generator=g, dtype=torch.float64)
    mask = (torch.rand(SHAPE, generator=g) > 0.5).to(torch.uint8)
    Q = 1e14 * torch.rand(SHAPE, generator=g, dtype=torch.float64)
    return T, mask, Q


def _residual(T_in, T_out, Q, mask, dt, **kw):
    return heat_residual_si(T_in, T_out, Q, mask, MAT, dt, SIM.dx, SIM.dy, SIM.dz, **kw)


def _solver_step(T: torch.Tensor, mask: torch.Tensor, Q: torch.Tensor, dt: float) -> torch.Tensor:
    state = SimulationState.zeros(SIM, device=torch.device("cpu"), dtype=torch.float64)
    state.T, state.material_mask = T.clone(), mask.clone()
    return TimeStepper(SIM, MAT).step_explicit_euler(state, dt, Q).T


def _rel(R: torch.Tensor, Q: torch.Tensor) -> float:
    return float(R.norm() / Q[..., 1:-1, 1:-1, 1:-1].norm())


@pytest.mark.unit
def test_uniform_field_without_source_has_zero_residual() -> None:
    T = torch.full(SHAPE, 800.0, dtype=torch.float64)
    R = _residual(T, T.clone(), torch.zeros_like(T), torch.ones(SHAPE, dtype=torch.uint8), 1e-5)
    assert R.shape == (1, 1, SIM.Nz - 2, SIM.Ny - 2, SIM.Nx - 2)
    assert torch.count_nonzero(R) == 0


@pytest.mark.unit
def test_one_solver_substep_is_exact_for_explicit_cp_form() -> None:
    """The reference must reproduce the solver's update rule, not just the continuous PDE."""
    T, mask, Q = _fields()
    T_out = _solver_step(T, mask, Q, 1e-9)
    R = _residual(T, T_out, Q, mask, 1e-9, storage="cp_in", diffusion="explicit")
    assert _rel(R, Q) < 1e-9


@pytest.mark.unit
def test_enthalpy_form_converges_to_the_solver_step() -> None:
    T, mask, Q = _fields(seed=1)
    errs = []
    for dt in (1e-6, 1e-7):
        T_out = _solver_step(T, mask, Q, dt)
        errs.append(_rel(_residual(T, T_out, Q, mask, dt, storage="enthalpy", diffusion="explicit"), Q))
    assert errs[1] < 0.2 * errs[0]


@pytest.mark.unit
def test_enthalpy_derivative_is_cp_eff() -> None:
    T = torch.tensor([300.0, 450.0, 899.0, 1200.0, 1600.0, 1650.0, 1663.0, 1670.0, 1690.0, 3000.0], dtype=torch.float64)
    h = 1e-3
    dH = (enthalpy(T + h, MAT) - enthalpy(T - h, MAT)) / (2 * h)
    torch.testing.assert_close(dH, cp_eff(T, MAT), rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_enthalpy_across_the_melt_range_includes_latent_heat() -> None:
    T = torch.tensor([MAT.T_solidus - 50.0, MAT.T_liquidus + 50.0], dtype=torch.float64)
    H = enthalpy(T, MAT)
    assert float(H[1] - H[0]) == pytest.approx(MAT.latent_heat_L + 120.0 * 732.0, rel=0.01)


@pytest.mark.unit
def test_powder_above_solidus_counts_as_solid() -> None:
    """The Triton kernel promotes powder at T >= T_solidus inside every substep."""
    T, _, Q = _fields(seed=2, T_lo=1700.0, T_hi=2500.0)  # all above T_solidus
    T_out = T + 1.0
    powder, solid = torch.zeros(SHAPE, dtype=torch.uint8), torch.ones(SHAPE, dtype=torch.uint8)
    torch.testing.assert_close(_residual(T, T_out, Q, powder, 1e-5), _residual(T, T_out, Q, solid, 1e-5))


@pytest.mark.unit
def test_powder_below_solidus_uses_powder_conductivity() -> None:
    T, _, Q = _fields(seed=3)
    T_out = T + 1.0
    powder, solid = torch.zeros(SHAPE, dtype=torch.uint8), torch.ones(SHAPE, dtype=torch.uint8)
    assert not torch.allclose(_residual(T, T_out, Q, powder, 1e-5), _residual(T, T_out, Q, solid, 1e-5))


@pytest.mark.unit
def test_trapezoid_is_mean_of_explicit_and_implicit() -> None:
    T, mask, Q = _fields(seed=4)
    T_out = _solver_step(T, mask, Q, 1e-7)
    r = {d: _residual(T, T_out, Q, mask, 1e-7, diffusion=d) for d in ("explicit", "implicit", "trapezoid")}
    torch.testing.assert_close(r["trapezoid"], 0.5 * (r["explicit"] + r["implicit"]))


@pytest.mark.unit
def test_mask_out_is_used_for_the_implicit_term() -> None:
    T, _, Q = _fields(seed=5)
    T_out = T + 1.0
    powder, solid = torch.zeros(SHAPE, dtype=torch.uint8), torch.ones(SHAPE, dtype=torch.uint8)
    a = _residual(T, T_out, Q, powder, 1e-5, diffusion="implicit")
    b = _residual(T, T_out, Q, powder, 1e-5, diffusion="implicit", mask_out=solid)
    assert not torch.allclose(a, b)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"dt": 0.0}, "dt"),
        ({"storage": "bogus"}, "storage"),
        ({"diffusion": "bogus"}, "diffusion"),
        ({"T_out": torch.zeros(1, 1, 3, 3, 3, dtype=torch.float64)}, "shape"),
    ],
)
def test_invalid_arguments_are_refused(kw: dict, match: str) -> None:
    T, mask, Q = _fields()
    args = {"T_out": T.clone(), "dt": 1e-5}
    args.update(kw)
    T_out, dt = args.pop("T_out"), args.pop("dt")
    with pytest.raises(ValueError, match=match):
        _residual(T, T_out, Q, mask, dt, **args)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mat",
    [
        MAT.model_copy(update={"use_T_dep": False}),
        MAT.model_copy(update={"use_lut": False, "cp_T_coeff": 4e-4}),
        MAT.model_copy(update={"use_T_dep": False, "latent_heat_L": 0.0}),
    ],
    ids=["constant", "linear", "no-latent"],
)
def test_enthalpy_derivative_is_cp_eff_without_lut(mat: MaterialConfig) -> None:
    T = torch.tensor([300.0, 900.0, 1650.0, 1663.0, 1690.0, 3000.0], dtype=torch.float64)
    h = 1e-3
    dH = (enthalpy(T + h, mat) - enthalpy(T - h, mat)) / (2 * h)
    torch.testing.assert_close(dH, cp_eff(T, mat), rtol=1e-6, atol=1e-6)
