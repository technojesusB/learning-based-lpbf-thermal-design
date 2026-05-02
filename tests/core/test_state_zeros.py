"""Tests for SimulationState.zeros classmethod — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.utils.units import LengthUnit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2d_cfg(nx: int = 4, ny: int = 4) -> SimulationConfig:
    """Small 2D SimulationConfig (Lz=None → is_3d=False)."""
    return SimulationConfig(
        Lx=float(nx),
        Ly=float(ny),
        Lz=None,
        Nx=nx,
        Ny=ny,
        Nz=1,
        length_unit=LengthUnit.METERS,
        dt_base=1e-5,
        T_ambient=300.0,
    )


def _make_3d_cfg(nx: int = 4, ny: int = 4, nz: int = 4) -> SimulationConfig:
    """Small 3D SimulationConfig (Lz is not None, Nz > 1 → is_3d=True)."""
    return SimulationConfig(
        Lx=float(nx),
        Ly=float(ny),
        Lz=float(nz),
        Nx=nx,
        Ny=ny,
        Nz=nz,
        length_unit=LengthUnit.METERS,
        dt_base=1e-5,
        T_ambient=300.0,
    )


# ---------------------------------------------------------------------------
# Tests: SimulationState.zeros classmethod
# ---------------------------------------------------------------------------


class TestSimulationStateZeros:
    """Tests for the SimulationState.zeros factory classmethod."""

    def test_classmethod_exists(self):
        """SimulationState must expose a 'zeros' classmethod."""
        assert hasattr(SimulationState, "zeros"), (
            "SimulationState.zeros classmethod does not exist. "
            "Add @classmethod def zeros(cls, sim_cfg, device, dtype, T_initial)."
        )
        assert callable(SimulationState.zeros)

    def test_3d_shape(self):
        """3D config: T shape must be (1, 1, Nz, Ny, Nx)."""
        cfg = _make_3d_cfg(nx=4, ny=4, nz=4)
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        expected_shape = (1, 1, cfg.Nz, cfg.Ny, cfg.Nx)
        assert state.T.shape == expected_shape, (
            f"3D T shape mismatch: expected {expected_shape}, got {state.T.shape}"
        )

    def test_2d_shape(self):
        """2D config: T shape must be (1, 1, Ny, Nx)."""
        cfg = _make_2d_cfg(nx=4, ny=4)
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        expected_shape = (1, 1, cfg.Ny, cfg.Nx)
        assert state.T.shape == expected_shape, (
            f"2D T shape mismatch: expected {expected_shape}, got {state.T.shape}"
        )

    def test_default_temperature_is_T_ambient(self):
        """When T_initial is None, all voxels must be filled with sim_cfg.T_ambient."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert torch.all(cfg.T_ambient == state.T), (
            f"T values not equal to T_ambient={cfg.T_ambient}. "
            "Use T_ambient as the fill value when T_initial is None."
        )

    def test_custom_T_initial_fills_T(self):
        """When T_initial is provided, all voxels must be filled with T_initial."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        T_init = 1500.0
        state = SimulationState.zeros(cfg, device=device, T_initial=T_init)

        assert torch.all(T_init == state.T), (
            f"T values not equal to T_initial={T_init}."
        )

    def test_device_placement(self):
        """All tensors in the returned state must be on the specified device."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert state.T.device.type == device.type
        assert state.max_T is not None and state.max_T.device.type == device.type
        assert state.cooling_rate is not None and state.cooling_rate.device.type == device.type
        assert state.material_mask is not None and state.material_mask.device.type == device.type

    def test_max_T_equals_T(self):
        """max_T must be initialized as a clone of T (same values, different object)."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert state.max_T is not None
        assert torch.equal(state.T, state.max_T), "max_T must equal T on construction."
        # Must be a distinct object (clone)
        assert state.T is not state.max_T, "max_T must be a clone, not the same tensor."

    def test_cooling_rate_is_zeros(self):
        """cooling_rate must be all zeros after construction."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert state.cooling_rate is not None
        assert torch.all(state.cooling_rate == 0.0), (
            "cooling_rate must be all-zero on construction."
        )
        assert state.cooling_rate.shape == state.T.shape

    def test_material_mask_is_zeros_uint8(self):
        """material_mask must be all zeros with dtype=torch.uint8."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert state.material_mask is not None
        assert state.material_mask.dtype == torch.uint8, (
            f"material_mask dtype must be uint8, got {state.material_mask.dtype}"
        )
        assert torch.all(state.material_mask == 0), "material_mask must be all zeros."
        assert state.material_mask.shape == state.T.shape

    def test_dtype_float32_default(self):
        """Default dtype for T, max_T, cooling_rate must be float32."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        assert state.T.dtype == torch.float32
        assert state.max_T is not None and state.max_T.dtype == torch.float32
        assert state.cooling_rate is not None and state.cooling_rate.dtype == torch.float32

    def test_dtype_float64_when_specified(self):
        """When dtype=torch.float64 is passed, T must be float64."""
        cfg = _make_3d_cfg()
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device, dtype=torch.float64)

        assert state.T.dtype == torch.float64, (
            f"Expected float64, got {state.T.dtype}"
        )

    def test_t_is_zero(self):
        """Initial simulation time t must be 0.0."""
        cfg = _make_3d_cfg()
        state = SimulationState.zeros(cfg, device=torch.device("cpu"))
        assert state.t == 0.0

    def test_step_is_zero(self):
        """Initial step counter must be 0."""
        cfg = _make_3d_cfg()
        state = SimulationState.zeros(cfg, device=torch.device("cpu"))
        assert state.step == 0

    def test_returns_simulation_state_instance(self):
        """Return type must be SimulationState."""
        cfg = _make_3d_cfg()
        state = SimulationState.zeros(cfg, device=torch.device("cpu"))
        assert isinstance(state, SimulationState)

    def test_2d_all_fields_correct_shape(self):
        """For 2D configs, all auxiliary fields must match (1,1,Ny,Nx) shape."""
        cfg = _make_2d_cfg(nx=6, ny=8)
        device = torch.device("cpu")
        state = SimulationState.zeros(cfg, device=device)

        expected = (1, 1, cfg.Ny, cfg.Nx)
        assert state.T.shape == expected
        assert state.max_T is not None and state.max_T.shape == expected
        assert state.cooling_rate is not None and state.cooling_rate.shape == expected
        assert state.material_mask is not None and state.material_mask.shape == expected

    def test_3d_uses_nz_not_lz(self):
        """3D shape must use Nz (grid points), not Lz (physical length)."""
        cfg = _make_3d_cfg(nx=5, ny=6, nz=7)
        state = SimulationState.zeros(cfg, device=torch.device("cpu"))
        assert state.T.shape == (1, 1, 7, 6, 5), (
            f"Shape should use (Nz=7, Ny=6, Nx=5), got {state.T.shape}"
        )
