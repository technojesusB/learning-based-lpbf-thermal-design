"""Tests for make_coordinate_grids — TDD RED phase."""

from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.utils.units import LengthUnit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_3d_cfg(nx: int = 4, ny: int = 4, nz: int = 4) -> SimulationConfig:
    """Small 3D config."""
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


def _make_2d_cfg(nx: int = 4, ny: int = 4) -> SimulationConfig:
    """Small 2D config."""
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


# ---------------------------------------------------------------------------
# Tests: make_coordinate_grids
# ---------------------------------------------------------------------------


class TestMakeCoordinateGrids:
    """Tests for the make_coordinate_grids function."""

    def test_module_importable(self):
        """The grids module must be importable from neural_pbf.pipelines.grids."""
        from neural_pbf.pipelines import grids  # noqa: F401

        assert hasattr(grids, "make_coordinate_grids"), (
            "neural_pbf.pipelines.grids does not expose make_coordinate_grids."
        )

    def test_function_importable(self):
        """make_coordinate_grids must be importable directly."""
        from neural_pbf.pipelines.grids import make_coordinate_grids  # noqa: F401

        assert callable(make_coordinate_grids)

    def test_returns_three_tensors(self):
        """make_coordinate_grids must return a tuple of exactly 3 tensors."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg()
        result = make_coordinate_grids(cfg, device=torch.device("cpu"))

        assert isinstance(result, tuple), "Return type must be a tuple."
        assert len(result) == 3, f"Expected 3 tensors, got {len(result)}."
        for t in result:
            assert isinstance(t, torch.Tensor)

    def test_3d_shapes(self):
        """For 3D config, each grid must have shape (Nz, Ny, Nx) to match state.T."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        nx, ny, nz = 5, 6, 7
        cfg = _make_3d_cfg(nx=nx, ny=ny, nz=nz)
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        expected = (nz, ny, nx)
        assert X3.shape == expected, f"X3 shape: expected {expected}, got {X3.shape}"
        assert Y3.shape == expected, f"Y3 shape: expected {expected}, got {Y3.shape}"
        assert Z3.shape == expected, f"Z3 shape: expected {expected}, got {Z3.shape}"

    def test_device_placement(self):
        """All grids must be on the requested device (CPU)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cpu = torch.device("cpu")
        cfg = _make_3d_cfg()
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=cpu)

        for name, t in [("X3", X3), ("Y3", Y3), ("Z3", Z3)]:
            assert t.device.type == "cpu", (
                f"{name} is on {t.device}, expected cpu"
            )

    def test_x_range(self):
        """X3 values must span from 0 to Lx_m; x varies along axis 2 (width)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg(nx=5, ny=4, nz=4)
        X3, _, _ = make_coordinate_grids(cfg, device=torch.device("cpu"))

        x_vals = X3[0, 0, :]  # x varies along last axis
        assert abs(x_vals[0].item()) < 1e-7, f"X3 min is not 0: {x_vals[0].item()}"
        assert abs(x_vals[-1].item() - cfg.Lx_m) < 1e-6, (
            f"X3 max is not Lx_m={cfg.Lx_m}: {x_vals[-1].item()}"
        )

    def test_y_range(self):
        """Y3 values must span from 0 to Ly_m; y varies along axis 1 (height)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg(nx=4, ny=5, nz=4)
        _, Y3, _ = make_coordinate_grids(cfg, device=torch.device("cpu"))

        y_vals = Y3[0, :, 0]  # y varies along middle axis
        assert abs(y_vals[0].item()) < 1e-7, f"Y3 min is not 0: {y_vals[0].item()}"
        assert abs(y_vals[-1].item() - cfg.Ly_m) < 1e-6, (
            f"Y3 max is not Ly_m={cfg.Ly_m}: {y_vals[-1].item()}"
        )

    def test_z_range(self):
        """Z3 values must span from 0 to Lz_m; z varies along axis 0 (depth)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg(nx=4, ny=4, nz=6)
        _, _, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        z_vals = Z3[:, 0, 0]  # z varies along first axis
        assert abs(z_vals[0].item()) < 1e-7, f"Z3 min is not 0: {z_vals[0].item()}"
        assert abs(z_vals[-1].item() - cfg.Lz_m) < 1e-6, (
            f"Z3 max is not Lz_m={cfg.Lz_m}: {z_vals[-1].item()}"
        )

    def test_dtype_float32_default(self):
        """Default dtype must be float32."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg()
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        for name, t in [("X3", X3), ("Y3", Y3), ("Z3", Z3)]:
            assert t.dtype == torch.float32, (
                f"{name} dtype is {t.dtype}, expected float32"
            )

    def test_dtype_float64_when_specified(self):
        """When dtype=torch.float64 is passed, all grids must be float64."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg()
        X3, Y3, Z3 = make_coordinate_grids(
            cfg, device=torch.device("cpu"), dtype=torch.float64
        )

        for name, t in [("X3", X3), ("Y3", Y3), ("Z3", Z3)]:
            assert t.dtype == torch.float64, (
                f"{name} dtype is {t.dtype}, expected float64"
            )

    def test_correct_number_of_points_per_axis(self):
        """Number of unique values along each axis must equal Nx, Ny, Nz respectively."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        nx, ny, nz = 5, 6, 7
        cfg = _make_3d_cfg(nx=nx, ny=ny, nz=nz)
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        assert X3.shape[2] == nx, f"X axis should have {nx} points, got {X3.shape[2]}"
        assert Y3.shape[1] == ny, f"Y axis should have {ny} points, got {Y3.shape[1]}"
        assert Z3.shape[0] == nz, f"Z axis should have {nz} points, got {Z3.shape[0]}"

    def test_grids_are_distinct_tensors(self):
        """X3, Y3, Z3 must be distinct tensor objects (not the same memory)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_3d_cfg()
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        assert X3 is not Y3
        assert X3 is not Z3
        assert Y3 is not Z3

    def test_2d_config_raises_or_returns_degenerate(self):
        """For a 2D config (Lz=None), the function should work or raise a clear error.

        Since the training pipeline targets 3D simulations, passing a 2D config
        is an edge case. The function must either:
        - Return valid (Nx, Ny, Nz=1) grids, OR
        - Raise a ValueError with a descriptive message.
        """
        from neural_pbf.pipelines.grids import make_coordinate_grids

        cfg = _make_2d_cfg()
        # Must not silently produce wrong results
        try:
            result = make_coordinate_grids(cfg, device=torch.device("cpu"))
            assert isinstance(result, tuple)
            assert len(result) == 3
        except (ValueError, NotImplementedError):
            pass  # Also acceptable

    def test_small_grid_point_values_are_evenly_spaced(self):
        """Grid points must be linearly spaced (linspace semantics)."""
        from neural_pbf.pipelines.grids import make_coordinate_grids

        # Use nx=3 so we can easily check: 0, 0.5, 1 for Lx=1
        cfg = SimulationConfig(
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            Nx=3,
            Ny=3,
            Nz=3,
            length_unit=LengthUnit.METERS,
            dt_base=1e-5,
            T_ambient=300.0,
        )
        X3, Y3, Z3 = make_coordinate_grids(cfg, device=torch.device("cpu"))

        x_vals = X3[0, 0, :]  # x varies along last axis
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(x_vals, expected, atol=1e-6), (
            f"X3 values {x_vals.tolist()} are not evenly spaced [0, 0.5, 1.0]"
        )
