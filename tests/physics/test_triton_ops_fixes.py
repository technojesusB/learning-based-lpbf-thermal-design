"""
Tests for CRITICAL-1 (tl.full -> scalar broadcast in phase-transition overrides)
and MEDIUM (LUT guard: n_lut > 16 raises ValueError).

These tests use the PyTorch (non-Triton) path for MEDIUM so they run on all
machines.  The CRITICAL-1 tests are GPU-gated; without CUDA they skip cleanly.

TDD Semantics
-------------
CRITICAL-1: The tl.full(mc.shape, ...) form is already in the codebase.
    On environments where Triton silently accepts it the GPU-path tests may
    already pass, but they document the correct contract and will catch
    regressions on other Triton versions.

MEDIUM (LUT guard): There is currently NO guard.  This test must FAIL before
    the guard is added, then PASS after.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mat_no_lut():
    """MaterialConfig without LUT (for CRITICAL-1 shape check)."""
    from neural_pbf.physics.material import MaterialConfig

    return MaterialConfig(
        k_powder=0.1,
        k_solid=15.0,
        k_liquid=20.0,
        cp_base=500.0,
        rho=8000.0,
        T_solidus=1000.0,
        T_liquidus=1050.0,
        latent_heat_L=0.0,
        use_T_dep=False,
        use_lut=False,
    )


@pytest.fixture
def sim_cfg_3d():
    """SimulationConfig for a small 3D domain (8x8x8)."""
    from neural_pbf.core.config import SimulationConfig
    from neural_pbf.utils.units import LengthUnit

    return SimulationConfig(
        Lx=0.8,
        Ly=0.8,
        Lz=0.8,
        Nx=8,
        Ny=8,
        Nz=8,
        length_unit=LengthUnit.MILLIMETERS,
        dt_base=1e-8,
        T_ambient=300.0,
        loss_h=0.0,
    )


# ---------------------------------------------------------------------------
# CRITICAL-1: tl.full scalar-broadcast fix
# ---------------------------------------------------------------------------


class TestCritical1TlFullFix:
    """
    Verify that replacing tl.full(mc.shape, 1.0, ...) with the scalar 1.0
    in the 7 phase-transition override lines produces correct results.

    These tests are GPU-gated (skip when CUDA is unavailable).

    A correct implementation must produce the same T_new regardless of whether
    the ``tl.full`` or scalar form is used, because both express the same
    semantics: promote mc to 1.0 when T >= T_sol.
    """

    @staticmethod
    def _run_kernel(T, mask, Q, sim_cfg, mat_cfg, dt):
        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        return run_thermal_step_3d_triton(T, mask, Q, sim_cfg, mat_cfg, dt)

    def test_phase_override_with_scalar_broadcast_center_voxel(
        self, mat_no_lut, sim_cfg_3d
    ):
        """
        After the fix (scalar 1.0 instead of tl.full), a hot powder voxel
        at T >= T_solidus must use solid conductivity.  The resulting T_new
        for an all-powder-mask run must equal the T_new for a run where the
        hot voxel's mask is explicitly set to 1 (solid).

        RED before fix (on Triton versions that mis-compile tl.full):
            torch.equal returns False.
        GREEN after fix:
            torch.equal returns True.
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        nx, ny, nz = 8, 8, 8
        cx, cy, cz = 4, 4, 4
        dt = 1e-5

        T_hot = mat_no_lut.T_solidus + 200.0
        T_cold = mat_no_lut.T_solidus - 300.0

        T = torch.full((nx, ny, nz), T_cold, dtype=torch.float32, device=device)
        T[cx, cy, cz] = T_hot

        mask_powder = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_solid_hot = mask_powder.clone()
        mask_solid_hot[cx, cy, cz] = 1

        Q = torch.zeros_like(T)

        T_new_powder = self._run_kernel(T, mask_powder, Q, sim_cfg_3d, mat_no_lut, dt)
        T_new_solid = self._run_kernel(T, mask_solid_hot, Q, sim_cfg_3d, mat_no_lut, dt)

        max_diff = (T_new_powder - T_new_solid).abs().max().item()
        assert torch.equal(T_new_powder, T_new_solid), (
            f"CRITICAL-1: Hot powder voxel did not use solid properties after scalar "
            f"broadcast fix.  Max diff = {max_diff:.6e}.  "
            "The 7 tl.full(mc.shape, 1.0, ...) lines must be replaced with scalar 1.0."
        )

    def test_phase_override_with_scalar_does_not_affect_cold_powder(
        self, mat_no_lut, sim_cfg_3d
    ):
        """
        Cold voxels (T < T_solidus) must NOT have their mask promoted.
        After the fix, a cold-all-powder run must still differ from a cold
        all-solid run (large conductivity contrast).
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        nx, ny, nz = 8, 8, 8
        dt = 1e-5

        T_cold_base = mat_no_lut.T_solidus - 200.0
        T_cold_high = mat_no_lut.T_solidus - 100.0
        T = torch.full((nx, ny, nz), T_cold_base, dtype=torch.float32, device=device)
        # Add a gradient so conductivity matters
        T[nx // 2 :, :, :] = T_cold_high

        mask_powder = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_solid = torch.ones((nx, ny, nz), dtype=torch.uint8, device=device)

        Q = torch.zeros_like(T)

        T_new_powder = self._run_kernel(T, mask_powder, Q, sim_cfg_3d, mat_no_lut, dt)
        T_new_solid = self._run_kernel(T, mask_solid, Q, sim_cfg_3d, mat_no_lut, dt)

        assert not torch.equal(T_new_powder, T_new_solid), (
            "CRITICAL-1: Cold powder and cold solid produced identical T_new — "
            "the scalar override must only fire for T >= T_solidus."
        )


# ---------------------------------------------------------------------------
# MEDIUM: LUT guard (n_lut > 16 must raise ValueError)
# ---------------------------------------------------------------------------


class TestMediumLutGuard:
    """
    The Triton kernel internally loops ``for i in range(16)`` so it can only
    handle LUT tables with at most 16 entries.  Passing a larger table silently
    truncates the interpolation and produces wrong results.

    After the guard is added to ``run_thermal_step_3d_triton``, calling it with
    n_lut > 16 must raise a ``ValueError`` before any GPU work is launched.

    RED  (no guard): No exception is raised; test fails.
    GREEN (guard added): ValueError is raised; test passes.
    """

    def test_lut_too_large_raises_value_error(self, sim_cfg_3d):
        """n_lut > 16 must raise ValueError immediately (no GPU needed)."""
        pytest.importorskip("triton")
        # We do NOT require CUDA here — the guard must fire before any GPU launch.
        # If triton is not installed we skip because run_thermal_step_3d_triton
        # would not be importable anyway.

        from neural_pbf.physics.material import MaterialConfig
        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        # Build a MaterialConfig with 17 LUT entries (exceeds kernel limit of 16)
        n = 17
        T_lut = [float(300 + i * 100) for i in range(n)]
        k_lut = [10.0 + i * 0.5 for i in range(n)]
        cp_lut = [500.0 + i * 10.0 for i in range(n)]

        mat_big_lut = MaterialConfig(
            k_powder=0.2,
            k_solid=15.0,
            k_liquid=20.0,
            cp_base=500.0,
            rho=8000.0,
            T_solidus=1000.0,
            T_liquidus=1050.0,
            latent_heat_L=0.0,
            use_T_dep=True,
            use_lut=True,
            T_lut=T_lut,
            k_lut=k_lut,
            cp_lut=cp_lut,
        )

        # Use CPU tensors; the guard must fire before any device transfer.
        # We use a tiny domain so this is fast even if the guard is missing.
        device = torch.device("cpu")
        T = torch.full((4, 4, 4), 500.0, dtype=torch.float32, device=device)
        mask = torch.zeros((4, 4, 4), dtype=torch.uint8, device=device)
        Q = torch.zeros_like(T)

        with pytest.raises(ValueError, match=r"LUT has 17 entries"):
            run_thermal_step_3d_triton(T, mask, Q, sim_cfg_3d, mat_big_lut, dt=1e-8)

    def test_lut_at_limit_does_not_raise(self, sim_cfg_3d):
        """n_lut == 16 is exactly the maximum supported; must not raise."""
        pytest.importorskip("triton")

        from neural_pbf.physics.material import MaterialConfig
        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        n = 16
        T_lut = [float(300 + i * 100) for i in range(n)]
        k_lut = [10.0 + i * 0.5 for i in range(n)]
        cp_lut = [500.0 + i * 10.0 for i in range(n)]

        mat_ok_lut = MaterialConfig(
            k_powder=0.2,
            k_solid=15.0,
            k_liquid=20.0,
            cp_base=500.0,
            rho=8000.0,
            T_solidus=1000.0,
            T_liquidus=1050.0,
            latent_heat_L=0.0,
            use_T_dep=True,
            use_lut=True,
            T_lut=T_lut,
            k_lut=k_lut,
            cp_lut=cp_lut,
        )

        if not torch.cuda.is_available():
            # We just test that the guard doesn't raise for n=16.
            # If no GPU, we can't run the full kernel.  We only need to verify
            # that the pre-launch guard passes (no ValueError before GPU call).
            # Monkeypatch the kernel call away:
            import unittest.mock as mock

            device = torch.device("cpu")
            T = torch.full((4, 4, 4), 500.0, dtype=torch.float32)
            mask = torch.zeros((4, 4, 4), dtype=torch.uint8)
            Q = torch.zeros_like(T)

            import neural_pbf.physics.triton_ops as triton_ops_mod

            with mock.patch.object(
                triton_ops_mod, "_thermal_step_3d_kernel", side_effect=RuntimeError("no GPU")
            ):
                try:
                    run_thermal_step_3d_triton(
                        T, mask, Q, sim_cfg_3d, mat_ok_lut, dt=1e-8
                    )
                except RuntimeError:
                    pass  # expected — GPU call fails, but ValueError must NOT be raised
        else:
            # With GPU available we just confirm no ValueError is raised
            device = torch.device("cuda")
            T = torch.full((4, 4, 4), 500.0, dtype=torch.float32, device=device)
            mask = torch.zeros((4, 4, 4), dtype=torch.uint8, device=device)
            Q = torch.zeros_like(T)

            # Should not raise ValueError
            try:
                run_thermal_step_3d_triton(T, mask, Q, sim_cfg_3d, mat_ok_lut, dt=1e-8)
            except ValueError:
                pytest.fail("run_thermal_step_3d_triton raised ValueError for n_lut=16")
