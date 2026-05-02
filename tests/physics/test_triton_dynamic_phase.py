"""
Tests for dynamic phase transition overrides in the Triton GPU kernel.

These tests verify that the kernel uses solid material properties (mask=1)
whenever a voxel's temperature exceeds T_solidus, even when the persistent
mask buffer reports that voxel as powder (mask=0). All tests are GPU-gated.

TDD semantics:
    RED  — tests fail before the 7 tl.where overrides are added to the kernel.
    GREEN — tests pass after the overrides are in place.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mat_cfg():
    """
    Simple MaterialConfig with a large conductivity contrast between powder and
    solid, no LUT, no temperature dependence. This maximises the numerical
    signature of the override.
    """
    from neural_pbf.physics.material import MaterialConfig

    return MaterialConfig(
        k_powder=0.1,  # very low — clearly different from solid
        k_solid=15.0,
        k_liquid=20.0,
        cp_base=500.0,
        rho=8000.0,
        T_solidus=1000.0,
        T_liquidus=1050.0,
        latent_heat_L=0.0,  # suppress latent heat to keep arithmetic simple
        transition_sharpness=5.0,
        use_T_dep=False,
        use_lut=False,
    )


@pytest.fixture
def sim_cfg():
    """SimulationConfig for an 8x8x8 domain (0.8 mm cube)."""
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
# Helpers
# ---------------------------------------------------------------------------

# A timestep large enough to make conductivity differences visible in float32
_DT_AMPLIFIED = 1.0e-5


def _device() -> "torch.device":
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDynamicPhaseOverride:
    """Dynamic phase transition override tests — require CUDA + Triton."""

    @pytest.mark.unit
    def test_hot_powder_uses_solid_properties(self, mat_cfg, sim_cfg):
        """
        A voxel at T > T_solidus with mask=0 (powder) must produce the same
        T_new as when its mask is explicitly set to 1 (solid), because the
        register-only override inside the kernel promotes the voxel to solid
        thermal properties for that sub-step.

        RED  (no override): run with all-powder mask differs from run with
             hot voxel set solid — torch.equal returns False.
        GREEN (with override): both runs are bit-identical — torch.equal True.
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        device = _device()
        nx, ny, nz = 8, 8, 8
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        T_hot = mat_cfg.T_solidus + 100.0  # well above solidus
        T_cold = mat_cfg.T_solidus - 200.0

        T = torch.full((nx, ny, nz), T_cold, dtype=torch.float32, device=device)
        T[cx, cy, cz] = T_hot

        mask_powder = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_solid_center = mask_powder.clone()
        mask_solid_center[cx, cy, cz] = 1  # explicit solid for the hot voxel

        Q = torch.zeros_like(T)

        T_new_powder = run_thermal_step_3d_triton(
            T, mask_powder, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )
        T_new_solid = run_thermal_step_3d_triton(
            T, mask_solid_center, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )

        assert torch.equal(T_new_powder, T_new_solid), (
            "Hot powder voxel did not use solid properties — dynamic override missing. "
            f"Max diff: {(T_new_powder - T_new_solid).abs().max().item():.6e}. "
            "Add tl.where(Tc > T_sol, 1.0, mc) after each mask load in the kernel."
        )

    @pytest.mark.unit
    def test_cold_powder_keeps_powder_properties(self, mat_cfg, sim_cfg):
        """
        A voxel at T < T_solidus with mask=0 (powder) must produce a DIFFERENT
        T_new than when its mask is set to 1 (solid), because cold powder and cold
        solid have very different conductivities (0.1 vs 15.0 W/(m K)).

        This confirms the override is conditional on temperature — it must NOT fire
        for sub-solidus voxels in either RED or GREEN state.
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        device = _device()
        nx, ny, nz = 8, 8, 8
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        # Linear temperature gradient — all voxels well below T_solidus
        T_lo, T_hi = 600.0, mat_cfg.T_solidus - 150.0  # max = 850 K < 1000 K
        T = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
        for i in range(nx):
            T[i, :, :] = T_lo + i * (T_hi - T_lo) / (nx - 1)

        mask_powder = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_solid_center = mask_powder.clone()
        mask_solid_center[cx, cy, cz] = 1

        Q = torch.zeros_like(T)

        T_new_powder = run_thermal_step_3d_triton(
            T, mask_powder, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )
        T_new_solid = run_thermal_step_3d_triton(
            T, mask_solid_center, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )

        assert not torch.equal(T_new_powder, T_new_solid), (
            "Cold powder and cold solid produced identical T_new — "
            "the override must be firing incorrectly for sub-solidus temperatures."
        )

    @pytest.mark.unit
    def test_exactly_solidus_uses_solid_properties(self, mat_cfg, sim_cfg):
        """
        A voxel at T == T_solidus exactly must use solid properties (not powder).

        The external mask promotion in step_adaptive uses >= T_solidus (inclusive),
        so the kernel override must also use >= T_sol (not strictly >).

        RED  (before fix, using Tc > T_sol): T_new for mask=0 at solidus differs
             from T_new for mask=1 at solidus.
        GREEN (after fix, using Tc >= T_sol): both runs are bit-identical.
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        device = _device()
        nx, ny, nz = 8, 8, 8
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        T_solidus = mat_cfg.T_solidus  # exactly at the boundary
        T_cold = mat_cfg.T_solidus - 200.0

        T = torch.full((nx, ny, nz), T_cold, dtype=torch.float32, device=device)
        T[cx, cy, cz] = T_solidus  # exactly at solidus, not above

        mask_powder = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_solid_center = mask_powder.clone()
        mask_solid_center[cx, cy, cz] = 1  # explicit solid for the boundary voxel

        Q = torch.zeros_like(T)

        T_new_powder = run_thermal_step_3d_triton(
            T, mask_powder, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )
        T_new_solid = run_thermal_step_3d_triton(
            T, mask_solid_center, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED
        )

        assert torch.equal(T_new_powder, T_new_solid), (
            "Voxel at exactly T_solidus did not use solid properties — "
            "the override must use >= T_sol (not strictly >). "
            f"Max diff: {(T_new_powder - T_new_solid).abs().max().item():.6e}. "
            "Change tl.where(Tc > T_sol, ...) to tl.where(Tc >= T_sol, ...) in kernel."
        )

    @pytest.mark.unit
    def test_mask_buffer_not_mutated(self, mat_cfg, sim_cfg):
        """
        The persistent mask buffer passed to run_thermal_step_3d_triton must be
        byte-for-byte unchanged after the call. The dynamic phase promotion is
        register-only; updating the persistent mask is the caller's responsibility.
        """
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

        device = _device()
        nx, ny, nz = 8, 8, 8
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        T_hot = mat_cfg.T_solidus + 100.0
        T_cold = mat_cfg.T_solidus - 200.0

        T = torch.full((nx, ny, nz), T_cold, dtype=torch.float32, device=device)
        T[cx, cy, cz] = T_hot

        mask = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=device)
        mask_before = mask.clone()

        Q = torch.zeros_like(T)

        run_thermal_step_3d_triton(T, mask, Q, sim_cfg, mat_cfg, _DT_AMPLIFIED)

        assert torch.equal(mask, mask_before), (
            "run_thermal_step_3d_triton mutated the input mask buffer. "
            "Phase overrides must be register-only inside the kernel."
        )
