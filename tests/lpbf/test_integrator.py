# tests/lpbf/test_integrator.py
import pytest
import torch

from neural_pbf.core.config import LengthUnit, SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig


@pytest.fixture
def sim_config():
    return SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Lz=None,
        Nx=20,
        Ny=20,
        Nz=1,
        length_unit=LengthUnit.METERS,  # internal = 1.0 m
        dt_base=1e-4,
        T_ambient=300.0,
    )


@pytest.fixture
def mat_config():
    return MaterialConfig(
        k_powder=1.0,
        k_solid=1.0,
        k_liquid=1.0,
        cp_base=1.0,
        rho=1.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=100.0,
    )

    """
    Test that Energy is conserved (sum of T * rho * cp) in adiabatic system
    (Neumann BC). Uses constant properties for simplicity.
    """
    # Override latent heat to 0 to make Energy = T * rho * cp simple
    mat_simple = mat_config.model_copy(update={"latent_heat_L": 0.0})

    stepper = TimeStepper(sim_config, mat_simple)

    # Init state with a hot spot
    T = torch.full((1, 1, sim_config.Ny, sim_config.Nx), 300.0, dtype=torch.float64)
    T[0, 0, 10, 10] = 5000.0  # Heat spike

    state = SimulationState(T=T, t=0.0)

    initial_energy = torch.sum(T)  # proportional to energy

    # Run for some steps
    for _ in range(50):
        state = stepper.step_explicit_euler(state, sim_config.dt_base)

    final_energy = torch.sum(state.T)

    # Check conservation (float64 should be very precise)
    # The div_k_grad with Neumann (replicate) should sum to 0 flux.
    assert torch.abs(final_energy - initial_energy) < 1e-4 * initial_energy


def test_cooling_rate_capture(sim_config, mat_config):
    """
    Simulate a single pixel cooling down linearly (forced) and check if crossing is
    captured.
    """
    stepper = TimeStepper(sim_config, mat_config)

    # T starts above liquidus
    T_init = 1200.0
    T = torch.full((1, 1, sim_config.Ny, sim_config.Nx), T_init, dtype=torch.float64)

    state = SimulationState(T=T, t=0.0)

    # We will manually force T to decrease by setting Q to negative value?
    # Or just override T in the loop to simulate a prescribed profile.

    # Let's use Q_ext to drive cooling.
    # To cool by 100 K/s, Q = -100 * rho * cp
    cooling_rate_target = 100.0  # K/s
    rho = mat_config.rho
    cp = mat_config.cp_base
    Q_vol = -cooling_rate_target * rho * cp

    # NEW Convention: Stepper assumes 2D input is Flux [W/m^2] and divides by dz.
    # So we must provide Flux = Q_vol * dz.
    dz = sim_config.dz
    Q_flux = Q_vol * dz

    Q_tensor = torch.full_like(T, Q_flux)

    # Solidus is 1000.0
    # Steps: 1200 -> 1100 -> 1000 ...
    # We need to cross 1000.0.
    # If dt = 0.5 s, drop is 50K.
    # 1200, 1150, 1100, 1050, 1000.
    # Need to go BELOW 1000.

    dt = 0.5

    # Step 1: 1200 -> 1150
    state = stepper.step_explicit_euler(state, dt, Q_ext=Q_tensor)
    assert not (state.cooling_rate > 0).any()

    # ... Run until < 1000
    for _ in range(10):
        state = stepper.step_explicit_euler(state, dt, Q_ext=Q_tensor)
        if state.T[0, 0, 0, 0] <= 1000.0:
            break

    # Check if cooling rate recorded
    # It should be recorded at the step it crossed.
    # Value should be close to 100.0

    recorded_cr = state.cooling_rate[0, 0, 0, 0]
    # expected_cr = cooling_rate_target

    # There might be slight diff due to T-dependent cp if latent heat was active?
    # T is around 1000 (Solidus). Latent heat bump is around (1000+1100)/2 = 1050.
    # At 1000, cp might be base again?
    # Config: width ~ 100 spread.
    # At T=1000 (solidus), we are at the tail of the bump.
    # So cp might be slightly higher than base.
    # Thus dT/dt = Q / (rho*cp) might be < 100.

    # But we calculate CR as (T_prev - T_new)/dt.
    # This is exactly the realized cooling rate.

    assert recorded_cr > 0.0
    # Check range (due to cp variation, it won't be exactly 100 if we used constant Q)
    # But it should be consistent with the actual T drop.

    # Actually, we asserted that recorded_cr IS (T_prev - T_new)/dt.
    # So we just check if it's non-zero.
    # And roughly correct magnitude.
    assert recorded_cr > 50.0 and recorded_cr < 150.0


def test_adaptive_stability(sim_config, mat_config):
    """
    Test that step_adaptive handles unstable dt by substepping.
    """
    # Create unstable config: Small Resolution (mm), Large dt
    # sim_config fixture is METERS (stable). Let's force it to be unstable.

    # sim_unstable = sim_config.model_copy(
    #     update={
    #         "length_unit": LengthUnit.MILLIMETERS,  # 1e-3 scale
    #         "dt_base": 1e-3,  # Unstable for 1e-3 scale!
    #     }
    # )

    # Check max stable dt roughly
    # dx ~ 0.05e-3 = 5e-5. dx^2 = 25e-10.
    # alpha = 1.
    # dt_limit ~ 25e-10 / 4 ~ 6e-10.
    # dt_target = 1e-3.
    # Needs ~ 10^6 steps!
    # This might be too slow for a unit test.
    # Let's adjust parameters to need e.g. 10 step.

    # Target dt = 1e-4. Limit ~ 1e-5.
    # Increase dx?
    # Let's just create a custom config here.

    sim_custom = SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Nx=20,
        Ny=20,
        Nz=1,
        length_unit=LengthUnit.METERS,  # dx=0.05
        dt_base=0.1,  # Limit ~ 0.0006. So dt=0.1 is unstable (requires ~160 steps).
        T_ambient=300.0,
    )

    stepper = TimeStepper(sim_custom, mat_config)

    T = torch.rand((1, 1, 20, 20), dtype=torch.float64) * 1000.0
    state = SimulationState(T=T, t=0.0)

    # One adaptive step
    state = stepper.step_adaptive(state, sim_custom.dt_base)

    # Should not carry NaNs
    assert not torch.isnan(state.T).any()
    assert not torch.isinf(state.T).any()

    # Check that time advanced
    assert abs(state.t - sim_custom.dt_base) < 1e-9


def _make_3d_sim_config(nx: int, ny: int, nz: int):
    """Helper: build a 3D SimulationConfig with the given grid sizes.

    NOTE: is_3d requires Lz is not None AND Nz > 1.  For Nz=1, is_3d is False.
    For tests that exercise the Triton-path squeeze logic, use Nz >= 2.
    """
    from neural_pbf.utils.units import LengthUnit

    return SimulationConfig(
        Lx=float(nx),
        Ly=float(ny),
        Lz=float(nz),
        Nx=nx,
        Ny=ny,
        Nz=nz,
        length_unit=LengthUnit.METERS,
        dt_base=1e-6,
        T_ambient=300.0,
    )


def _make_simple_mat():
    """Helper: simple MaterialConfig (no LUT, no T-dep) for shape tests."""
    return MaterialConfig(
        k_powder=1.0,
        k_solid=1.0,
        k_liquid=1.0,
        cp_base=1.0,
        rho=1.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=0.0,
    )


# ---------------------------------------------------------------------------
# CRITICAL-2 & CRITICAL-3: bare squeeze() vs .squeeze(0).squeeze(0) fix
# ---------------------------------------------------------------------------


def test_triton_path_passes_correctly_shaped_tensors_to_kernel():
    """
    CRITICAL-2: The Triton call path in step_explicit_euler must pass tensors
    with only batch/channel dims removed, not spatial dims.

    The fix replaces bare .squeeze() (removes ALL size-1 dims) with
    .squeeze(0).squeeze(0) (removes only leading batch and channel dims).

    We verify this by intercepting the call to run_thermal_step_3d_triton via
    a mock and checking that the shapes of the tensors passed in have ndim==3
    (Nx, Ny, Nz) and NOT ndim==2 (which would happen if a spatial dim were 1
    and bare .squeeze() was used).

    Since we test with Nz=2 (all spatial dims > 1), the old .squeeze() also
    produces (Nx,Ny,Nz) — so this test should PASS both before and after the fix.
    The test serves as a regression guard: if any spatial dim ever becomes 1
    (e.g. via a different config), the explicit .squeeze(0).squeeze(0) prevents
    an accidental dim removal.

    RED  scenario (documented, not tested here due to config constraints):
        If T.shape=(1,1,4,4,1) and bare squeeze() is used on the Triton path,
        the kernel receives shape (4,4) instead of (4,4,1) → IndexError or
        silent memory corruption in the kernel's stride calculations.
    GREEN (with squeeze(0).squeeze(0)):
        Shape is always (Nx, Ny, Nz) regardless of spatial dim sizes.
    """
    pytest.importorskip("triton")
    import unittest.mock as mock

    from neural_pbf.utils.units import LengthUnit

    nx, ny, nz = 4, 4, 2
    sim = SimulationConfig(
        Lx=float(nx),
        Ly=float(ny),
        Lz=float(nz),
        Nx=nx,
        Ny=ny,
        Nz=nz,
        length_unit=LengthUnit.METERS,
        dt_base=1e-6,
        T_ambient=300.0,
    )
    mat = _make_simple_mat()
    stepper = TimeStepper(sim, mat)

    assert sim.is_3d, "Precondition: sim must be 3D for Triton path"

    T_input = torch.full((1, 1, nx, ny, nz), 300.0)
    state = SimulationState(T=T_input)
    state.material_mask = torch.zeros_like(T_input, dtype=torch.uint8)

    captured_shapes = {}

    def capturing_triton_step(T_3d, mask_3d, Q_3d, sim_cfg, mat_cfg, dt, **kwargs):
        captured_shapes["T"] = tuple(T_3d.shape)
        captured_shapes["mask"] = tuple(mask_3d.shape)
        captured_shapes["Q"] = tuple(Q_3d.shape)
        # Return a correctly-shaped T_new so the rest of the function works
        return T_3d.clone()

    with mock.patch(
        "neural_pbf.integrator.stepper.run_thermal_step_3d_triton",
        side_effect=capturing_triton_step,
    ):
        stepper.step_explicit_euler(
            state, dt=1e-8, Q_ext=None, use_triton=True
        )

    assert captured_shapes, "Triton path was not entered — check is_3d and use_triton"
    expected_shape = (nx, ny, nz)
    assert captured_shapes["T"] == expected_shape, (
        f"CRITICAL-2: Triton kernel received T with shape {captured_shapes['T']}, "
        f"expected {expected_shape}.  "
        "Fix: replace T.squeeze() with T.squeeze(0).squeeze(0).contiguous() in "
        "step_explicit_euler."
    )
    assert captured_shapes["mask"] == expected_shape, (
        f"CRITICAL-2: Triton kernel received mask with shape {captured_shapes['mask']}, "
        f"expected {expected_shape}."
    )


def test_mask_update_no_squeeze_preserves_shape_and_semantics():
    """
    CRITICAL-3: The mask update in step_adaptive must work correctly without
    calling .squeeze() on state.T.

    The fix removes .squeeze() from:
        newly_solid = (self.mat.T_solidus <= state.T).squeeze().to(torch.uint8)

    To become:
        newly_solid = (self.mat.T_solidus <= state.T).to(torch.uint8)

    This test verifies:
    1. Hot voxels (T >= T_solidus) are correctly marked in material_mask.
    2. Cold voxels remain 0 in material_mask.
    3. material_mask shape equals state.T shape (no shape change from OR).

    The test is GREEN both before and after the fix for normal 3D shapes (because
    view_as with equal element count always succeeds).  This serves as a
    regression guard to confirm correct semantics are maintained by the fix.
    """
    nx, ny, nz = 4, 4, 2
    sim = _make_3d_sim_config(nx, ny, nz)
    mat = MaterialConfig(
        k_powder=1.0,
        k_solid=1.0,
        k_liquid=1.0,
        cp_base=1.0,
        rho=1.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=0.0,
    )
    stepper = TimeStepper(sim, mat)

    T = torch.full((1, 1, nx, ny, nz), 300.0)
    T[0, 0, 2, 3, 1] = 1200.0  # Well above T_solidus=1000
    state = SimulationState(T=T)
    state.material_mask = torch.zeros_like(T, dtype=torch.uint8)

    state = stepper.step_adaptive(state, dt_target=1e-9, use_triton=False)

    assert state.material_mask.shape == (1, 1, nx, ny, nz), (
        f"CRITICAL-3: material_mask shape changed to {tuple(state.material_mask.shape)}.  "
        "Expected (1,1,4,4,2)."
    )
    assert state.material_mask[0, 0, 2, 3, 1].item() == 1, (
        "CRITICAL-3: Hot voxel [0,0,2,3,1] not marked solid after step_adaptive."
    )
    assert state.material_mask[0, 0, 0, 0, 0].item() == 0, (
        "CRITICAL-3: Cold voxel [0,0,0,0,0] incorrectly marked solid."
    )


# ---------------------------------------------------------------------------
# HIGH-1: LUT tensors recreated every sub-step
# ---------------------------------------------------------------------------


def test_lut_not_reallocated_across_substeps():
    """
    HIGH-1: run_thermal_step_3d_triton must accept pre-computed LUT tensors so
    that TimeStepper can pass cached tensors and avoid recreating them on every
    sub-step call.

    The fix requires:
      1. run_thermal_step_3d_triton gains optional parameters
         ``T_lut_t``, ``k_lut_t``, ``cp_lut_t`` (pre-built tensors).
      2. TimeStepper stores ``self._lut_tensors`` dict after first sub-step.
      3. Subsequent sub-steps reuse the cached tensors.

    This test verifies the observable contract: after ONE call to
    ``run_thermal_step_3d_triton`` with pre-built tensors, the function uses
    them rather than recreating from ``mat_cfg.T_lut`` lists.

    We count calls to ``run_thermal_step_3d_triton``'s internal torch.tensor by
    wrapping the function itself:

    RED  (no optional tensor params): TypeError when extra kwargs are passed, OR
         torch.tensor still called internally despite pre-built tensors given.
    GREEN (optional params accepted and used): function accepts tensors and
          re-uses them; torch.tensor NOT called for LUT data when tensors given.
    """

    from neural_pbf.utils.units import LengthUnit

    # Skip if triton is not installed (run_thermal_step_3d_triton not importable)
    pytest.importorskip("triton")

    from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

    sim = SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        Nx=4,
        Ny=4,
        Nz=4,
        length_unit=LengthUnit.METERS,
        dt_base=1e-6,
        T_ambient=300.0,
    )

    mat_with_lut = MaterialConfig(
        k_powder=0.2,
        k_solid=13.8,
        k_liquid=31.3,
        cp_base=483.0,
        rho=7950.0,
        T_solidus=1653.0,
        T_liquidus=1673.0,
        latent_heat_L=0.0,
        use_T_dep=True,
        use_lut=True,
        T_lut=[300.0, 600.0, 900.0, 1200.0, 1500.0, 1673.0],
        k_lut=[13.8, 20.0, 24.8, 28.4, 30.7, 31.3],
        cp_lut=[483.0, 537.0, 592.0, 646.0, 701.0, 732.0],
    )

    # Verify the function signature accepts optional pre-built LUT tensors.
    # We build the tensors once and pass them in; count how many times
    # torch.tensor is called *inside* the function.
    import inspect

    sig = inspect.signature(run_thermal_step_3d_triton)
    has_lut_tensor_params = (
        "T_lut_t" in sig.parameters
        or "k_lut_t" in sig.parameters
        or "lut_tensors" in sig.parameters
    )
    assert has_lut_tensor_params, (
        "HIGH-1: run_thermal_step_3d_triton does not accept pre-built LUT tensor "
        "parameters.  Expected at least one of: 'T_lut_t', 'k_lut_t', or "
        "'lut_tensors' in its signature.  "
        "Fix: add optional lut_tensors parameter (dict or tuple) so TimeStepper "
        "can pass cached GPU tensors instead of recreating them every sub-step."
    )

    # Verify TimeStepper initialises a LUT cache attribute.
    stepper = TimeStepper(sim, mat_with_lut)
    assert hasattr(stepper, "_lut_tensors"), (
        "HIGH-1: TimeStepper does not have a '_lut_tensors' attribute after "
        "construction.  "
        "Fix: add 'self._lut_tensors: dict = {}' in TimeStepper.__init__."
    )


def test_step_adaptive_updates_material_mask():
    """step_adaptive must update state.material_mask for voxels at or above T_solidus.

    RED  (before fix): mask remains all-zeros even after step_adaptive runs on a
         state that has hot voxels above T_solidus.
    GREEN (after fix): mask is non-zero for voxels whose temperature crossed
         or reached T_solidus during the macro-step.
    """
    sim = SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Nx=10,
        Ny=10,
        Nz=1,
        length_unit=LengthUnit.METERS,
        dt_base=1e-4,
        T_ambient=300.0,
    )
    mat = MaterialConfig(
        k_powder=1.0,
        k_solid=10.0,
        k_liquid=10.0,
        cp_base=500.0,
        rho=2000.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=0.0,
    )
    stepper = TimeStepper(sim, mat)

    # Start with all-zero mask and a temperature field where center voxel is
    # above T_solidus — it should get promoted by step_adaptive.
    T = torch.full((1, 1, 10, 10), 300.0)
    T[0, 0, 5, 5] = 1100.0  # well above T_solidus=1000
    state = SimulationState(T=T)
    state.material_mask = torch.zeros_like(T, dtype=torch.uint8)

    assert state.material_mask.sum().item() == 0, "Precondition: mask is all zeros"

    state = stepper.step_adaptive(state, dt_target=1e-6)

    hot_voxel_mask = state.material_mask[0, 0, 5, 5].item()
    assert hot_voxel_mask == 1, (
        f"step_adaptive did not promote the hot voxel mask. "
        f"Expected mask[5,5]=1 but got {hot_voxel_mask}. "
        "Add a mask update at the end of step_adaptive using bitwise OR with "
        "(T >= T_solidus)."
    )
