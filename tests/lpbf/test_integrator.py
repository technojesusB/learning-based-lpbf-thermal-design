# tests/lpbf/test_integrator.py
import pytest
import torch

from lpbf.core.config import LengthUnit, SimulationConfig
from lpbf.core.state import SimulationState
from lpbf.integrator.stepper import TimeStepper
from lpbf.physics.material import MaterialConfig


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


def test_conservation_adiabatic(sim_config, mat_config):
    """
    Test that Energy is conserved (sum of T * rho * cp) in adiabatic system (Neumann BC).
    Uses constant properties for simplicity.
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
    Simulate a single pixel cooling down linearly (forced) and check if crossing is captured.
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
    Q_cool = -cooling_rate_target * rho * cp
    Q_tensor = torch.full_like(T, Q_cool)

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
    expected_cr = cooling_rate_target

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

    sim_unstable = sim_config.model_copy(
        update={
            "length_unit": LengthUnit.MILLIMETERS,  # 1e-3 scale
            "dt_base": 1e-3,  # Unstable for 1e-3 scale!
        }
    )

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
