# tests/lpbf/test_units_and_physics.py
import math

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.diagnostics.energy import EnergyMonitor
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.ops import div_k_grad
from neural_pbf.utils.units import LengthUnit


@pytest.fixture
def basic_configs():
    sim = SimulationConfig(
        Lx=1.0, Ly=1.0, Nx=11, Ny=11, length_unit=LengthUnit.MILLIMETERS,
        dt_base=0.001, default_dz=0.1 # 100 microns
    )
    mat = MaterialConfig(
        k_powder=10.0, k_solid=10.0, k_liquid=10.0,
        cp_base=500.0, rho=2000.0,
        T_solidus=1000.0, T_liquidus=1100.0, latent_heat_L=200000.0
    )
    return sim, mat

def test_config_dz_conversion(basic_configs):
    sim, _ = basic_configs
    # Lx = 1.0 mm -> 0.001 m
    assert math.isclose(sim.Lx_m, 0.001)
    
    # dz defaults. default_dz=0.1 mm -> 0.0001 m
    assert sim.is_3d is False
    assert math.isclose(sim.dz, 0.0001)

def test_2d_source_normalization(basic_configs):
    """
    Verify that in 2D mode, a surface flux Q [W/m^2] is normalized by dz to get [W/m^3].
    """
    sim, mat = basic_configs
    stepper = TimeStepper(sim, mat)
    
    # State with uniform T
    T = torch.full((1, 1, sim.Ny, sim.Nx), 300.0)
    state = SimulationState(T=T)
    
    # Q_ext as uniform Surface Flux of 1.0e5 W/m^2 (LARGE enough for float32 visibility)
    Q_flux = torch.ones_like(T) * 1.0e5
    
    # Step simulation with Q_ext
    # Expected behavior:
    # Q_vol = 1e5 / 0.0001 = 1e9 W/m^3
    # dT/dt = 1e9 / (2000 * 500) = 1000 K/s
    # dt = 0.001 s
    # delta_T = 1.0 K
    
    state_next = stepper.step_explicit_euler(state, dt=sim.dt_base, Q_ext=Q_flux)
    
    T_new = state_next.T
    delta_T = T_new - T
    
    assert torch.allclose(delta_T, torch.tensor(1.0), atol=1e-3)

def test_energy_conservation(basic_configs):
    """
    Conservation check: Heat up a closed box.
    Input Energy [J] should equal Change in Enthalpy [J].
    """
    sim, mat = basic_configs
    # Disable loss for pure conservation check
    sim = sim.model_copy(update={"loss_h": 0.0})
    
    stepper = TimeStepper(sim, mat)
    T_init = torch.full((1, 1, sim.Ny, sim.Nx), 300.0)
    state = SimulationState(T=T_init)
    
    monitor = EnergyMonitor(sim, mat)
    monitor.initialize(state)
    
    # Inject Heat Pulse
    # Power = 100 W spread over the center pixel
    power_W = 100.0
    # Q_flux [W/m^2]?? 
    # Stepper logic: Q_input -> Q_vol.
    # We need to construct Q such that Integrated Q_vol dV = Power.
    # Q_vol = Power / (dx dy dz) for one pixel.
    # Q_flux = Power / (dx dy) for one pixel.
    
    Q_flux = torch.zeros_like(T_init)
    xc, yc = sim.Nx // 2, sim.Ny // 2
    
    # Apply to one pixel
    pixel_area = sim.dx * sim.dy
    intensity_flux = power_W / pixel_area # [W/m^2]
    Q_flux[0, 0, yc, xc] = intensity_flux
    
    # Run 10 steps
    dt = sim.dt_base
    for _ in range(10):
        state = stepper.step_explicit_euler(state, dt, Q_ext=Q_flux)
        monitor.update(state, dt, power_in=power_W)
        
    diff = monitor.stats.error_J
    
    # Tolerances: Explicit Euler is not perfectly conservative depends on T_prev.
    # With consistent physics, error should be small.
    # The error is mostly time truncation error.
    assert abs(diff) < 1.0, f"Energy imbalance: {diff} J"

def test_gradients_linear():
    """Gradient of linear field should be constant/zero div."""
    T = torch.linspace(300, 400, 10).view(1, 1, 1, 10).float() # 1x10 grid
    # k constant
    k = torch.ones_like(T)
    
    # div_k_grad 
    div = div_k_grad(T, k, dx=1.0, dy=1.0)
    
    # Interior points should have 0 divergence.
    # Relax tolerance for float32 noise relative to values 300+.
    assert torch.allclose(div[..., 1:-1], torch.tensor(0.0), atol=1e-3)
