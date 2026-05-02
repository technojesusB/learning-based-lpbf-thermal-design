# tests/lpbf/test_integration.py
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from neural_pbf.utils.units import LengthUnit


def test_simulation_pipeline_smoke():
    """
    Run a minimal 10-step simulation using the core components
    to ensure no runtime errors in the main loop.
    """
    # 1. Config
    sim = SimulationConfig(
        Lx=0.5,
        Ly=0.5,
        Nx=32,
        Ny=32,
        length_unit=LengthUnit.MILLIMETERS,
        dt_base=1e-5,
        default_dz=0.05,
    )
    mat = MaterialConfig(
        k_powder=10.0,
        k_solid=10.0,
        k_liquid=10.0,
        cp_base=500.0,
        rho=2000.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=200000.0,
    )

    # 2. Components
    stepper = TimeStepper(sim, mat)

    # 3. State
    T = torch.full((1, 1, sim.Ny, sim.Nx), 300.0)
    state = SimulationState(T=T)

    # 4. Source
    source = GaussianBeam(GaussianSourceConfig(power=100.0, sigma=0.05))

    # 5. Grid
    # Simple grid generation
    x = torch.linspace(0, sim.Lx_m, sim.Nx)
    y = torch.linspace(0, sim.Ly_m, sim.Ny)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    X = X.unsqueeze(0).unsqueeze(0)
    Y = Y.unsqueeze(0).unsqueeze(0)

    # 6. Loop
    for _ in range(10):
        intensity = source.intensity(X, Y, None, x0=sim.Lx_m / 2, y0=sim.Ly_m / 2)
        state = stepper.step_explicit_euler(state, sim.dt_base, Q_ext=intensity)

    # 7. Check output
    assert state.step == 10
    assert state.T.max() > 300.0
