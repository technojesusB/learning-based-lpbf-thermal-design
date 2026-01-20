# Usage Guide

## Project Structure
- `src/neural_pbf`: Main package.
    - `core`: Core data structures (`SimulationState`, `config`).
    - `physics`: Material models (`MaterialConfig`) and operators (`ops`).
    - `scan`: Heat sources (`sources`) and scan path generation (`engine`).
    - `integrator`: Time stepping logic (`stepper`).
    - `viz`: Visualization tools.
    - `tracking`: Experiment tracking and logging.

## Running a Simulation
```python
import torch
from neural_pbf.core.config import SimulationConfig, LengthUnit
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.scan.sources import GaussianSourceConfig, GaussianBeam

# 1. Configuration
sim_cfg = SimulationConfig(
    Lx=2.0, Ly=2.0, # 2x2 mm
    Nx=100, Ny=100,
    length_unit=LengthUnit.MILLIMETERS,
    dt_base=5e-5 # 50 microseconds
)

mat_cfg = MaterialConfig(
    k_powder=0.2, k_solid=20.0, k_liquid=30.0,
    cp_base=500.0,
    rho=8000.0,
    T_solidus=1500.0, T_liquidus=1600.0,
    latent_heat_L=200000.0
)

# 2. Initialize
stepper = TimeStepper(sim_cfg, mat_cfg)
T_init = torch.full((1, 1, 100, 100), 300.0) # Ambient
state = SimulationState(T=T_init, t=0.0)

# 3. Heat Source
source_cfg = GaussianSourceConfig(power=200.0, eta=0.4, sigma=0.05e-3) # 50 micron spot
beam = GaussianBeam(source_cfg)

# 4. Loop
# Simple point dwell at center
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny),
    torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx),
    indexing='ij'
)

for step in range(100):
    # Compute source field (stationary for this example)
    Q = beam.intensity(grid_x, grid_y, None, x0=1e-3, y0=1e-3)
    
    # Adaptively step
    state = stepper.step_adaptive(state, sim_cfg.dt_base, Q_ext=Q) 

    if step % 10 == 0:
        print(f"Step {step}, Max T: {state.max_T.max():.2f} K")

# 5. Visualize
from neural_pbf.viz.static import plot_temperature_field
plot_temperature_field(state, sim_cfg)
```

## Testing & Quality Assurance

This project uses modern Python tooling for quality assurance.

### Commands

**Running Tests:**
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=neural_pbf

# Run specific test file
uv run pytest tests/lpbf/test_integrator.py
```

**Linting and Formatting:**
```bash
# Check code style and logical errors (Ruff)
uv run ruff check .

# Auto-fix fixable errors
uv run ruff check . --fix

# Type checking (Pyright)
uv run pyright
```

**Continuous Integration (CI):**
The CI pipeline automatically runs `ruff check` and `pytest` on every push. ensure these commands pass locally before committing.
