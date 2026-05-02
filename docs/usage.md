# Usage Guide

## Project Structure

<!-- AUTO-GENERATED from src/neural_pbf/ layout — do not edit this section manually -->

- `src/neural_pbf`: Main package.
    - `core`: Core data structures (`SimulationState`, `SimulationConfig`).
    - `physics`: Material models (`MaterialConfig`) and FD operators (`ops`, `triton_ops`).
    - `scan`: Heat sources (`sources`), zig-zag/raster/island path generation (`path_generator`), and scan engine (`engine`).
    - `integrator`: Explicit Euler time stepping with CFL sub-stepping (`stepper`).
    - `data`: Lazy-loading HDF5 dataset for offline surrogate training (`hdf5_dataset`).
    - `pipelines`: Coordinate grid construction (`grids`) and end-to-end surrogate training pipeline (`training_pipeline`).
    - `models`: Neural surrogate architecture (`surrogate`), physics-informed loss (`loss`), replay buffer (`replay_buffer`), and hyperparameter config (`config`).
    - `viz`: Visualization tools (static plots, interactive Plotly/Dash, animations).
    - `tracking`: Experiment tracking, MLflow backend, diagnostics, and artifact generation.

<!-- END AUTO-GENERATED -->

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

# 5. High-Fidelity Experiments
The repository includes several high-fidelity scripts that utilize optimized Triton kernels and advanced visualization:
- `experiments/ss316l_multi_hatch.py`: Simulates 4-hatch zig-zag pattern with calibrated SS316L parameters.
- `experiments/run_3d.py`: Standard 3D track simulation.

# 6. Visualize
from neural_pbf.viz.static import plot_temperature_field
plot_temperature_field(state, sim_cfg)
```

## Offline Dataset & Surrogate Training

### Generate an Offline HDF5 Dataset
```bash
# Default: 10 runs × 15 snapshots on CUDA, output to data/offline_dataset.h5
uv run python scripts/generate_offline_dataset.py

# Custom: 50 runs, 128³ grid, CPU fallback
uv run python scripts/generate_offline_dataset.py \
    --runs 50 --nx 128 --ny 64 --nz 16 \
    --out data/my_dataset.h5 --device cpu
```

Each HDF5 sample group contains `T_in`, `Q`, `T_target`, `T_lf` (fp16), `mask` (uint8), and `scalars` `[t, laser_x, laser_y]` (fp32). See [Offline Dataset Guide](offline_dataset_generation.md) for the full schema.

### Train the Surrogate
```bash
uv run python scripts/train_surrogate.py
```

### Visualise a Trained Surrogate
```bash
uv run python scripts/viz_surrogate.py
```

---

## Testing & Quality Assurance

<!-- AUTO-GENERATED from pyproject.toml [project.scripts] — do not edit this section manually -->

### Command Reference

| Command | Description |
|---------|-------------|
| `uv run lint` | `ruff check .` — lint `src/` |
| `uv run format` | `ruff format --check .` — check formatting |
| `uv run typecheck` | `pyright src/` — static type checking |
| `uv run test` | `pytest tests/` — full test suite |

<!-- END AUTO-GENERATED -->

### Running Tests
```bash
# Run all tests
uv run test

# Run tests with coverage
uv run pytest tests/ --cov=src/neural_pbf --cov-report=term-missing

# Run a specific file
uv run pytest tests/lpbf/test_integrator.py -v

# Run a specific test by name
uv run pytest tests/ -k "test_zigzag_alternates" -v
```

**Linting and Formatting:**
```bash
# Check code style and logical errors (Ruff)
uv run lint

# Auto-fix fixable errors
uv run ruff check . --fix

# Type checking (Pyright, src/ only)
uv run typecheck
```

> [!NOTE]
> You may notice `# type: ignore` comments in high-performance GPU kernels (e.g., `triton_ops.py`). These are necessary because static type checkers like Pyright do not yet fully support Triton's Domain-Specific Language (DSL), such as `tl.constexpr` and dynamic kernel launchers. These ignores are intentional to maintain both high-performance and strict compile-time optimization.

**Continuous Integration (CI):**
The CI pipeline automatically runs `ruff check` and `pytest` on every push. ensure these commands pass locally before committing.
