import datetime
import shutil
import sys
from pathlib import Path

import torch

from src.neural_pbf.core.config import LengthUnit, SimulationConfig  # noqa: E402
from src.neural_pbf.core.state import SimulationState  # noqa: E402
from src.neural_pbf.integrator.stepper import TimeStepper  # noqa: E402
from src.neural_pbf.physics.material import MaterialConfig  # noqa: E402
from src.neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig  # noqa: E402
from src.neural_pbf.schemas.artifacts import ArtifactConfig  # noqa: E402
from src.neural_pbf.schemas.diagnostics import DiagnosticsConfig  # noqa: E402
from src.neural_pbf.schemas.run_meta import RunMeta  # noqa: E402
from src.neural_pbf.schemas.tracking import TrackingConfig  # noqa: E402
from src.neural_pbf.tracking.run_context import RunContext  # noqa: E402
from src.neural_pbf.viz.temperature_artifacts import (
    TemperatureArtifactBuilder,
)

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def run_3d_experiment(
    mat_cfg: MaterialConfig | None = None,
    source_cfg: GaussianSourceConfig | None = None,
    sim_cfg: SimulationConfig | None = None,
    art_cfg: ArtifactConfig | None = None,
    track_cfg: TrackingConfig | None = None,
    diag_cfg: DiagnosticsConfig | None = None,
    run_name: str | None = None,
    scan_speed: float = 0.8,
    total_time: float = 0.6e-3,
):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Physics Configuration (SI Units)
    if mat_cfg is None:
        mat_cfg = MaterialConfig(
            k_powder=0.2,
            k_solid=15.0,
            k_liquid=30.0,
            cp_base=500.0,
            rho=7900.0,
            T_solidus=1650.0,
            T_liquidus=1700.0,
            latent_heat_L=2.7e5,
            transition_sharpness=5.0,
        )

    if source_cfg is None:
        source_cfg = GaussianSourceConfig(
            power=150.0,
            eta=0.35,
            sigma=40e-6,
            depth=50e-6,  # [m] Volumetric absorption depth
        )
    source = GaussianBeam(source_cfg)

    # 2. Simulation Config
    if sim_cfg is None:
        sim_cfg = SimulationConfig(
            Lx=0.8,
            Ly=0.4,
            Lz=0.2,
            Nx=80,
            Ny=40,
            Nz=20,
            dt_base=2e-6,
            length_unit=LengthUnit.MILLIMETERS,
        )

    # Artifacts
    if run_name is None:
        run_name = f"run_3d_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_dir = Path("artifacts")
    out_dir = base_dir / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    if track_cfg is None:
        track_cfg = TrackingConfig(run_name=run_name, base_dir=base_dir)

    if art_cfg is None:
        art_cfg = ArtifactConfig(
            enabled=True,
            png_every_n_steps=50,
            html_every_n_steps=100,  # Interactive volume rendering
            make_report=True,
        )

    if diag_cfg is None:
        diag_cfg = DiagnosticsConfig(log_every_n_steps=50)

    # 3. Context & State
    viz = TemperatureArtifactBuilder(art_cfg)
    run_meta = RunMeta(
        seed=42,
        device=device,
        dtype=str(torch.float32),
        started_at=datetime.datetime.now().isoformat(),
        grid_shape=[sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz],
        dx=sim_cfg.dx,
        dy=sim_cfg.dy,
        dz=sim_cfg.dz,
        dt=sim_cfg.dt_base,
        material_summary={
            k: v for k, v in mat_cfg.model_dump().items() if v is not None
        },
        scan_summary={
            k: v for k, v in source_cfg.model_dump().items() if v is not None
        },
    )

    ctx = RunContext(
        tracking_cfg=track_cfg,
        artifact_cfg=art_cfg,
        diagnostics_cfg=diag_cfg,
        run_meta=run_meta,
        out_dir=out_dir,
        artifact_builder=viz,
    )
    ctx.start()

    # Initial State
    # Shape must be (B, C, Nx, Ny, Nz)
    T = torch.full(
        (1, 1, sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz),
        300.0,
        device=device,
        dtype=torch.float32,
    )
    mask = torch.zeros_like(T, dtype=torch.int32)

    # Stepper & Grid
    stepper = TimeStepper(sim_cfg, mat_cfg)

    # Pre-compute grid for source evaluation
    x_lin = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device)
    y_lin = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device)
    z_lin = torch.linspace(0, sim_cfg.Lz_m, sim_cfg.Nz, device=device)
    X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")

    # 4. Loop
    # Use arguments
    steps = int(total_time / sim_cfg.dt_base)

    print(f"Running 3D Sim: {steps} steps, dt={sim_cfg.dt_base:.2e}s")

    state = SimulationState(T=T, t=0.0, material_mask=mask)

    start_pos = torch.tensor([0.2e-3, 0.2e-3], device=device)
    velocity = torch.tensor([scan_speed, 0.0], device=device)
    t_tensor = torch.tensor(0.0, device=device)

    try:
        for i in range(steps):
            current_pos = start_pos + velocity * t_tensor

            # Calculate Source Term [W/m^3]
            # z0 should be top surface: Lz
            Q_vol = source.intensity(
                X,
                Y,
                Z,
                x0=float(current_pos[0]),
                y0=float(current_pos[1]),
                z0=float(sim_cfg.Lz_m),
            )

            # Use adaptive stepping for stability
            state = stepper.step_adaptive(
                state, dt_target=sim_cfg.dt_base, Q_ext=Q_vol.unsqueeze(0).unsqueeze(0)
            )

            t_tensor += sim_cfg.dt_base
            state.t = float(t_tensor)

            ctx.maybe_snapshot(i, state, meta={"t": state.t})

            if i % 50 == 0:
                T_max = state.T.max().item()
                print(f"Step {i}/{steps} | T_max={T_max:.1f} K")

    except KeyboardInterrupt:
        print("Interrupted.")

    ctx.end(state)
    print(f"3D Experiment Complete. Results in {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_3d_experiment()
