
import torch
import numpy as np
from pathlib import Path
import shutil
import datetime
import sys

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.neural_pbf.core.config import SimulationConfig, LengthUnit
from src.neural_pbf.core.state import SimulationState
from src.neural_pbf.physics.material import MaterialConfig
from src.neural_pbf.integrator.stepper import TimeStepper
from src.neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from src.neural_pbf.viz.temperature_artifacts import TemperatureArtifactBuilder
from src.neural_pbf.schemas.artifacts import ArtifactConfig
from src.neural_pbf.schemas.tracking import TrackingConfig
from src.neural_pbf.schemas.diagnostics import DiagnosticsConfig
from src.neural_pbf.schemas.run_meta import RunMeta
from src.neural_pbf.tracking.run_context import RunContext

from typing import Optional

def run_2d_experiment(
    mat_cfg: Optional[MaterialConfig] = None,
    source_cfg: Optional[GaussianSourceConfig] = None,
    sim_cfg: Optional[SimulationConfig] = None,
    art_cfg: Optional[ArtifactConfig] = None,
    track_cfg: Optional[TrackingConfig] = None,
    diag_cfg: Optional[DiagnosticsConfig] = None,
    run_name: Optional[str] = None,
    scan_speed: float = 0.8,
    total_time: float = 0.8e-3
):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Physics Configuration (SI Units: m, W, J, kg, K)
    # SS316L approximation
    if mat_cfg is None:
        mat_cfg = MaterialConfig(
            k_powder=0.2,      # [W/(m K)]
            k_solid=15.0,      # [W/(m K)]
            k_liquid=30.0,     # [W/(m K)]
            cp_base=500.0,     # [J/(kg K)]
            rho=7900.0,        # [kg/m^3]
            T_solidus=1650.0,  # [K]
            T_liquidus=1700.0, # [K]
            latent_heat_L=2.7e5, # [J/kg]
            transition_sharpness=5.0
        )

    # Laser 
    if source_cfg is None:
        source_cfg = GaussianSourceConfig(
            power=150.0,       # [W] Lower power for 2D stability check
            eta=0.35,          # Absorption
            sigma=40e-6        # [m] 40um radius
        )
    source = GaussianBeam(source_cfg)

    # 2. Simulation Config
    # Lx=1.0mm, Nx=100 -> dx = 1e-5 m (10um)
    if sim_cfg is None:
        sim_cfg = SimulationConfig(
            Lx=1.0, Ly=0.5, Lz=None, # 2D
            Nx=100, Ny=50, Nz=1,
            dt_base=2e-6,           # [s] Conservative dt (2 microseconds)
            length_unit=LengthUnit.MILLIMETERS
        )
    
    # Artifacts
    if run_name is None:
        run_name = f"run_2d_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_dir = Path("artifacts")
    out_dir = base_dir / run_name
    
    # Only clear if it's a fresh run we generated the name for, 
    # OR if the user deliberately re-uses a name, we might want to warn or just overwrite.
    # For safety, let's only wipe if it exists.
    if out_dir.exists(): shutil.rmtree(out_dir)

    if track_cfg is None:
        track_cfg = TrackingConfig(run_name=run_name, base_dir=base_dir)
        
    if art_cfg is None:
        art_cfg = ArtifactConfig(
            enabled=True,
            png_every_n_steps=50,
            html_every_n_steps=100,
            make_report=True
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
        dx=sim_cfg.dx, dy=sim_cfg.dy, dz=sim_cfg.dz, dt=sim_cfg.dt_base,
        material_summary={k: v for k, v in mat_cfg.model_dump().items() if v is not None},
        scan_summary={k: v for k, v in source_cfg.model_dump().items() if v is not None}
    )
    

    # The error was missing out_dir. RunContext(..., run_meta, out_dir, ...)
    # Wait, looking at signature:
    # __init__(self, tracking_cfg, artifact_cfg, diagnostics_cfg, run_meta, out_dir, artifact_builder=None)
    # My previous code was: ctx = RunContext(track_cfg, diag_cfg, art_cfg, viz) 
    # This is completely wrong order and missing args.
    
    ctx = RunContext(
        tracking_cfg=track_cfg,
        artifact_cfg=art_cfg,
        diagnostics_cfg=diag_cfg,
        run_meta=run_meta,
        out_dir=out_dir,
        artifact_builder=viz
    )
    ctx.start()

    # Initial State
    # Shape must be (B, C, Nx, Ny) for ops.py padding
    T = torch.full(
        (1, 1, sim_cfg.Nx, sim_cfg.Ny), 
        300.0, 
        device=device, 
        dtype=torch.float32
    )
    # Mask: 0=Powder
    mask = torch.zeros_like(T, dtype=torch.int32)
    
    # Stepper & Grid
    stepper = TimeStepper(sim_cfg, mat_cfg)
    
    # Pre-compute grid for source evaluation
    x_lin = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device)
    y_lin = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    
    # Grid needs to broadcast with (B, C, ...) if used with state directly, 
    # but source.intensity expects spatial grids. 
    # The output Q will be (Nx, Ny), we need to unsqueeze it.

    # 4. Loop
    # Use arguments for speed and time
    steps = int(total_time / sim_cfg.dt_base)
    
    print(f"Running 2D Sim: {steps} steps, dt={sim_cfg.dt_base:.2e}s")
    
    state = SimulationState(T=T, t=0.0, material_mask=mask)
    
    # Simple Path
    start_pos = torch.tensor([0.2e-3, 0.25e-3], device=device)
    velocity = torch.tensor([scan_speed, 0.0], device=device)
    t_tensor = torch.tensor(0.0, device=device)

    try:
        for i in range(steps):
            current_pos = start_pos + velocity * t_tensor
            
            # Calculate Source Flux [W/m^2]
            Q_flux = source.intensity(
                X, Y, None, 
                x0=float(current_pos[0]), 
                y0=float(current_pos[1])
            )
            
            # Step (Adaptive)
            # Q_flux is (Nx, Ny), need (1, 1, Nx, Ny)
            state = stepper.step_adaptive(
                state, 
                dt_target=sim_cfg.dt_base,
                Q_ext=Q_flux.unsqueeze(0).unsqueeze(0)
            )
            
            t_tensor += sim_cfg.dt_base
            state.t = float(t_tensor)
            
            # Snapshot
            ctx.maybe_snapshot(i, state, meta={"t": state.t})
            
            if i % 100 == 0:
                T_max = state.T.max().item()
                print(f"Step {i}/{steps} | T_max={T_max:.1f} K")
                if T_max > 5000.0:
                    print("WARN: T > 5000K, likely unstable.")
            
    except KeyboardInterrupt:
        print("Interrupted.")
    
    ctx.end(state)
    print(f"2D Experiment Complete. Results in {out_dir}")
    return out_dir

if __name__ == "__main__":
    run_2d_experiment()
