
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

def run_hatch_experiment():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Physics Configuration (High Physicality)
    mat_cfg = MaterialConfig(
        k_powder=0.2,      
        k_solid=25.0,      # High conductivity
        k_liquid=45.0,     
        cp_base=500.0,     
        rho=7900.0,        
        T_solidus=1650.0,  
        T_liquidus=1700.0, 
        latent_heat_L=2.7e5, 
        transition_sharpness=5.0
    )

    source_cfg = GaussianSourceConfig(
        power=180.0,       # Slightly higher power for 3D
        eta=0.35,          
        sigma=40e-6,
        depth=50e-6
    )
    source = GaussianBeam(source_cfg)

    # 2. Simulation Config (Extreme High-Fidelity)
    sim_cfg = SimulationConfig(
        Lx=1.0, Ly=0.5, Lz=0.25, 
        Nx=512, Ny=256, Nz=128,
        dt_base=2e-6,           
        length_unit=LengthUnit.MILLIMETERS
    )
    
    run_name = f"hatch_pattern_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path("artifacts")
    out_dir = base_dir / run_name
    if out_dir.exists(): shutil.rmtree(out_dir)

    track_cfg = TrackingConfig(run_name=run_name, base_dir=base_dir)
    art_cfg = ArtifactConfig(
        enabled=True,
        png_every_n_steps=10,
        html_every_n_steps=200,
        make_report=True
    )
    diag_cfg = DiagnosticsConfig(log_every_n_steps=10)

    # 3. Context & State
    viz = TemperatureArtifactBuilder(art_cfg)
    run_meta = RunMeta(
        seed=42,
        device=device,
        dtype=str(torch.float32),
        started_at=datetime.datetime.now().isoformat(),
        grid_shape=[sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz],
        dx=sim_cfg.dx, dy=sim_cfg.dy, dz=sim_cfg.dz, dt=sim_cfg.dt_base,
        material_summary=mat_cfg.model_dump(),
        scan_summary=source_cfg.model_dump()
    )
    
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
    T = torch.full((1, 1, sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz), 300.0, device=device, dtype=torch.float32)
    mask = torch.zeros_like(T, dtype=torch.int32)
    state = SimulationState(T=T, t=0.0, material_mask=mask)
    stepper = TimeStepper(sim_cfg, mat_cfg)
    
    # Pre-compute grid for source evaluation
    x_lin = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device)
    y_lin = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device)
    z_lin = torch.linspace(0, sim_cfg.Lz_m, sim_cfg.Nz, device=device)
    X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")

    # 4. Scan Pattern Logic
    scan_speed = 0.8  # m/s
    
    # Hatch 1: 0.2 to 0.8 mm in X
    x_start, x_end = 0.2e-3, 0.8e-3
    y0 = 0.2e-3
    
    # Hatch spacing for 1/3 overlap
    # Effective diameter ~ 4 * sigma
    D_eff = 4 * 40e-6 # 160 um
    hatch_spacing = (2/3) * D_eff # ~106.7 um
    y1 = y0 + hatch_spacing

    dist = x_end - x_start
    t_hatch = dist / scan_speed
    steps_hatch = int(t_hatch / sim_cfg.dt_base)
    
    print(f"Running Bidirectional Hatch: {2*steps_hatch} total steps")
    print(f"Hatch Spacing: {hatch_spacing*1e6:.1f} um (1/3 overlap)")

    try:
        # HATCH 1 (Forward)
        for i in range(steps_hatch):
            t_curr = i * sim_cfg.dt_base
            curr_x = x_start + scan_speed * t_curr
            curr_y = y0
            
            Q_vol = source.intensity(X, Y, Z, x0=float(curr_x), y0=float(curr_y), z0=float(sim_cfg.Lz_m))
            state = stepper.step_adaptive(state, dt_target=sim_cfg.dt_base, Q_ext=Q_vol.unsqueeze(0).unsqueeze(0))
            
            state.t = t_curr
            ctx.maybe_snapshot(i, state, meta={"t": state.t, "hatch": 1})
            if i % 50 == 0:
                print(f"Hatch 1 | Step {i}/{steps_hatch} | T_max={state.T.max().item():.1f} K")

        # HATCH 2 (Backward)
        offset = steps_hatch
        for i in range(steps_hatch):
            t_curr = i * sim_cfg.dt_base
            curr_x = x_end - scan_speed * t_curr # BACKWARD
            curr_y = y1
            
            Q_vol = source.intensity(X, Y, Z, x0=float(curr_x), y0=float(curr_y), z0=float(sim_cfg.Lz_m))
            state = stepper.step_adaptive(state, dt_target=sim_cfg.dt_base, Q_ext=Q_vol.unsqueeze(0).unsqueeze(0))
            
            state.t = t_hatch + t_curr
            ctx.maybe_snapshot(offset + i, state, meta={"t": state.t, "hatch": 2})
            if i % 50 == 0:
                print(f"Hatch 2 | Step {i}/{steps_hatch} | T_max={state.T.max().item():.1f} K")
            
    except KeyboardInterrupt:
        print("Interrupted.")
    
    ctx.end(state)
    print(f"Experiment Complete. Results in {out_dir}")

if __name__ == "__main__":
    run_hatch_experiment()
