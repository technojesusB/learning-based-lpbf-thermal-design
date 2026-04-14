import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# Ensure we can import from src
sys.path.append(str(Path.cwd() / "src"))

from neural_pbf.core.config import LengthUnit, SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from neural_pbf.schemas.artifacts import ArtifactConfig
from neural_pbf.schemas.diagnostics import DiagnosticsConfig
from neural_pbf.schemas.run_meta import RunMeta
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.run_context import RunContext
from neural_pbf.viz.temperature_artifacts import TemperatureArtifactBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_trajectory(
    t: float, v: float, x_min: float, x_max: float, h: float, n_hatches: int
):
    """S-shaped hach trajectory with scan_time check."""
    length = x_max - x_min
    time_per_hatch = length / v
    hatch_idx = int(t / time_per_hatch)
    
    # If we are past the hatches, laser is logically finished
    if hatch_idx >= n_hatches:
        return x_max, 0.0 # doesn't matter, laser should be off
        
    t_in_hatch = t % time_per_hatch
    y = 0.1e-3 + hatch_idx * h
    if hatch_idx % 2 == 0:
        x = x_min + v * t_in_hatch
    else:
        x = x_max - v * t_in_hatch
    return x, y


def run_ss316l_cooling_hatch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}")

    # SS316L Material
    mat_cfg = MaterialConfig.ss316l_preset()

    # Laser Source (Baseline 41f3ae4)
    v_scan = 1.0
    h_mm = 0.08
    source_cfg = GaussianSourceConfig(power=200.0, eta=0.35, sigma=40e-6, depth=30e-6)
    source = GaussianBeam(source_cfg)

    # Hi-Fid Grid (Baseline 41f3ae4)
    sim_cfg = SimulationConfig(
        Lx=1.0,  # 1mm
        Ly=0.5,  # 0.5mm
        Lz=0.125,  # 125um
        Nx=1024,
        Ny=512,
        Nz=128,
        dt_base=1e-6,
        length_unit=LengthUnit.MILLIMETERS,
    )

    run_name = "SS316L_Cooling_4Hatch"
    base_dir = Path("artifacts")
    out_dir = base_dir / run_name

    track_cfg = TrackingConfig(run_name=run_name, base_dir=base_dir)
    art_cfg = ArtifactConfig(
        enabled=True,
        png_every_n_steps=20,
        html_every_n_steps=200,
        make_report=True,
        save_raw=True,
        buffer_steps=False,  # RAM optimization: Avoid keeping full 67M tensors in RAM
        downsample=2,        # Perf optimization: Reduce rendering overhead for massive grids
        show_phase_map=True,
        T_solidus=mat_cfg.T_solidus,
        T_liquidus=mat_cfg.T_liquidus,
    )
    diag_cfg = DiagnosticsConfig(log_every_n_steps=100)

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
        material_summary=mat_cfg.model_dump(),
        scan_summary=source_cfg.model_dump(),
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
    T = torch.full((1, 1, sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz), 300.0, device=device)
    mask = torch.zeros_like(T, dtype=torch.int32)
    state = SimulationState(T=T, t=0.0, material_mask=mask)
    stepper = TimeStepper(sim_cfg, mat_cfg)

    # Grid
    x_l = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device)
    y_l = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device)
    z_l = torch.linspace(0, sim_cfg.Lz_m, sim_cfg.Nz, device=device)
    X, Y, Z = torch.meshgrid(x_l, y_l, z_l, indexing="ij")

    # Timing
    n_hatches = 4
    length_m = 0.6e-3 # scan length per hatch
    scan_time = n_hatches * (length_m / v_scan)
    cooling_time = 0.4e-3 # 0.4ms cooling
    total_time = scan_time + cooling_time
    steps = int(total_time / sim_cfg.dt_base)

    # Resume Logic
    start_step = 0
    states_dir = out_dir / "states"
    if states_dir.exists():
        checkpoints = sorted(list(states_dir.glob("step_*_T.npy")))
        if checkpoints:
            last_ckpt = checkpoints[-1]
            try:
                # step_000320_T.npy -> 320
                resume_step = int(last_ckpt.stem.split("_")[1])
                print(f"Resuming from step {resume_step} using {last_ckpt}")
                T_np = np.load(last_ckpt)
                T = torch.from_numpy(T_np).to(device).unsqueeze(0).unsqueeze(0)
                state.T = T
                state.t = resume_step * sim_cfg.dt_base
                # approx mask from T for now
                state.material_mask = (T > mat_cfg.T_solidus).int()
                start_step = resume_step + 1
            except Exception as e:
                print(f"Failed resume: {e}")

    print(f"Simulating SS316L Cooling: {n_hatches} hatches + {cooling_time*1e3:.1f}ms cooling.")
    print(f"Grid: {sim_cfg.Nx}x{sim_cfg.Ny}x{sim_cfg.Nz} ({sim_cfg.Nx*sim_cfg.Ny*sim_cfg.Nz/1e6:.1f}M cells)")
    print(f"Resolution: {sim_cfg.dx*1e6:.2f}um x {sim_cfg.dy*1e6:.2f}um x {sim_cfg.dz*1e6:.2f}um")
    print(f"Time Step: {sim_cfg.dt_base*1e6:.1f}us")
    print(f"Total steps: {steps} (Scan time: {scan_time*1e3:.2f}ms)")
    
    # Test stability to see n_sub
    dt_crit = stepper.estimate_stability_dt(state)
    n_sub = int(np.ceil(sim_cfg.dt_base / dt_crit))
    print(f"Estimated stability dt: {dt_crit*1e9:.2f}ns -> Sub-steps per iter: {n_sub}")

    pbar = tqdm(range(start_step, steps), file=sys.stdout, mininterval=2.0)
    try:
        for i in pbar:
            t = i * sim_cfg.dt_base
            
            if t < scan_time:
                # Laser is ON
                x_m, y_m = get_trajectory(
                    t,
                    v_scan,
                    x_min=0.2e-3,
                    x_max=0.8e-3,
                    h=h_mm * 1e-3,
                    n_hatches=n_hatches,
                )
                Q_vol = source.intensity(X, Y, Z, x0=x_m, y0=y_m, z0=sim_cfg.Lz_m)
            else:
                # Laser is OFF
                x_m, y_m = 0.0, 0.0 # not used
                Q_vol = torch.zeros_like(X)

            state = stepper.step_adaptive(
                state,
                dt_target=sim_cfg.dt_base,
                Q_ext=Q_vol.unsqueeze(0).unsqueeze(0),
                use_triton=True,
            )
            state.t = t

            ctx.maybe_snapshot(i, state, meta={"t": t, "laser_x": x_m, "laser_y": y_m})

            if i % 20 == 0:
                T_max = state.T.max().item()
                mode = "COOL" if t >= scan_time else "SCAN"
                pbar.set_postfix(
                    {
                        "T_max": f"{T_max:.1f}K",
                        "mode": mode,
                    }
                )

    except KeyboardInterrupt:
        print("Interrupted.")

    ctx.end(state)
    print(f"Simulation Complete. Results in {out_dir}")


if __name__ == "__main__":
    run_ss316l_cooling_hatch()
