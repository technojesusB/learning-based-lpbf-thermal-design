import datetime
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.neural_pbf.core.config import LengthUnit, SimulationConfig
from src.neural_pbf.core.state import SimulationState
from src.neural_pbf.integrator.stepper import TimeStepper
from src.neural_pbf.physics.material import MaterialConfig
from src.neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from src.neural_pbf.schemas.artifacts import ArtifactConfig
from src.neural_pbf.schemas.diagnostics import DiagnosticsConfig
from src.neural_pbf.schemas.run_meta import RunMeta
from src.neural_pbf.schemas.tracking import TrackingConfig
from src.neural_pbf.tracking.run_context import RunContext
from src.neural_pbf.viz.temperature_artifacts import TemperatureArtifactBuilder


def get_trajectory(
    t: float, v: float, x_min: float, x_max: float, h: float, n_hatches: int
):
    """
    Zig-zag trajectory logic for n_hatches.
    """
    length = x_max - x_min
    time_per_hatch = length / v

    hatch_idx = int(t / time_per_hatch)
    if hatch_idx >= n_hatches:
        # Stay at end of last hatch
        hatch_idx = n_hatches - 1
        t = time_per_hatch * n_hatches

    t_in_hatch = t % time_per_hatch

    y = 0.1e-3 + hatch_idx * h

    x = x_min + v * t_in_hatch if hatch_idx % 2 == 0 else x_max - v * t_in_hatch

    return x, y


def run_ss316l_hi_fid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. SS316L Calibrated Parameters
    mat_cfg = MaterialConfig.ss316l_preset()

    # Laser: 200W, 1m/s, 40um radius
    v_scan = 1.0  # m/s
    h_mm = 0.08  # 80um spacing for 33% overlap
    source_cfg = GaussianSourceConfig(
        power=200.0,
        eta=0.35,
        sigma=40e-6,
        depth=30e-6,
    )
    source = GaussianBeam(source_cfg)

    # 2. Simulation Domain
    # Lx=1.0, Ly=0.5, Lz=0.2 (mm)
    sim_cfg = SimulationConfig(
        Lx=1.0,
        Ly=0.5,
        Lz=0.125,
        Nx=1024,
        Ny=512,
        Nz=128,
        dt_base=1e-6,
        length_unit=LengthUnit.MILLIMETERS,
    )

    run_name = "SS316L_HiFid_MultiHatch"
    base_dir = Path("artifacts")
    out_dir = base_dir / run_name

    track_cfg = TrackingConfig(run_name=run_name, base_dir=base_dir)
    art_cfg = ArtifactConfig(
        enabled=True,
        png_every_n_steps=20,
        html_every_n_steps=100,
        make_report=True,
        save_raw=True,  # User requested raw persistence
        buffer_steps=False,  # Disable RAM buffering to avoid OOM
    )
    diag_cfg = DiagnosticsConfig(log_every_n_steps=50)

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

    # Pre-compute grid
    x_l = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device)
    y_l = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device)
    z_l = torch.linspace(0, sim_cfg.Lz_m, sim_cfg.Nz, device=device)
    X, Y, Z = torch.meshgrid(x_l, y_l, z_l, indexing="ij")

    # 4. Simulation Loop
    n_hatches = 4
    length_m = 0.6e-3  # 0.2 to 0.8 mm
    total_time = n_hatches * (length_m / v_scan)
    steps = int(total_time / sim_cfg.dt_base)

    # Resume Logic
    start_step = 0
    states_dir = out_dir / "states"
    if states_dir.exists():
        checkpoints = sorted(list(states_dir.glob("step_*.npy")))
        if checkpoints:
            last_ckpt = checkpoints[-1]
            try:
                # Extract step index from filename step_XXXXXX.npy
                resume_step = int(last_ckpt.stem.split("_")[1])
                print(
                    f"Found checkpoint: {last_ckpt}. Resuming from step {resume_step}..."
                )

                # Load T
                T_np = np.load(last_ckpt)
                T = torch.from_numpy(T_np).to(device)

                # Ensure dimensions match (broadcasting fix if saved reduced)
                if T.ndim == 3:
                    # If saved as [Nx, Ny, Nz] but we need [1, 1, Nx, Ny, Nz]
                    T = T.unsqueeze(0).unsqueeze(0)

                # Restore state vars
                state.T = T
                # Restore Mask (approximate from T)
                state.material_mask = (T > mat_cfg.T_solidus).int()

                start_step = resume_step + 1
                state.t = start_step * sim_cfg.dt_base
                state.step = start_step

            except Exception as e:
                print(f"Failed to resume from checkpoint {last_ckpt}: {e}")
                print("Starting from scratch.")

    print(
        f"Simulating SS316L Hi-Fid: Steps {start_step} to {steps} ({steps - start_step} remaining)..."
    )

    pbar = tqdm(range(start_step, steps), file=sys.stdout, mininterval=2.0)
    try:
        for i in pbar:
            t = i * sim_cfg.dt_base
            x_m, y_m = get_trajectory(
                t,
                v_scan,
                x_min=0.2e-3,
                x_max=0.8e-3,
                h=h_mm * 1e-3,
                n_hatches=n_hatches,
            )

            Q_vol = source.intensity(X, Y, Z, x0=x_m, y0=y_m, z0=sim_cfg.Lz_m)

            state = stepper.step_adaptive(
                state,
                dt_target=sim_cfg.dt_base,
                Q_ext=Q_vol.unsqueeze(0).unsqueeze(0),
                use_triton=True,
            )
            state.t = t

            ctx.maybe_snapshot(i, state, meta={"t": t, "laser_x": x_m, "laser_y": y_m})

            if i % 10 == 0:
                T_max = state.T.max().item()
                pbar.set_postfix(
                    {
                        "T_max": f"{T_max:.1f}K",
                        "pos": f"({x_m * 1e3:.2f},{y_m * 1e3:.2f})",
                    }
                )

    except KeyboardInterrupt:
        print("Interrupted.")

    ctx.end(state)
    print(f"Simulation Complete. Results in {out_dir}")


if __name__ == "__main__":
    run_ss316l_hi_fid()
