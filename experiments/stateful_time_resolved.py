# experiments/stateful_time_resolved.py
from __future__ import annotations

import datetime
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from neural_pbf.schemas.artifacts import ArtifactConfig
from neural_pbf.schemas.diagnostics import DiagnosticsConfig
from neural_pbf.schemas.run_meta import RunMeta
from neural_pbf.schemas.state import FinalState, StateMeta, ThermalStates

# Tracking & Diagnostics
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.run_context import RunContext
from neural_pbf.utils.grid import make_xy_grid
from neural_pbf.utils.history import make_smooth_preheat_field
from neural_pbf.utils.io import save_state
from neural_pbf.utils.state_recorder import StateRecorder
from neural_pbf.viz.temperature_artifacts import TemperatureArtifactBuilder

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# We keep DotEvent locally as a simple experiment parameter implementation
@dataclass
class DotEvent:
    x: float
    y: float
    power: float
    dwell: float  # laser-on duration
    travel: float  # time until next dot (laser off)
    sigma: float
    eta: float = 1.0


def update_history_maps(
    E_acc: torch.Tensor,
    t_since: torch.Tensor,
    spot: torch.Tensor,
    dt_event: float,
    energy: float,
    reset_threshold: float = 0.2,
):
    """
    Update auxiliary history maps (E_acc, t_since).
    Logic adapted from legacy datagenerator.state.
    """
    # 1. Update t_since everywhere for the duration of the event
    t_since = t_since + dt_event

    # 2. Identify where energy was deposited (active zone)
    # We use the spot intensity field normalized or thresholded
    hit_mask = spot > (spot.max() * reset_threshold)

    # 3. Accumulate energy
    E_acc = torch.where(hit_mask, E_acc + energy, E_acc)

    # 4. Reset t_since where hit
    t_since = torch.where(hit_mask, torch.zeros_like(t_since), t_since)

    return E_acc, t_since


def main() -> None:
    # ----------------------------
    # 0. Configuration & Tracking Setup
    # ----------------------------
    # Parse from env or defaults
    tracking_cfg = TrackingConfig(
        enabled=True,
        backend=os.environ.get("TRACKING_BACKEND", "none"),  # type: ignore
        experiment_name="lpbf-thermal",
        run_name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    )

    artifact_cfg = ArtifactConfig(
        enabled=True,
        png_every_n_steps=50,
        html_every_n_steps=250,  # Set to 0 if plotly slow/missing
        make_report=True,
    )

    diagnostics_cfg = DiagnosticsConfig(
        enabled=True,
        log_every_n_steps=10,  # frequent logging for metrics
        check_nan_inf=True,
        strict=False,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.float32

    # Run Metadata
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------
    # Domain & Config
    # ----------------------------
    H = W = 128
    Lx = 1.0
    Ly = 1.0

    sim_config = SimulationConfig(
        Lx=Lx,
        Ly=Ly,
        Nx=W,
        Ny=H,
        length_unit="m",
        dt_base=6.0e-4,
        T_ambient=0.05,
        loss_h=0.2,
    )
    dt = 6.0e-4
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    # ----------------------------
    # Material
    # ----------------------------
    mat = MaterialConfig(
        k_powder=0.0012,
        k_solid=0.018,
        k_liquid=0.012,
        cp_base=1.0,
        T_solidus=0.60,
        T_liquidus=0.70,
        latent_heat_L=0.35,
        rho=1.0,
    )

    # ----------------------------
    # Initialize RunContext
    # ----------------------------
    run_meta = RunMeta(
        seed=seed,
        device=device_str,
        dtype=str(dtype),
        started_at=datetime.datetime.now().isoformat(),
        dx=dx,
        dy=dy,
        dz=0.0,
        dt=dt,
        grid_shape=[H, W],
        material_summary=mat.__dict__,  # assuming dataclass can simple dict
        notes="Integration test run",
    )

    out_dir = (
        Path("artifacts") / tracking_cfg.run_name
        if tracking_cfg.run_name
        else Path("artifacts/latest")
    )

    # Artifact Bundle
    artifact_builder = (
        TemperatureArtifactBuilder(artifact_cfg) if artifact_cfg.enabled else None
    )

    ctx = RunContext(
        tracking_cfg=tracking_cfg,
        artifact_cfg=artifact_cfg,
        diagnostics_cfg=diagnostics_cfg,
        run_meta=run_meta,
        out_dir=out_dir,
        artifact_builder=artifact_builder,
    )

    ctx.start()

    # ----------------------------
    # Grid & Init State
    # ----------------------------
    X, Y = make_xy_grid(H, W, device=device, dtype=dtype)

    T0 = make_smooth_preheat_field(
        H=H,
        W=W,
        device=device,
        dtype=dtype,
        ambient=sim_config.T_ambient,
        amplitude=0.02,
        kernel_size=51,
        sigma=12.0,
        clamp_min=sim_config.T_ambient,
    )

    state = SimulationState(T=T0, t=0.0, step=0)

    E_acc = torch.zeros_like(T0)
    t_since = torch.full_like(T0, 1.0)
    stepper = TimeStepper(sim_config, mat)

    # ----------------------------
    # Dot Schedule
    # ----------------------------
    xs = torch.linspace(0.35, 0.65, 10).tolist()
    # xs = xs + list(reversed(xs))
    xs = xs + list(reversed(xs))
    ys = [0.5] * len(xs)

    events: list[DotEvent] = [
        DotEvent(x=float(x), y=float(y), power=2.0, dwell=0.04, travel=0.01, sigma=0.02)
        for x, y in zip(xs, ys, strict=True)
    ]

    # Legacy Recorders
    rec = StateRecorder(keys=["T", "E_acc", "t_since", "cooling_rate"])
    snapshot_every = 1  # Keep high freq for legacy recorder?

    T_peak_global = state.T.clone()
    cooling_delta_t = float(events[0].dwell)
    cooling_delta_steps = max(1, int(round(cooling_delta_t / dt)))
    cooling_delta_t_eff = cooling_delta_steps * dt

    t_peak_step = torch.zeros((1, 1, H, W), device=device, dtype=torch.long)
    after_set = torch.zeros((1, 1, H, W), device=device, dtype=torch.bool)
    cooling_rate_map = torch.full(
        (1, 1, H, W), float("nan"), device=device, dtype=dtype
    )

    rec.add(
        state.t,
        -1,
        {
            "T": state.T,
            "E_acc": E_acc,
            "t_since": t_since,
            "cooling_rate": cooling_rate_map,
        },
    )

    # ----------------------------
    # Simulation Loop
    # ----------------------------
    global_step = 0
    total_steps_est = sum(
        [max(1, int(ev.dwell / dt)) + max(1, int(ev.travel / dt)) for ev in events]
    )
    logger.info(
        f"Starting simulation: {len(events)} events, approx {total_steps_est} steps."
    )

    current_scan_pos = None

    try:
        for i, ev in enumerate(events):
            source_config = GaussianSourceConfig(
                power=ev.power, eta=ev.eta, sigma=ev.sigma
            )
            source = GaussianBeam(source_config)
            spot = source.intensity(X, Y, None, x0=ev.x, y0=ev.y)
            current_scan_pos = (ev.x, ev.y)

            # --- Dwell ---
            steps_dwell = max(1, int(ev.dwell / dt))
            dt_eff = ev.dwell / steps_dwell

            for _ in range(steps_dwell):
                global_step += 1
                ctx.on_step_start(global_step, state.T)

                state = stepper.step_explicit_euler(state, dt_eff, Q_ext=spot)

                # Peak logic
                curr_T = state.T
                new_peak_mask = curr_T > T_peak_global
                if new_peak_mask.any():
                    T_peak_global = torch.where(new_peak_mask, curr_T, T_peak_global)
                    t_peak_step = torch.where(
                        new_peak_mask,
                        torch.full_like(t_peak_step, state.step),
                        t_peak_step,
                    )

                # Tracking logs
                step_meta = {
                    "scan_power": ev.power,
                    "scan_pos": current_scan_pos,
                    "event_idx": i,
                }
                ctx.log_step(global_step, state.T, meta=step_meta)
                ctx.maybe_snapshot(global_step, state.T, meta=step_meta)

            # --- Map Update ---
            E_acc, t_since = update_history_maps(
                E_acc, t_since, spot, ev.dwell, ev.power * ev.dwell
            )

            # --- Travel ---
            current_scan_pos = None  # Laser off
            if ev.travel > 0.0:
                steps_travel = max(1, int(ev.travel / dt))
                dt_travel = ev.travel / steps_travel

                for _ in range(steps_travel):
                    global_step += 1
                    ctx.on_step_start(global_step, state.T)

                    state = stepper.step_explicit_euler(state, dt_travel, Q_ext=None)
                    t_since += dt_travel

                    # Cooling rate logic...
                    steps_since_peak = state.step - t_peak_step
                    due = (steps_since_peak >= cooling_delta_steps) & (~after_set)
                    if due.any():
                        new_cr = (T_peak_global - state.T) / max(
                            cooling_delta_t_eff, 1e-12
                        )
                        cooling_rate_map = torch.where(due, new_cr, cooling_rate_map)
                        after_set = after_set | due

                    # Tracking logs
                    step_meta = {"scan_power": 0.0, "scan_pos": None, "event_idx": i}
                    ctx.log_step(global_step, state.T, meta=step_meta)
                    ctx.maybe_snapshot(global_step, state.T, meta=step_meta)

            # Legacy snapshot
            if i % snapshot_every == 0:
                rec.add(
                    state.t,
                    i,
                    {
                        "T": state.T,
                        "E_acc": E_acc,
                        "t_since": t_since,
                        "cooling_rate": cooling_rate_map,
                    },
                )

        # End loop
        logger.info("Simulation loop finished.")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        ctx.end(state.T, status="FAILED")
        raise e

    # ----------------------------
    # Finalize
    # ----------------------------
    nan_mask = torch.isnan(cooling_rate_map)
    fallback_cr = (T_peak_global - state.T) / max(cooling_delta_t_eff, 1e-12)
    cooling_rate_map = torch.where(nan_mask, fallback_cr, cooling_rate_map)

    # Legacy save
    snapshot_state = rec.to_snapshot_state()
    states = ThermalStates(
        final=FinalState(
            T=state.T.detach().cpu(),
            E_acc=E_acc.detach().cpu(),
            t_since=t_since.detach().cpu(),
            T_peak_global=T_peak_global.detach().cpu(),
            cooling_rate=cooling_rate_map.detach().cpu(),
        ),
        snapshots=snapshot_state,
        meta=StateMeta(
            H=H,
            W=W,
            dt=dt,
            loss_h=sim_config.loss_h,
            T_ambient=sim_config.T_ambient,
            description="Time-resolved LPBF dot-by-dot simulation (Refactored)",
            cooling_delta_t=cooling_delta_t,
            cooling_delta_t_eff=cooling_delta_t_eff,
            cooling_delta_steps=cooling_delta_steps,
        ),
    )

    legacy_out_path = (
        ctx.dirs.get("states", Path(".")) / "states.pt"
        if hasattr(ctx, "dirs")
        else Path("states.pt")
    )
    save_state(states, legacy_out_path)
    logger.info(f"Saved states to {legacy_out_path}")

    # End Context
    ctx.end(state.T, meta={"final_steps": global_step})


if __name__ == "__main__":
    main()
