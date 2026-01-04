# experiments/stateful_time_resolved.py
from __future__ import annotations

import torch

from utils.grid import make_xy_grid
from utils.history import make_smooth_preheat_field
from utils.state_recorder import StateRecorder
from utils.io import save_state

from datagenerator.state import init_state, update_history_maps
from datagenerator.stepper import advance_temperature

from physics.material import MaterialConfig
from scan.dots import DotEvent, make_dot_Q_fn, make_zero_Q_fn

from schemas.state import ThermalStates, FinalState, StateMeta

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # ----------------------------
    # Domain
    # ----------------------------
    H = W = 128
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    # ----------------------------
    # Material (scaled toy values to keep explicit scheme stable)
    # ----------------------------
    mat = MaterialConfig(
        k_powder=0.0012,
        k_solid=0.018,
        k_liquid=0.012,
        cp_base=1.0,
        T_solidus=0.60,
        T_liquidus=0.70,
        latent_heat_L=0.35,
        latent_width=0.03,
        rho=1.0,
    )

    # Ambient + simple losses
    T_ambient = 0.05
    loss_h = 0.2

    # Global time step (must satisfy stability)
    dt = 6.0e-4

    # ----------------------------
    # Grids
    # ----------------------------
    X, Y = make_xy_grid(H, W, device=device, dtype=dtype)

    # ----------------------------
    # Initial history field
    # ----------------------------
    T0 = make_smooth_preheat_field(
        H=H,
        W=W,
        device=device,
        dtype=dtype,
        ambient=T_ambient,
        amplitude=0.02,
        kernel_size=51,
        sigma=12.0,
        clamp_min=T_ambient,
    )

    # ----------------------------
    # Initialize state maps
    # ----------------------------
    state = init_state(H, W, device, dtype, T0=T0, t_since_init=1.0)

    # ----------------------------
    # Define a simple dot sequence (placeholder for hatchlines later)
    # ----------------------------
    xs = torch.linspace(0.35, 0.65, 10).tolist()
    xs = xs + list(reversed(xs))  # go back over heated region
    ys = [0.5] * len(xs)

    events: list[DotEvent] = []
    for x, y in zip(xs, ys):
        events.append(
            DotEvent(
                x=float(x),
                y=float(y),
                power=2.0,
                dwell=0.04,     # laser ON duration
                travel=0.01,    # laser OFF duration before next dot
                sigma=0.02,
                eta=1.0,
            )
        )

    # ----------------------------
    # Recorder for visualization / animation
    # ----------------------------
    rec = StateRecorder(keys=["T", "E_acc", "t_since"])
    snapshot_every = 1  # store every event; increase for long runs

    # Track global peak over entire run
    T_peak_global = state.T.clone()

    # Global time
    t_global = 0.0

    # Zero heat source function for travel phases
    Q0 = make_zero_Q_fn(state.T)

    # Initial snapshot
    rec.add(t_global, -1, {"T": state.T, "E_acc": state.E_acc, "t_since": state.t_since})

    # ----------------------------
    # Event loop
    # ----------------------------
    for i, ev in enumerate(events):
        # (1) Laser ON: integrate for dwell time with constant dot heat source
        Q_fn, spot = make_dot_Q_fn(X, Y, ev)

        state.T, T_peak_interval = advance_temperature(
            T=state.T,
            mat=mat,
            dx=dx,
            dy=dy,
            T_ambient=T_ambient,
            loss_h=loss_h,
            dt=dt,
            duration=ev.dwell,
            Q_fn=Q_fn,
            t0=t_global,
        )
        t_global += ev.dwell

        # Update global peak map
        T_peak_global = torch.maximum(T_peak_global, T_peak_interval)

        # Update history maps (optional, but useful conditioning later)
        energy = ev.power * ev.dwell
        state = update_history_maps(
            state,
            spot=spot,
            dt_event=ev.dwell,
            energy=energy,
            reset_threshold=0.2,
        )

        # (2) Laser OFF: travel time with Q=0
        if ev.travel > 0.0:
            state.T, _ = advance_temperature(
                T=state.T,
                mat=mat,
                dx=dx,
                dy=dy,
                T_ambient=T_ambient,
                loss_h=loss_h,
                dt=dt,
                duration=ev.travel,
                Q_fn=Q0,
                t0=t_global,
            )
            t_global += ev.travel
            state.t_since += ev.travel  # no hits during travel

        # Snapshot after completing dwell+travel for this event
        if i % snapshot_every == 0:
            rec.add(t_global, i, {"T": state.T, "E_acc": state.E_acc, "t_since": state.t_since})

        # Console debug
        print(
            f"event {i:02d} | "
            f"T_final.max={float(state.T.max()):.4f} | "
            f"T_peak_global.max={float(T_peak_global.max()):.4f} | "
            f"E_acc.max={float(state.E_acc.max()):.4f} | "
            f"t={t_global:.3f}"
        )

    # ----------------------------
    # Pack into Pydantic state container + save
    # ----------------------------
    snapshot_state = rec.to_snapshot_state()

    states = ThermalStates(
        final=FinalState(
            T=state.T.detach().cpu(),
            E_acc=state.E_acc.detach().cpu(),
            t_since=state.t_since.detach().cpu(),
            T_peak_global=T_peak_global.detach().cpu(),
        ),
        snapshots=snapshot_state,
        meta=StateMeta(
            H=H,
            W=W,
            dt=dt,
            loss_h=loss_h,
            T_ambient=T_ambient,
            description="Time-resolved LPBF dot-by-dot thermal simulation (laser on/off with carried state)",
        ),
    )

    out_path = "artifacts/runs/run_0001/states.pt"
    save_state(states, out_path)
    print(f"Saved states to {out_path}")


if __name__ == "__main__":
    main()
