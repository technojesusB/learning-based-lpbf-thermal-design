# experiments/stateful_time_resolved.py
from __future__ import annotations

import torch

from utils.grid import make_xy_grid
from utils.history import make_smooth_preheat_field
from datagenerator.state import init_state, update_history_maps
from datagenerator.stepper import advance_temperature
from physics.material import MaterialConfig
from scan.dots import DotEvent, make_dot_Q_fn, make_zero_Q_fn


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    H = W = 128
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    # --- physics / numerics ---
    mat = MaterialConfig(
        k_powder=0.0012, k_solid=0.018, k_liquid=0.012,
        cp_base=1.0, T_solidus=0.60, T_liquidus=0.70,
        latent_heat_L=0.35, latent_width=0.03,
    )
    T_ambient = 0.05
    loss_h = 0.2

    # timestep for global integration (must be stable)
    dt = 6.0e-4

    X, Y = make_xy_grid(H, W, device=device, dtype=dtype)

    # --- initial state with spatial history ---
    T0 = make_smooth_preheat_field(
        H=H, W=W, device=device, dtype=dtype,
        ambient=T_ambient, amplitude=0.02,
        kernel_size=51, sigma=12.0, clamp_min=T_ambient
    )
    state = init_state(H, W, device, dtype, T0=T0, t_since_init=1.0)

    # --- define a dot sequence (later: hatchlines -> dots) ---
    xs = torch.linspace(0.35, 0.65, 10).tolist()
    xs = xs + list(reversed(xs))  # go back
    ys = [0.5] * len(xs)

    events = []
    for x, y in zip(xs, ys):
        events.append(DotEvent(
            x=float(x), y=float(y),
            power=2.0,
            dwell=0.04,      # laser on
            travel=0.01,     # laser off before next dot
            sigma=0.02,
            eta=1.0
        ))

    # --- global time loop over events ---
    t_global = 0.0
    Q0 = make_zero_Q_fn(state.T)

    T_peak_global = state.T.clone()

    for i, ev in enumerate(events):
        # (1) Laser ON (dwell)
        Q_fn, spot = make_dot_Q_fn(X, Y, ev)
        state.T, T_peak_interval = advance_temperature(
            state.T, mat, dx, dy,
            T_ambient=T_ambient, loss_h=loss_h,
            dt=dt, duration=ev.dwell, Q_fn=Q_fn, t0=t_global
        )
        t_global += ev.dwell

        # update peak tracking
        T_peak_global = torch.maximum(T_peak_global, T_peak_interval)

        # history bookkeeping (optional but useful)
        energy = ev.power * ev.dwell
        state = update_history_maps(state, spot=spot, dt_event=ev.dwell, energy=energy, reset_threshold=0.2)

        # (2) Laser OFF (travel time)
        if ev.travel > 0:
            state.T, _ = advance_temperature(
                state.T, mat, dx, dy,
                T_ambient=T_ambient, loss_h=loss_h,
                dt=dt, duration=ev.travel, Q_fn=Q0, t0=t_global
            )
            t_global += ev.travel
            state.t_since += ev.travel  # since no hit during travel

        print(
            f"event {i:02d} | T_final.max={float(state.T.max()):.4f} | "
            f"T_peak_global.max={float(T_peak_global.max()):.4f} | "
            f"E_acc.max={float(state.E_acc.max()):.4f} | t={t_global:.3f}"
        )


if __name__ == "__main__":
    main()
