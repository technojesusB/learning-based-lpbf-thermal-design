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
    # Material (toy-scaled)
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

    T_ambient = 0.05
    loss_h = 0.2
    dt = 6.0e-4

    # ----------------------------
    # Grid
    # ----------------------------
    X, Y = make_xy_grid(H, W, device=device, dtype=dtype)

    # ----------------------------
    # Initial temperature / history
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

    state = init_state(H, W, device, dtype, T0=T0, t_since_init=1.0)

    # ----------------------------
    # Dot sequence (demo path)
    # ----------------------------
    xs = torch.linspace(0.35, 0.65, 10).tolist()
    xs = xs + list(reversed(xs))
    ys = [0.5] * len(xs)

    events: list[DotEvent] = [
        DotEvent(
            x=float(x),
            y=float(y),
            power=2.0,
            dwell=0.04,
            travel=0.01,
            sigma=0.02,
            eta=1.0,
        )
        for x, y in zip(xs, ys)
    ]

    # ----------------------------
    # Recorder (incl. cooling rate)
    # ----------------------------
    rec = StateRecorder(keys=["T", "E_acc", "t_since", "cooling_rate"])
    snapshot_every = 1

    # ----------------------------
    # Global trackers
    # ----------------------------
    T_peak_global = state.T.clone()

    cooling_delta_t = float(events[0].dwell)            # Option A
    cooling_delta_steps = max(1, int(round(cooling_delta_t / dt)))
    cooling_delta_t_eff = cooling_delta_steps * dt

    t_peak_step = torch.zeros((1, 1, H, W), device=device, dtype=torch.long)
    T_after_peak = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    after_set = torch.zeros((1, 1, H, W), device=device, dtype=torch.bool)

    cooling_rate_map = torch.full(
        (1, 1, H, W), float("nan"), device=device, dtype=dtype
    )

    global_step = 0
    t_global = 0.0
    Q0 = make_zero_Q_fn(state.T)

    # ----------------------------
    # Initial snapshot
    # ----------------------------
    rec.add(
        t_global,
        -1,
        {
            "T": state.T,
            "E_acc": state.E_acc,
            "t_since": state.t_since,
            "cooling_rate": cooling_rate_map,
        },
    )

    # ----------------------------
    # Event loop
    # ----------------------------
    for i, ev in enumerate(events):
        # ---- Laser ON ----
        Q_fn, spot = make_dot_Q_fn(X, Y, ev)

        state.T, T_peak_interval, global_step = advance_temperature(
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
            step0=global_step,
        )
        t_global += ev.dwell

        # Peak update
        better = T_peak_interval > T_peak_global
        T_peak_global = torch.where(better, T_peak_interval, T_peak_global)
        t_peak_step = torch.where(
            better, torch.full_like(t_peak_step, global_step), t_peak_step
        )

        # History maps
        state = update_history_maps(
            state,
            spot=spot,
            dt_event=ev.dwell,
            energy=ev.power * ev.dwell,
            reset_threshold=0.2,
        )

        # ---- Laser OFF ----
        if ev.travel > 0.0:
            state.T, _, global_step = advance_temperature(
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
                step0=global_step,
            )
            t_global += ev.travel
            state.t_since += ev.travel

        # ---- Cooling-rate capture ----
        due = (global_step >= (t_peak_step + cooling_delta_steps)) & (~after_set)

        T_after_peak = torch.where(due, state.T, T_after_peak)
        after_set = after_set | due

        new_cr = (T_peak_global - state.T) / max(cooling_delta_t_eff, 1e-12)
        cooling_rate_map = torch.where(due, new_cr, cooling_rate_map)

        # ---- Snapshot ----
        if i % snapshot_every == 0:
            rec.add(
                t_global,
                i,
                {
                    "T": state.T,
                    "E_acc": state.E_acc,
                    "t_since": state.t_since,
                    "cooling_rate": cooling_rate_map,
                },
            )

        print(
            f"event {i:02d} | "
            f"T_final.max={float(state.T.max()):.4f} | "
            f"T_peak_global.max={float(T_peak_global.max()):.4f} | "
            f"E_acc.max={float(state.E_acc.max()):.4f} | "
            f"t={t_global:.3f}"
        )

    # ----------------------------
    # Finalize cooling rate
    # ----------------------------
    nan_mask = torch.isnan(cooling_rate_map)
    fallback_cr = (T_peak_global - state.T) / max(cooling_delta_t_eff, 1e-12)
    cooling_rate_map = torch.where(nan_mask, fallback_cr, cooling_rate_map)

    # ----------------------------
    # Pack states
    # ----------------------------
    snapshot_state = rec.to_snapshot_state()

    states = ThermalStates(
        final=FinalState(
            T=state.T.detach().cpu(),
            E_acc=state.E_acc.detach().cpu(),
            t_since=state.t_since.detach().cpu(),
            T_peak_global=T_peak_global.detach().cpu(),
            cooling_rate=cooling_rate_map.detach().cpu(),
        ),
        snapshots=snapshot_state,
        meta=StateMeta(
            H=H,
            W=W,
            dt=dt,
            loss_h=loss_h,
            T_ambient=T_ambient,
            description="Time-resolved LPBF dot-by-dot simulation with online cooling-rate tracking",
            cooling_delta_t=cooling_delta_t,
            cooling_delta_t_eff=cooling_delta_t_eff,
            cooling_delta_steps=cooling_delta_steps,
        ),
    )

    out_path = "artifacts/runs/run_0002/states.pt"
    save_state(states, out_path)
    print(f"Saved states to {out_path}")


if __name__ == "__main__":
    main()
