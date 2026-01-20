# experiments/stateful_time_resolved.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig
from neural_pbf.schemas.state import FinalState, StateMeta, ThermalStates
from neural_pbf.utils.grid import make_xy_grid
from neural_pbf.utils.history import make_smooth_preheat_field
from neural_pbf.utils.io import save_state
from neural_pbf.utils.state_recorder import StateRecorder


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
    # E_acc += spot * (energy / spot.sum()) ?
    # Legacy logic: "energy" is total Joules. spot is intensity shape.
    # If we assume spot is normalized such that sum(spot)*dt = energy...
    # For now, let's just assume simple accumulation of the source field integrated over time:
    # E_acc += Q * dt
    # But here we just want a qualitative proxy.
    E_acc = torch.where(hit_mask, E_acc + energy, E_acc)

    # 4. Reset t_since where hit
    t_since = torch.where(hit_mask, torch.zeros_like(t_since), t_since)

    return E_acc, t_since


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # ----------------------------
    # Domain & Config
    # ----------------------------
    H = W = 128
    # Domain size 1.0 x 1.0 mm (arbitrary)
    Lx = 1.0
    Ly = 1.0

    sim_config = SimulationConfig(
        Lx=Lx,
        Ly=Ly,
        Nx=W,
        Ny=H,
        length_unit="m",  # Interpretation depends on usage, but code below uses normalized 0-1 coords mostly
        dt_base=6.0e-4,  # Will be overridden by stepping loop
        T_ambient=0.05,
        loss_h=0.2,
    )

    # Explicit dx, dy for manual grid creation (legacy Grid was 0..1 normalized?)
    # In legacy: dx = 1.0 / (W - 1). This implies Lx=1.0 (unitless or normalized).
    # We'll stick to that interpretation.
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
        rho=1.0,
    )

    dt = 6.0e-4

    # ----------------------------
    # Grid
    # ----------------------------
    # make_xy_grid returns tuple(X, Y)
    X, Y = make_xy_grid(H, W, device=device, dtype=dtype)

    # ----------------------------
    # Initial temperature
    # ----------------------------
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
    # T0 = torch.full((1, 1, H, W), sim_config.T_ambient, device=device, dtype=dtype)

    # ----------------------------
    # State Init
    # ----------------------------
    state = SimulationState(
        T=T0,
        t=0.0,
        step=0,
        # max_T will be auto-init
    )

    # Auxiliary fields (not in SimulationState)
    E_acc = torch.zeros_like(T0)
    t_since = torch.full_like(T0, 1.0)  # t_since_init=1.0

    # Stepper
    stepper = TimeStepper(sim_config, mat)
    # We need to monkey-patch or ensure stepper uses our manual dx/dy if config doesn't match?
    # Config has Lx=1, Nx=128 => dx = 1/127 approx 0.0078
    # Legacy dx = 1/127. So it matches.
    # Note: sim_config.dx is calculated property.

    # ----------------------------
    # Dot sequence (demo path)
    # ----------------------------
    xs = torch.linspace(0.35, 0.65, 10).tolist()
    # xs = xs + list(reversed(xs)) # Legacy logic
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
    # Recorder & Trackers
    # ----------------------------
    rec = StateRecorder(keys=["T", "E_acc", "t_since", "cooling_rate"])
    snapshot_every = 1

    T_peak_global = state.T.clone()

    cooling_delta_t = float(events[0].dwell)
    cooling_delta_steps = max(1, int(round(cooling_delta_t / dt)))
    cooling_delta_t_eff = cooling_delta_steps * dt

    t_peak_step = torch.zeros((1, 1, H, W), device=device, dtype=torch.long)
    after_set = torch.zeros((1, 1, H, W), device=device, dtype=torch.bool)
    cooling_rate_map = torch.full(
        (1, 1, H, W), float("nan"), device=device, dtype=dtype
    )

    # Initial snapshot
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
    # Event loop
    # ----------------------------
    for i, ev in enumerate(events):
        # 1. Setup Heat Source
        # Q field is effectively constant during the dwell of a stationary dot
        source_config = GaussianSourceConfig(power=ev.power, eta=ev.eta, sigma=ev.sigma)
        source = GaussianBeam(source_config)

        # Compute Intensity Field Q [W/m^3 or W/m^2 depending on dim]
        # Our sim is 2D, physics assumes surface source is added as source term
        spot = source.intensity(X, Y, None, x0=ev.x, y0=ev.y)

        # ---- Laser ON (Dwell) ----
        steps_dwell = max(1, int(ev.dwell / dt))
        dt_eff = ev.dwell / steps_dwell

        for _ in range(steps_dwell):
            state = stepper.step_explicit_euler(state, dt_eff, Q_ext=spot)

            # Peak tracking (Manual b/c we need timing)
            # state.max_T is updated by stepper, but we need t_peak_step for cooling calc
            # Note: stepper updates state.max_T to be max(prev, current)
            # We need to check if current > old_global_peak
            curr_T = state.T
            new_peak_mask = curr_T > T_peak_global

            if new_peak_mask.any():
                T_peak_global = torch.where(new_peak_mask, curr_T, T_peak_global)
                t_peak_step = torch.where(
                    new_peak_mask, torch.full_like(t_peak_step, state.step), t_peak_step
                )
                # Reset cooling rate logic for these pixels?
                # The logic: cooling rate is measured dt_cool AFTER peak.
                # If we have a new peak, we reset the timer.

        # History Map Update (Active zone)
        # Use the spot field we computed
        E_acc, t_since = update_history_maps(
            E_acc,
            t_since,
            spot,
            ev.dwell,
            ev.power * ev.dwell,  # approx energy
        )

        # ---- Laser OFF (Travel) ----
        if ev.travel > 0.0:
            steps_travel = max(1, int(ev.travel / dt))
            dt_travel = ev.travel / steps_travel

            for _ in range(steps_travel):
                state = stepper.step_explicit_euler(state, dt_travel, Q_ext=None)

                # Update t_since
                t_since += dt_travel

                # Check for cooling rate capture
                # Current step vs t_peak_step
                steps_since_peak = state.step - t_peak_step

                # Check pixels that are due for measurement AND haven't been set yet for this peak
                # "after_set" logic from legacy was a bit complex to handle "set once per peak".
                # Simplified:
                due = (steps_since_peak >= cooling_delta_steps) & (~after_set)

                # However, "after_set" needs to be reset when a new peak happens.
                # Let's check legacy logic:
                # better = T_peak_interval > T_peak_global
                # t_peak_step = ...
                # IMPLICITLY, if we update t_peak_step, we should probably clear after_set?
                # Legacy didn't explicitly clear after_set?
                # Ah, legacy tracked "T_after_peak" and "after_set" per interval?

                # Re-implementing simplified cooling rate logic:
                # CR = (T_peak - T(t_peak + delta)) / delta_t
                if due.any():
                    new_cr = (T_peak_global - state.T) / max(cooling_delta_t_eff, 1e-12)
                    cooling_rate_map = torch.where(due, new_cr, cooling_rate_map)
                    after_set = after_set | due

            # Reset after_set only where new peaks happened?
            # This logic is tricky to replicate perfectly without the full block.
            # But the Stepper now captures instantaneous cooling rate at solidification!
            # state.cooling_rate (from Stepper) = dT/dt at solidification.
            # The experiment wants "Cooling rate over Delta T" (CR_delta).
            # We'll stick to the "due" logic.
            pass

        # ---- Snapshot ----
        if i % snapshot_every == 0:
            rec.add(
                state.t,
                i,
                {
                    "T": state.T,
                    "E_acc": E_acc,  # state.E_acc is not in SimulationState
                    "t_since": t_since,
                    "cooling_rate": cooling_rate_map,
                },
            )

        print(
            f"event {i:02d} | "
            f"T.max={float(state.T.max()):.4f} | "
            f"E_acc.max={float(E_acc.max()):.4f} | "
            f"t={state.t:.3f}"
        )

    # ----------------------------
    # Finalize
    # ----------------------------
    nan_mask = torch.isnan(cooling_rate_map)
    fallback_cr = (T_peak_global - state.T) / max(cooling_delta_t_eff, 1e-12)
    cooling_rate_map = torch.where(nan_mask, fallback_cr, cooling_rate_map)

    # Pack states
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

    out_path = "artifacts/runs/run_0002/states.pt"
    save_state(states, out_path)
    print(f"Saved states to {out_path}")


if __name__ == "__main__":
    main()
