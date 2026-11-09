"""Measurement logic for the large-step residual check pre-registered in docs/physics_residual_check.md."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.fm_physics import enthalpy, heat_residual_si
from neural_pbf.physics.material import MaterialConfig

Variant = tuple[str, str]
VARIANTS: tuple[Variant, ...] = tuple(
    (s, d) for s in ("enthalpy", "cp_in") for d in ("explicit", "implicit", "trapezoid")
)
SOLVER_FORM: Variant = ("cp_in", "explicit")
MIN_DT_K = 50.0
SUBSTEP_TOL = 1e-3
SAFETY_FACTOR = 0.9  # step_adaptive's default, which the generator uses


def relative_residual(
    R: Tensor, T_in: Tensor, T_out: Tensor, mat: MaterialConfig, dt_s: float, min_dT: float | None = None
) -> float:
    """‖R‖ / ‖ρ·(H(T_out) − H(T_in))/dt‖ over the interior, optionally only where |T_out − T_in| > min_dT.

    Returns NaN when no voxel qualifies, and NaN (0/0) when nothing changed in the selection, which only happens
    without the filter on a step that stores no heat.
    """
    T_in_i, T_out_i = T_in[..., 1:-1, 1:-1, 1:-1], T_out[..., 1:-1, 1:-1, 1:-1]
    stored = mat.rho * (enthalpy(T_out_i, mat) - enthalpy(T_in_i, mat)) / dt_s
    sel = torch.ones_like(T_in_i, dtype=torch.bool) if min_dT is None else (T_out_i - T_in_i).abs() > min_dT
    if not bool(sel.any()):
        return math.nan
    return float(R[sel].norm() / stored[sel].norm())


def _residuals(
    T_in: Tensor, T_out: Tensor, Q: Tensor, mask_in: Tensor, mask_out: Tensor, mat: MaterialConfig,
    sim: SimulationConfig, dt_s: float, min_dT: float | None,
) -> dict[Variant, float]:  # fmt: skip
    T_in, T_out, Q = T_in.double(), T_out.double(), Q.double()
    out = {}
    for storage, diffusion in VARIANTS:
        R = heat_residual_si(
            T_in, T_out, Q, mask_in, mat, dt_s, sim.dx, sim.dy, sim.dz, storage, diffusion, mask_out=mask_out
        )
        out[(storage, diffusion)] = relative_residual(R, T_in, T_out, mat, dt_s, min_dT)
    return out


def substep_check(
    T_in: Tensor, mask_in: Tensor, Q: Tensor, mat: MaterialConfig, sim: SimulationConfig, exposure_time: float,
    dtype: torch.dtype = torch.float32,
) -> dict[Variant, float]:  # fmt: skip
    """Run one solver substep live in `dtype` and return every variant's relative residual on it (no filter).

    The mask is promoted above T_solidus first, which the Triton kernel does inside each substep. In float32 the
    solver form sits near 1e-4 on a full domain from rounding alone; in float64 it is exact to machine precision.
    """
    T = T_in.to(dtype).clone()
    mask = (mask_in.bool() | (mat.T_solidus <= T)).to(torch.uint8)
    state = SimulationState.zeros(sim, device=T.device, dtype=dtype)
    state.T, state.material_mask = T.clone(), mask
    stepper = TimeStepper(sim, mat)
    dt_crit = stepper.estimate_stability_dt(state) * SAFETY_FACTOR
    dt_sub = exposure_time / math.ceil(exposure_time / dt_crit)
    T_out = stepper.step_explicit_euler(state, dt_sub, Q.to(dtype)).T
    return _residuals(T, T_out, Q, mask, mask, mat, sim, dt_sub, None)


def exposure_residuals(
    T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor, mask_out: Tensor, mat: MaterialConfig,
    sim: SimulationConfig, exposure_time: float,
) -> dict[Variant, float]:  # fmt: skip
    """Every variant's relative residual over one whole exposure, on voxels that changed by more than 50 K."""
    return _residuals(T_in, T_target, Q, mask_in, mask_out, mat, sim, exposure_time, MIN_DT_K)


def summarize(rows: Iterable[dict[Any, Any]], variants: Iterable[Variant]) -> dict[tuple[str, Variant], dict]:
    """n, empty count, median and p90 per (regime, variant); NaN rows count as empty."""
    rows = list(rows)
    table = {}
    for regime in sorted({r["regime"] for r in rows}):
        for v in variants:
            vals = np.array([r[v] for r in rows if r["regime"] == regime], dtype=float)
            ok = vals[~np.isnan(vals)]
            table[(regime, v)] = {
                "n": int(ok.size),
                "n_empty": int(vals.size - ok.size),
                "median": float(np.median(ok)) if ok.size else math.nan,
                "p90": float(np.percentile(ok, 90)) if ok.size else math.nan,
            }
    return table


def select_variant(table: dict[tuple[str, Variant], dict], eligible: set[Variant]) -> Variant | None:
    """The eligible variant with the lowest melt-regime median, or None."""
    scored = [(table[("melt", v)]["median"], v) for v in eligible if ("melt", v) in table]
    scored = [(m, v) for m, v in scored if not math.isnan(m)]
    return min(scored)[1] if scored else None
