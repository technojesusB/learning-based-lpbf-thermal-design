"""Dataset-generation pipeline helpers shared by the CLI script and notebooks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.sampling import (
    _topup_sample_indices,
    heuristic_sample_indices,
    randomize_material,
)
from ..physics.material import MaterialConfig
from ..scan.path_generator import generate_pulsed_path
from ..scan.sources import GaussianSourceConfig

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class TrajectoryPlan:
    """All deterministic inputs for one dataset run, computed before the loop.

    Frozen so it can be passed around safely between notebook cells and the
    worker without risk of accidental mutation.
    """

    run_idx: int
    run_seed: int
    mat_key: str
    mat_cfg: MaterialConfig
    power: float
    exposure_time: float
    sigma: float
    point_distance: float
    hatch_spacing: float
    pattern: str
    path_kwargs: dict[str, Any]
    path_points: np.ndarray
    sample_indices: frozenset[int]
    beam_cfg: GaussianSourceConfig


def prepare_trajectory(
    args: argparse.Namespace,
    run_idx: int,
    material_zoo: dict[str, MaterialConfig],
) -> TrajectoryPlan:
    """Deterministically generate the trajectory plan for *run_idx*.

    Sets global NumPy and Python ``random`` seeds exactly as ``run_worker``
    does, so the notebook can call this first (to display the path preview)
    and then pass the returned plan to ``run_worker`` without any RNG drift.
    """
    run_seed = args.seed + run_idx
    random.seed(run_seed)
    np.random.seed(run_seed)
    # torch seed is set separately inside run_worker to avoid importing torch here

    target_materials = [m.strip() for m in args.materials.split(",")]
    mat_key = target_materials[run_idx % len(target_materials)]
    base_mat = material_zoo[mat_key]
    mat_cfg = randomize_material(base_mat, scale=0.1)

    power = random.uniform(150.0, 350.0)
    exposure_time = random.uniform(30e-6, 80e-6)
    sigma = random.uniform(15e-6, 40e-6)
    _pd_base = (4.0 * sigma) * (2.0 / 3.0)
    point_distance = _pd_base * random.uniform(0.9, 1.1)
    hatch_spacing = _pd_base * random.uniform(0.9, 1.1)

    pattern = random.choice(["zigzag", "raster", "island"])

    from ..core.config import SimulationConfig
    from ..utils.units import LengthUnit

    sim_cfg = SimulationConfig(
        Lx=1.0,
        Ly=0.5,
        Lz=0.125,
        Nx=args.nx,
        Ny=args.ny,
        Nz=args.nz,
        length_unit=LengthUnit.MILLIMETERS,
    )
    Lx_m, Ly_m = sim_cfg.Lx_m, sim_cfg.Ly_m

    path_kwargs: dict[str, Any] = dict(
        pattern=pattern,
        Lx=Lx_m,
        Ly=Ly_m,
        point_distance=point_distance,
        hatch_spacing=hatch_spacing,
        angle_deg=random.uniform(0, 180),
    )
    if pattern == "island":
        path_kwargs["island_size"] = random.uniform(1e-4, 4e-4)

    path_points = generate_pulsed_path(**path_kwargs)

    sample_indices_list = heuristic_sample_indices(
        len(path_points), args.samples_per_run
    )
    sample_indices_list = _topup_sample_indices(
        sample_indices_list, len(path_points), args.samples_per_run
    )

    beam_cfg = GaussianSourceConfig(power=power, eta=0.4, sigma=sigma, depth=1.0e-4)

    return TrajectoryPlan(
        run_idx=run_idx,
        run_seed=run_seed,
        mat_key=mat_key,
        mat_cfg=mat_cfg,
        power=power,
        exposure_time=exposure_time,
        sigma=sigma,
        point_distance=point_distance,
        hatch_spacing=hatch_spacing,
        pattern=pattern,
        path_kwargs=path_kwargs,
        path_points=path_points,
        sample_indices=frozenset(sample_indices_list),
        beam_cfg=beam_cfg,
    )


def save_path_preview(
    plan: TrajectoryPlan,
    out_path: Path,
    Lx_m: float,
    Ly_m: float,
) -> Path:
    """Render and save a scan-path preview PNG; return the path."""
    import matplotlib.pyplot as plt

    pts = plan.path_points
    sample_list = np.array(sorted(plan.sample_indices))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pts[:, 0] * 1e3, pts[:, 1] * 1e3, "r-", lw=0.5, alpha=0.3)
    ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, c="black", s=2)
    if len(sample_list):
        ax.scatter(
            pts[sample_list, 0] * 1e3,
            pts[sample_list, 1] * 1e3,
            c="blue",
            s=8,
            zorder=5,
        )
    ax.set_xlim(0, Lx_m * 1e3)
    ax.set_ylim(0, Ly_m * 1e3)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title(
        f"Run {plan.run_idx} — {plan.mat_key} | {plan.pattern} | "
        f"P={plan.power:.0f}W σ={plan.sigma * 1e6:.0f}µm"
    )
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
