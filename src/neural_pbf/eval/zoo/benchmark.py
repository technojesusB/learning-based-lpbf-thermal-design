"""Zoo benchmark driver: sweep a stepper across multiple materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from neural_pbf.eval.data.trajectory import Trajectory
from neural_pbf.eval.protocols import BaseStepper
from neural_pbf.eval.rollout.autoregressive import ar_summary
from neural_pbf.eval.rollout.engine import RolloutEngine

from .materials import MaterialZoo


@dataclass
class BenchmarkRecord:
    material_name: str
    in_distribution: bool
    stepper_name: str
    mode: str
    mean_mae: float
    final_mae: float
    peak_mae: float
    mean_latency_ms: float
    vram_peak_mb: float
    diverged: bool


def run_zoo_benchmark(
    stepper: BaseStepper,
    trajectories: dict[str, Trajectory],
    zoo: MaterialZoo | None = None,
    mode: str = "one_step",
    conditioning_seq: list[dict[str, Any] | None] | None = None,
) -> pd.DataFrame:
    """Evaluate *stepper* on every trajectory in *trajectories*.

    Args:
        stepper:          The stepper to benchmark.
        trajectories:     Dict mapping material name → :class:`Trajectory`.
        zoo:              Optional :class:`MaterialZoo` for ID/OOD tagging.
                          When ``None``, all materials are treated as ID.
        mode:             ``"one_step"`` or ``"autoregressive"``.
        conditioning_seq: Shared per-step conditioning sequence.

    Returns:
        :class:`pandas.DataFrame` with one row per material.
    """
    engine = RolloutEngine()
    records: list[BenchmarkRecord] = []

    for name, traj in trajectories.items():
        entry = zoo.get(name) if zoo is not None else None
        in_dist = entry.in_distribution if entry is not None else True

        result = engine.run(
            stepper=stepper,
            gt_trajectory=traj,
            mode=mode,
            conditioning_seq=conditioning_seq,
        )
        summary = ar_summary(result)
        n = len(result.latencies_s)

        records.append(
            BenchmarkRecord(
                material_name=name,
                in_distribution=in_dist,
                stepper_name=result.stepper_name,
                mode=mode,
                mean_mae=summary.get("mean_mae", float("nan")),
                final_mae=summary.get("final_mae", float("nan")),
                peak_mae=summary.get("peak_mae", float("nan")),
                mean_latency_ms=1000.0 * sum(result.latencies_s) / n
                if n > 0
                else float("nan"),
                vram_peak_mb=result.vram_peak_bytes / 1e6,
                diverged=result.diverged_at_step is not None,
            )
        )

    return pd.DataFrame([vars(r) for r in records])
