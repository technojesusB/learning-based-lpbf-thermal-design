"""High-level composer: generate all figures for a RolloutResult."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from neural_pbf.eval.rollout.engine import RolloutResult

from .spatial import profile_slices, triple_view
from .temporal import error_evolution_plot, peak_tracking_plot


def generate_report_figures(
    result: RolloutResult,
    output_dir: Path | str,
) -> list[Path]:
    """Write a standard set of evaluation figures to *output_dir*.

    Generated files:
    - ``triple_view_last.png`` — GT/Pred/Error at the final step
    - ``profile_slices_last.png`` — 1-D slices at the final step
    - ``peak_tracking.png`` — T_max over time
    - ``error_evolution.png`` — global MAE per step

    Returns:
        List of :class:`Path` objects for files actually written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if result.pred_T:
        last = len(result.pred_T) - 1
        gt_last = result.gt_snapshots[last].T
        pred_last = result.pred_T[last]

        p = out / "triple_view_last.png"
        triple_view(gt_last, pred_last, title=f"Step {last}", savepath=p)
        written.append(p)

        p = out / "profile_slices_last.png"
        profile_slices(gt_last, pred_last, savepath=p)
        written.append(p)

    p = out / "peak_tracking.png"
    peak_tracking_plot(result, savepath=p)
    written.append(p)

    p = out / "error_evolution.png"
    error_evolution_plot(result, savepath=p)
    written.append(p)

    plt.close("all")
    return written
