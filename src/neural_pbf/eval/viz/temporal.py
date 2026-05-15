"""Temporal trajectory plots: peak tracking, thermocouple probes, error evolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt

from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")

if TYPE_CHECKING:
    from neural_pbf.eval.rollout.engine import RolloutResult


def peak_tracking_plot(
    result: RolloutResult,
    *,
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Line plot of T_max evolution for GT and Pred over the rollout."""
    steps = list(range(len(result.pred_T)))
    T_max_pred = [T.max().item() for T in result.pred_T]
    T_max_gt = [snap.T.max().item() for snap in result.gt_snapshots[: len(steps)]]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(THEME.base.bg_figure)
        ax.set_facecolor(THEME.base.bg_axis)
        ax.grid(
            THEME.base.grid.enabled,
            which=THEME.base.grid.which,
            linestyle=THEME.base.grid.linestyle,
            alpha=THEME.base.grid.alpha,
            color=THEME.base.grid.color,
        )

        ax.plot(steps, T_max_gt, label="GT T_max", color=THEME.base.color_gt)
        ax.plot(
            steps,
            T_max_pred,
            label=f"{result.stepper_name} T_max",
            linestyle="--",
            color=THEME.base.color_train,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Peak Temperature [K]")
        ax.set_title(
            f"Peak Temperature Evolution — {result.stepper_name} ({result.mode})"
        )
        ax.legend(facecolor=THEME.base.bg_axis, edgecolor=THEME.base.grid.color)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


def probe_history_plot(
    result: RolloutResult,
    probes: list[tuple[str, int, int, int | None]],
    *,
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Virtual thermocouple temperature history.

    Args:
        result: RolloutResult.
        probes: List of ``(name, ix, iy, iz_or_None)`` tuples.
        savepath: Optional output path.
    """
    n_steps = len(result.pred_T)
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(THEME.base.bg_figure)
        ax.set_facecolor(THEME.base.bg_axis)
        ax.grid(
            THEME.base.grid.enabled,
            which=THEME.base.grid.which,
            linestyle=THEME.base.grid.linestyle,
            alpha=THEME.base.grid.alpha,
            color=THEME.base.grid.color,
        )

        for probe_name, ix, iy, iz in probes:
            pred_hist: list[float] = []
            gt_hist: list[float] = []
            for i in range(n_steps):
                T_p = result.pred_T[i].detach().cpu().float()
                T_g = result.gt_snapshots[i].T.detach().cpu().float()
                if T_p.ndim == 5 and iz is not None:
                    pred_hist.append(T_p[0, 0, iz, iy, ix].item())
                    gt_hist.append(T_g[0, 0, iz, iy, ix].item())
                else:
                    pred_hist.append(T_p[0, 0, iy, ix].item())
                    gt_hist.append(T_g[0, 0, iy, ix].item())

            ax.plot(gt_hist, label=f"{probe_name} GT", color=THEME.base.color_gt)
            ax.plot(pred_hist, label=f"{probe_name} Pred", linestyle="--")

        ax.set_xlabel("Step")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Virtual Thermocouple History")
        ax.legend(facecolor=THEME.base.bg_axis, edgecolor=THEME.base.grid.color)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


def error_evolution_plot(
    result: RolloutResult,
    *,
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Global MAE per step as a line plot."""
    steps = list(range(len(result.per_step_metrics)))
    maes = [m["mae_global"] for m in result.per_step_metrics]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(THEME.base.bg_figure)
        ax.set_facecolor(THEME.base.bg_axis)
        ax.grid(
            THEME.base.grid.enabled,
            which=THEME.base.grid.which,
            linestyle=THEME.base.grid.linestyle,
            alpha=THEME.base.grid.alpha,
            color=THEME.base.grid.color,
        )

        ax.plot(steps, maes, marker="o", markersize=3, color=THEME.base.color_fm)
        ax.set_xlabel("Step")
        ax.set_ylabel("MAE [K]")
        ax.set_title(f"Error Evolution — {result.stepper_name} ({result.mode})")
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig
