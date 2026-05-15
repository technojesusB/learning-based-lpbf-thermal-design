"""Physical fidelity dashboard: IoU, Depth, T_max, and Hotspot Offset (2×3 layout)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _palette(
    names: list[str], color_map: dict[str, str] | None = None
) -> dict[str, str]:
    """Dynamically assign colors, prioritizing the provided color_map."""
    if color_map:
        # Filter map to only include requested names
        return {n: color_map[n] for n in names if n in color_map}

    out: dict[str, str] = {}
    palette = THEME.physical.color_palette
    for i, n in enumerate(names):
        out[str(n)] = palette[i % len(palette)]
    return out


def _apply_grid(ax, axis=None):
    """Apply grid settings from the official schema."""
    g = THEME.physical.grid
    ax.grid(
        g.enabled,
        which=g.which,
        axis=axis or g.axis,
        linestyle=g.linestyle,
        alpha=g.alpha,
        color=g.color,
    )


def plot_physical_dashboard(
    df: pd.DataFrame,
    output_path: Path | str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Write a 2×3 high-fidelity physical benchmarking dashboard.

    Layout (2 rows × 3 columns):
        [0,0] Meltpool IoU Distribution (Box + Strip)
        [0,1] Meltpool Depth Error Distribution (Box + Strip)
        [0,2] Meltpool Depth: GT vs Pred (Scatter)
        [1,0] Hotspot Offset Distribution (Box + Strip)
        [1,1] T_max Error Distribution (Box + Strip)
        [1,2] T_max: GT vs Pred (Scatter)

    Args:
        df:           DataFrame with columns Model, Sample, IoU, Depth_GT,
                      Depth_Pred, T_max_GT, T_max_Pred, T_max_Error, Offset_vox.
        output_path:  Destination PNG path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("Empty DataFrame passed to plot_physical_dashboard.")

    # Pre-calculate depth error if not present
    if (
        "Depth_Error" not in df.columns
        and "Depth_Pred" in df.columns
        and "Depth_GT" in df.columns
    ):
        df["Depth_Error"] = df["Depth_Pred"] - df["Depth_GT"]

    # --- Setup ---
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        2, 3, figsize=THEME.physical.figsize, dpi=THEME.physical.dpi
    )
    fig.patch.set_facecolor(THEME.physical.bg_figure)

    # Sort models for stable color assignment
    names = sorted(df["Model"].unique().tolist())
    pal = _palette(names, color_map=color_map)

    # --- Helper: Distribution Plot (Box + Strip) ---
    def draw_dist(ax, y_col, title, y_label, ylim=None):
        sns.boxplot(
            data=df,
            x="Model",
            y=y_col,
            hue="Model",
            ax=ax,
            palette=pal,
            legend=False,
            whis=100,
            fliersize=0,
            zorder=2,
            boxprops=dict(alpha=THEME.physical.alpha_raw),
            whiskerprops=dict(
                color=THEME.physical.color_gt,
                linewidth=THEME.physical.line_width_whisker,
                alpha=THEME.physical.alpha_raw,
            ),
            capprops=dict(
                color=THEME.physical.color_gt,
                linewidth=THEME.physical.line_width_whisker,
                alpha=THEME.physical.alpha_raw,
            ),
            medianprops=dict(
                color=THEME.physical.color_gt,
                linewidth=THEME.physical.line_width_median,
                alpha=THEME.physical.alpha_raw,
            ),
        )
        sns.stripplot(
            data=df,
            x="Model",
            y=y_col,
            hue="Model",
            ax=ax,
            palette=pal,
            jitter=THEME.physical.stripplot_jitter,
            size=THEME.physical.stripplot_size,
            alpha=THEME.physical.alpha_stripplot,
            legend=False,
            zorder=3,
        )
        ax.set_title(
            title,
            fontsize=THEME.physical.font_size_title,
            fontweight=THEME.physical.font_weight_title,
            pad=THEME.physical.title_pad,
        )
        ax.set_xlabel("Model", fontsize=THEME.physical.font_size_label)
        ax.set_ylabel(y_label, fontsize=THEME.physical.font_size_label)
        ax.tick_params(labelsize=THEME.physical.font_size_tick)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_facecolor(THEME.physical.bg_axis)
        _apply_grid(ax)

    # --- Helper: Correlation Plot (Scatter) ---
    def draw_scatter(ax, x_col, y_col, title, x_label, y_label):
        ax.set_facecolor(THEME.physical.bg_axis)
        if x_col in df.columns and y_col in df.columns:
            for mname in names:
                grp = df[df["Model"] == mname]
                ax.scatter(
                    grp[x_col],
                    grp[y_col],
                    label=str(mname),
                    color=pal.get(str(mname), "#ffffff"),
                    alpha=THEME.physical.alpha_scatter,
                    s=THEME.physical.marker_size,
                    edgecolor="w",
                    linewidth=0.5,
                    zorder=5,
                )
            all_vals = pd.concat([df[x_col], df[y_col]]).dropna()
            if not all_vals.empty:
                lo, hi = float(all_vals.min()), float(all_vals.max())
                pad = (hi - lo) * 0.1 + 1e-6
                ax.plot(
                    [lo - pad, hi + pad],
                    [lo - pad, hi + pad],
                    "w--",
                    linewidth=1.5,
                    label="ideal",
                    zorder=4,
                )
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing {x_col}/{y_col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel(x_label, fontsize=THEME.physical.font_size_label)
        ax.set_ylabel(y_label, fontsize=THEME.physical.font_size_label)
        ax.set_title(
            title,
            fontsize=THEME.physical.font_size_title,
            fontweight=THEME.physical.font_weight_title,
            pad=THEME.physical.title_pad,
        )
        ax.tick_params(labelsize=THEME.physical.font_size_tick)
        ax.legend(fontsize=THEME.physical.font_size_legend)
        _apply_grid(ax, axis="both")

    # --- Row 1: Geometry Focus ---
    draw_dist(axes[0, 0], "IoU", "Meltpool IoU Distribution", "IoU", ylim=(-0.1, 1.1))
    draw_dist(
        axes[0, 1],
        "Depth_Error",
        "Meltpool Depth Error Distribution",
        "Depth Error [vox]",
    )
    draw_scatter(
        axes[0, 2],
        "Depth_GT",
        "Depth_Pred",
        "Meltpool Depth: GT vs Pred",
        "GT Depth [vox]",
        "Pred Depth [vox]",
    )

    # --- Row 2: Thermal & Localization Focus ---
    draw_dist(axes[1, 0], "Offset_vox", "Hotspot Offset Distribution", "Offset [vox]")
    tmax_error_col = (
        "$T_{max}$ Error Distribution" if "$T_{max}$" in df.columns else "T_max_Error"
    )
    draw_dist(
        axes[1, 1],
        tmax_error_col,
        "$T_{max}$ Error [K] Distribution",
        "$T_{max}$ Error [K]",
    )
    draw_scatter(
        axes[1, 2],
        "T_max_GT",
        "T_max_Pred",
        "$T_{max}$: GT vs Pred",
        "GT $T_{max}$ [K]",
        "Pred $T_{max}$ [K]",
    )

    # Final layout polishing
    fig.tight_layout(pad=5.0)
    fig.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    fig.savefig(output_path, dpi=THEME.physical.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Physical fidelity dashboard (2x3) saved → %s", output_path)
