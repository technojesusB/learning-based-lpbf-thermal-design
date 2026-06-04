"""TV / PBD structural metric comparison chart."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _palette(names: list[str], color_map: dict[str, str] | None = None) -> dict[str, str]:
    """Dynamically assign colors, prioritizing the provided color_map."""
    if color_map:
        return {n: color_map[n] for n in names if n in color_map}

    out: dict[str, str] = {}
    # Use spectral palette as structural metrics are related to spectral fidelity
    palette = THEME.spectral.color_palette
    for i, n in enumerate(names):
        out[str(n)] = palette[i % len(palette)]
    return out


def plot_tv_pbd_comparison(
    df: pd.DataFrame,
    output_path: Path | str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Write a 1×2 bar chart comparing Total Variation and Patch Border Discontinuity.

    Args:
        df:           DataFrame with columns Model, TV, PBD.
        output_path:  Destination PNG path.
        color_map:    Optional canonical color mapping for models.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.style.context("dark_background"):
        # Use spectral settings as fallback for structural plots
        # Wider figsize (16x7) for straight labels and better spacing
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=THEME.spectral.dpi)
        fig.patch.set_facecolor(THEME.spectral.bg_figure)

        for ax in axes:
            ax.set_facecolor(THEME.spectral.bg_axis)
            ax.tick_params(labelsize=THEME.spectral.font_size_tick)

        if not df.empty:
            models = sorted(df["Model"].unique().tolist())
            pal = _palette(models, color_map=color_map)
            colors = [pal.get(m, "#ffffff") for m in df["Model"]]

            # [0] TV Bar
            axes[0].bar(df["Model"], df["TV"], color=colors, alpha=THEME.system.alpha_bar)
            axes[0].set_title("Total Variation (TV)", fontsize=THEME.spectral.font_size_title)
            axes[0].set_ylabel("TV", fontsize=THEME.spectral.font_size_label)
            axes[0].tick_params(axis="x", rotation=0)

            # [1] PBD Bar
            axes[1].bar(df["Model"], df["PBD"], color=colors, alpha=THEME.system.alpha_bar)
            axes[1].set_title(
                "Patch Border Discontinuity (PBD)",
                fontsize=THEME.spectral.font_size_title,
            )
            axes[1].set_ylabel(
                "PBD (boundary / interior ratio)",
                fontsize=THEME.spectral.font_size_label,
            )
            axes[1].axhline(
                y=1.0,
                color=THEME.spectral.color_gt,
                linestyle="--",
                linewidth=THEME.spectral.line_width_sub,
                alpha=0.6,
                label="ideal=1",
            )
            axes[1].legend(fontsize=THEME.spectral.font_size_legend)
            axes[1].tick_params(axis="x", rotation=0)
        else:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        fig.suptitle(
            "Structural Metrics: TV & PBD",
            fontsize=THEME.spectral.font_size_title + 2,
            fontweight=THEME.spectral.font_weight_title,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=THEME.spectral.dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("TV/PBD comparison saved → %s", output_path)
