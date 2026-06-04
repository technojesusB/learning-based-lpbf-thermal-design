"""System dashboard: training vs inference performance (2×2 layout)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _get_total_vram_mb() -> float:
    """Get total GPU memory in MB using pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = info.total / (1024 * 1024)
        pynvml.nvmlShutdown()
        return float(total_mb)
    except Exception as exc:
        logger.warning("Could not detect total VRAM via pynvml: %s. Using 16GB fallback.", exc)
        return 16384.0


def _palette(names: list[str], color_map: dict[str, str] | None = None) -> dict[str, str]:
    """Dynamically assign colors, prioritizing the provided color_map."""
    if color_map:
        return {n: color_map[n] for n in names if n in color_map}

    out: dict[str, str] = {}
    palette = THEME.system.color_palette
    for i, n in enumerate(names):
        out[str(n)] = palette[i % len(palette)]
    return out


def _apply_gpu_axes(ax, total_vram_mb, title):
    """Standardized GPU scatter axes with hardware-based breathing room."""
    # X-Axis: 0 to 105%, but ticks only every 10% up to 100%
    ax.set_xlim(0, 105)
    ax.set_xticks(np.arange(0, 101, 10))

    # Y-Axis: Hardware capacity + 1000MB buffer
    ax.set_ylim(0, total_vram_mb + 1000)

    ax.set_xlabel("GPU Util [%]", fontsize=THEME.system.font_size_label)
    ax.set_ylabel("GPU Mem [MB]", fontsize=THEME.system.font_size_label)
    ax.set_title(title, fontsize=THEME.system.font_size_title)


def plot_system_dashboard(
    train_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    output_path: Path | str,
    color_map: dict[str, str] | None = None,
) -> None:
    """Write a 2×2 system benchmark dashboard PNG.

    Layout:
        [0,0] Training Duration [h] + s/sample (line)
        [0,1] Training GPU Util [%] vs GPU Mem [MB]
        [1,0] Inference Throughput s/sample
        [1,1] Inference GPU Util [%] vs GPU Mem [MB]

    Args:
        train_df:      DataFrame with columns Version, Duration [h], s/sample,
                       GPU Util [%], GPU Mem [MB], Inf GPU Util [%], Inf GPU Mem [MB].
        inference_df:  DataFrame with columns Model, s_per_sample.
        output_path:   Destination PNG path.
        color_map:     Optional canonical color mapping for models.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_vram = _get_total_vram_mb()

    with plt.style.context("dark_background"):
        # Use standard HERO 2x2 figsize
        fig, axes = plt.subplots(2, 2, figsize=(16.7, 15), dpi=THEME.system.dpi)
        fig.patch.set_facecolor(THEME.system.bg_figure)

        for ax in axes.flatten():
            ax.set_facecolor(THEME.system.bg_axis)
            ax.tick_params(labelsize=THEME.system.font_size_tick)

        if not train_df.empty:
            versions = sorted(train_df["Version"].unique().tolist())
            pal = _palette(versions, color_map=color_map)
            colors = [pal.get(v, "#ffffff") for v in train_df["Version"]]

            # [0, 0] Training Duration + Efficiency
            ax00 = axes[0, 0]
            ax00.bar(
                train_df["Version"],
                train_df["Duration [h]"].fillna(0),
                color=colors,
                alpha=THEME.system.alpha_bar,
            )
            ax00.set_title("Training Duration & Efficiency", fontsize=THEME.system.font_size_title)
            ax00.set_ylabel("Duration [h]", fontsize=THEME.system.font_size_label)
            ax00.tick_params(axis="x", rotation=0)

            ax00_twin = ax00.twinx()
            valid_eff = train_df["s/sample"].dropna()
            if not valid_eff.empty:
                ax00_twin.plot(
                    train_df["Version"],
                    train_df["s/sample"].fillna(0),
                    color=THEME.system.color_secondary,
                    marker="o",
                    linewidth=THEME.system.line_width_sub,
                    label="s/sample",
                )
                ax00_twin.set_ylabel(
                    "s/sample",
                    color=THEME.system.color_secondary,
                    fontsize=THEME.system.font_size_label,
                )
                ax00_twin.tick_params(
                    axis="y",
                    colors=THEME.system.color_secondary,
                    labelsize=THEME.system.font_size_tick,
                )

            # [0, 1] Training GPU Scatter (with Legends)
            ax01 = axes[0, 1]
            for _, row in train_df.iterrows():
                util = row["GPU Util [%]"]
                mem = row["GPU Mem [MB]"]
                ver = str(row["Version"])
                if not pd.isna(util) and not pd.isna(mem):
                    ax01.scatter(
                        util,
                        mem,
                        color=pal.get(ver, "#ffffff"),
                        s=THEME.system.marker_size_scatter,
                        zorder=5,
                        alpha=THEME.system.alpha_scatter,
                        label=ver,
                    )
            ax01.legend(fontsize=THEME.system.font_size_legend, loc="upper left", framealpha=0.3)
            _apply_gpu_axes(ax01, total_vram, "Training Resource Consumption")

            # [1, 1] Inference GPU Scatter (with Legends)
            ax11 = axes[1, 1]
            has_inf_data = False
            for _, row in train_df.iterrows():
                util = row["Inf GPU Util [%]"]
                mem = row["Inf GPU Mem [MB]"]
                ver = str(row["Version"])
                if not pd.isna(util) and not pd.isna(mem):
                    has_inf_data = True
                    ax11.scatter(
                        util,
                        mem,
                        color=pal.get(ver, "#ffffff"),
                        s=THEME.system.marker_size_scatter,
                        zorder=5,
                        alpha=THEME.system.alpha_scatter,
                        label=ver,
                    )
            if not has_inf_data:
                ax11.text(
                    0.5,
                    0.5,
                    "No inference data found",
                    ha="center",
                    va="center",
                    transform=ax11.transAxes,
                    color="gray",
                )
            else:
                ax11.legend(
                    fontsize=THEME.system.font_size_legend,
                    loc="upper left",
                    framealpha=0.3,
                )
            _apply_gpu_axes(ax11, total_vram, "Inference Resource Consumption")

        # [1, 0] Inference Throughput
        ax10 = axes[1, 0]
        if not inference_df.empty:
            inf_models = sorted(inference_df["Model"].unique().tolist())
            inf_pal = _palette(inf_models, color_map=color_map)
            inf_colors = [inf_pal.get(m, "#ffffff") for m in inference_df["Model"]]

            ax10.bar(
                inference_df["Model"],
                inference_df["s_per_sample"],
                color=inf_colors,
                alpha=THEME.system.alpha_bar,
            )
            ax10.set_title("Inference Throughput", fontsize=THEME.system.font_size_title)
            ax10.set_ylabel("s/sample", fontsize=THEME.system.font_size_label)
            ax10.tick_params(axis="x", rotation=0)
        else:
            ax10.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax10.transAxes)

        plt.tight_layout()
        fig.savefig(output_path, dpi=THEME.system.dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("System dashboard saved → %s", output_path)
