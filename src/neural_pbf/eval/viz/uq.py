"""UQ figures: coverage, ensemble spread, sensitivity, and the leakage demonstration.

Every function takes tables the pipeline already saved. Nothing here loads a model, uses the GPU,
or computes a test metric, so figures can be re-rendered at any time without touching the test guard.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

COVERAGE_FILE = "coverage_rows.csv"
POINTS_FILE = "sample_points.csv"
SENSITIVITY_FILE = "sensitivity_rows.csv"
ERRORS_FILE = "point_errors.csv"

NOMINAL_FILE = "coverage_nominal_vs_empirical.png"
SPREAD_FILE = "spread_vs_error.png"
MATERIAL_FILE = "coverage_per_material.png"
RUN_SPREAD_FILE = "coverage_per_run.png"
SENSITIVITY_FIG_FILE = "sensitivity_vs_k.png"
LEAKAGE_FILE = "leakage_demo.png"
MODEL_FIGURES = ("nominal", "spread", "material")

_QUANTITY_LABELS = {
    "depth_vox": "Meltpool Depth [vox]",
    "T_max_K": "$T_{max}$ [K]",
    "volume_vox": "Meltpool Volume [vox]",
}
_METHOD_LABELS = {
    "absolute": "Absolute",
    "normalized": "Normalized",
    "constant_single_draw": "Constant (single draw)",
    "raw_ensemble": "Raw ensemble",
}
_COVERAGE_COLUMNS = {"quantity", "alpha", "method", "stratum_type", "stratum", "coverage"}


def model_figure_name(model: str, kind: str) -> str:
    """File name of a per-model figure; the report links to the same name."""
    slug = "".join(c if c.isalnum() else "_" for c in model).lower()
    return f"{slug}_{kind}.png"


def _require(df: pd.DataFrame, columns: Iterable[str], figure: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{figure}: missing columns {missing}")


def _select(df: pd.DataFrame, figure: str, **equals: Any) -> Any:
    keep = np.ones(len(df), dtype=bool)
    for column, value in equals.items():
        keep &= (df[column] == value).to_numpy()
    if not keep.any():
        raise ValueError(f"{figure}: no rows for {equals}")
    return df[keep]


def _apply_grid(ax: Any, axis: str | None = None) -> None:
    g = THEME.uq.grid
    ax.grid(g.enabled, which=g.which, axis=axis or g.axis, linestyle=g.linestyle, alpha=g.alpha, color=g.color)


def _decorate(ax: Any, title: str, xlabel: str, ylabel: str, grid_axis: str | None = None) -> None:
    t = THEME.uq
    ax.set_facecolor(t.bg_axis)
    ax.set_title(title, fontsize=t.font_size_title, fontweight=t.font_weight_title, pad=t.title_pad)
    ax.set_xlabel(xlabel, fontsize=t.font_size_label)
    ax.set_ylabel(ylabel, fontsize=t.font_size_label)
    ax.tick_params(labelsize=t.font_size_tick)
    _apply_grid(ax, grid_axis)


def _ideal_h(ax: Any, value: float, label: str | None = None) -> None:
    t = THEME.uq
    ax.axhline(value, color=t.color_ideal, linestyle=t.linestyle_ideal, linewidth=t.line_width_ideal, label=label)


def _new_figure(rows: int, cols: int, figsize: tuple[int, int], **kwargs: Any) -> tuple[Any, Any]:
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=THEME.uq.dpi, squeeze=False, **kwargs)
    fig.patch.set_facecolor(THEME.uq.bg_figure)
    return fig, axes


def _figure_legend(fig: Any, handles: list[Any], ncol: int) -> None:
    t = THEME.uq
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        fontsize=t.font_size_legend,
        framealpha=t.legend_framealpha,
        bbox_to_anchor=(0.5, 0.0),
    )


def _suptitle(fig: Any, text: str) -> None:
    fig.suptitle(text, fontsize=THEME.uq.font_size_title, fontweight=THEME.uq.font_weight_title)


def _model_name(df: pd.DataFrame, figure: str) -> str:
    models = list(dict.fromkeys(df["model"]))
    if len(models) != 1:
        raise ValueError(f"{figure}: expected rows of one model, got {models}")
    return str(models[0])


def _selection_title(method: str, alpha: float) -> str:
    return f"{_METHOD_LABELS[method]}, α = {alpha}"


def _finish(fig: Any, output_path: Path | str, tracker: Any, *, bottom: float = 0.0) -> None:
    """Save the PNG atomically or, given a tracker, log a temp copy and leave the saved PNG alone.

    The temp copy is needed because ``log_figure`` deletes the file it saves.
    """
    output_path = Path(output_path)
    temp = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        fig.tight_layout(rect=(0, bottom, 1, 1))
        if tracker is not None:
            with tempfile.TemporaryDirectory() as tmp:
                log_figure(tracker, fig, str(Path(tmp) / output_path.name), dpi=THEME.uq.dpi, artifact_path="figures")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(temp, dpi=THEME.uq.dpi, bbox_inches="tight", facecolor=THEME.uq.bg_figure)
        os.replace(temp, output_path)
        logger.info("UQ figure saved → %s", output_path)
    finally:
        plt.close(fig)
        temp.unlink(missing_ok=True)


def plot_coverage_vs_nominal(df: pd.DataFrame, output_path: Path | str, tracker: Any = None) -> None:
    """Write a 1×3 chart of empirical against nominal coverage, one line per method.

    Args:
        df:           Coverage rows of one model. Hollow markers show alphas with no run-level support.
        output_path:  Destination PNG path.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_coverage_vs_nominal"
    _require(df, _COVERAGE_COLUMNS | {"model", "run_level_supported"}, figure)
    rows = _select(df, figure, stratum_type="all")
    t = THEME.uq
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(1, len(_QUANTITY_LABELS), t.figsize, sharey=True)
        nominal = 1 - rows["alpha"]
        limits = [float(nominal.min()) - 0.1, 1.0]
        for ax, (quantity, label) in zip(axes[0], _QUANTITY_LABELS.items(), strict=True):
            ax.plot(limits, limits, color=t.color_ideal, linestyle=t.linestyle_ideal, linewidth=t.line_width_ideal)
            for method, color in t.method_colors.items():
                sub = rows[(rows["quantity"] == quantity) & (rows["method"] == method)]
                sub = sub.sort_values("alpha", ascending=False)
                ax.plot(1 - sub["alpha"], sub["coverage"], color=color, linewidth=t.line_width_main, zorder=3)
                for supported, face in ((True, color), (False, "none")):
                    part = sub[sub["run_level_supported"] == supported]
                    ax.plot(
                        1 - part["alpha"],
                        part["coverage"],
                        linestyle="none",
                        marker=t.method_markers[method],
                        markersize=t.marker_size,
                        markerfacecolor=face,
                        markeredgecolor=color,
                        zorder=4,
                    )
            _decorate(ax, label, "Nominal coverage", "Empirical coverage", grid_axis="both")
        handles = [
            Line2D([], [], color=c, marker=t.method_markers[m], markersize=t.marker_size, label=_METHOD_LABELS[m])
            for m, c in t.method_colors.items()
        ]
        handles += [
            Line2D([], [], color=t.color_ideal, linestyle=t.linestyle_ideal, label="ideal"),
            Line2D([], [], color=t.color_ideal, linestyle="none", marker=t.marker_default, markerfacecolor="none",
                   markersize=t.marker_size, label="no run-level support"),
        ]  # fmt: skip
        _suptitle(fig, _model_name(rows, figure))
        _figure_legend(fig, handles, ncol=len(handles))
        _finish(fig, output_path, tracker, bottom=0.1)


def plot_spread_vs_error(df: pd.DataFrame, output_path: Path | str, tracker: Any = None) -> None:
    """Write a 1×3 scatter of ensemble spread against the absolute error of the ensemble mean.

    Args:
        df:           Rows of one model with columns quantity, spread, abs_error.
        output_path:  Destination PNG path.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_spread_vs_error"
    _require(df, {"model", "quantity", "spread", "abs_error"}, figure)
    t = THEME.uq
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(1, len(_QUANTITY_LABELS), t.figsize)
        for ax, (quantity, label) in zip(axes[0], _QUANTITY_LABELS.items(), strict=True):
            sub = _select(df, figure, quantity=quantity)
            ax.scatter(
                sub["spread"],
                sub["abs_error"],
                s=t.marker_size_scatter,
                color=t.color_points,
                alpha=t.alpha_scatter,
                edgecolor=t.color_marker_edge,
                linewidth=t.marker_edge_width,
                zorder=5,
            )
            _decorate(ax, label, "Ensemble spread (σ)", "Absolute error of ensemble mean", grid_axis="both")
        _suptitle(fig, _model_name(df, figure))
        _finish(fig, output_path, tracker)


def plot_coverage_per_material(
    df: pd.DataFrame, output_path: Path | str, method: str, alpha: float, tracker: Any = None
) -> None:
    """Write a 1×3 chart of coverage per material with run-level bootstrap 95% intervals.

    Args:
        df:           Coverage rows of one model.
        output_path:  Destination PNG path.
        method:       Conformal method to show.
        alpha:        Miscoverage level to show; the dashed line is ``1 - alpha``.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_coverage_per_material"
    _require(df, _COVERAGE_COLUMNS | {"model", "boot_lo", "boot_hi"}, figure)
    rows = _select(df, figure, stratum_type="material", method=method, alpha=alpha)
    t = THEME.uq
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(1, len(_QUANTITY_LABELS), t.figsize, sharey=True)
        for ax, (quantity, label) in zip(axes[0], _QUANTITY_LABELS.items(), strict=True):
            sub = rows[rows["quantity"] == quantity]
            x = np.arange(len(sub))
            # The point estimate can sit outside the bootstrap interval, so clamp the error bars at zero.
            lower = np.maximum(sub["coverage"] - sub["boot_lo"], 0)
            upper = np.maximum(sub["boot_hi"] - sub["coverage"], 0)
            ax.errorbar(
                x,
                sub["coverage"],
                yerr=[lower, upper],
                fmt=t.method_markers[method],
                markersize=t.marker_size,
                color=t.method_colors[method],
                capsize=t.capsize,
                linewidth=t.line_width_sub,
                zorder=4,
            )
            _ideal_h(ax, 1 - alpha)
            ax.set_xticks(x, sub["stratum"])
            ax.set_ylim(*t.coverage_ylim)
            ax.set_xlim(-0.5, max(len(sub) - 0.5, 0.5))
            _decorate(ax, label, "Material", f"Coverage (dashed: {1 - alpha:.2f})")
        _suptitle(fig, f"{_model_name(rows, figure)}: {_selection_title(method, alpha)}")
        _finish(fig, output_path, tracker)


def plot_run_coverage_spread(
    df: pd.DataFrame, output_path: Path | str, method: str, alpha: float, tracker: Any = None
) -> None:
    """Write a box and strip chart of per-run coverage, each test run counted once.

    Args:
        df:           Coverage rows of one model.
        output_path:  Destination PNG path.
        method:       Conformal method to show.
        alpha:        Miscoverage level to show; the dashed line is ``1 - alpha``.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_run_coverage_spread"
    _require(df, _COVERAGE_COLUMNS | {"model"}, figure)
    rows = _select(df, figure, stratum_type="run", method=method, alpha=alpha)
    rows = rows.assign(Quantity=rows["quantity"].map(_QUANTITY_LABELS))
    t = THEME.uq
    color = t.method_colors[method]
    order = list(_QUANTITY_LABELS.values())
    line = dict(color=t.color_gt, linewidth=t.line_width_sub, alpha=t.alpha_raw)
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(1, 1, t.figsize_default)
        ax = axes[0, 0]
        sns.boxplot(
            data=rows, x="Quantity", y="coverage", order=order, ax=ax, color=color, whis=100, fliersize=0, zorder=2,
            boxprops=dict(alpha=t.alpha_raw), whiskerprops=line, capprops=line, medianprops=line,
        )  # fmt: skip
        sns.stripplot(
            data=rows, x="Quantity", y="coverage", order=order, ax=ax, color=color, jitter=t.stripplot_jitter,
            size=t.stripplot_size, alpha=t.alpha_stripplot, zorder=3,
        )  # fmt: skip
        _ideal_h(ax, 1 - alpha, label=f"nominal {1 - alpha:.2f}")
        ax.set_ylim(*t.coverage_ylim)
        _decorate(ax, "Per-Run Coverage", "Quantity", "Coverage per run")
        _suptitle(fig, f"{_model_name(rows, figure)}: {_selection_title(method, alpha)}")
        ax.legend(fontsize=t.font_size_legend, framealpha=t.legend_framealpha, loc="lower right")
        _finish(fig, output_path, tracker)


def plot_sensitivity_vs_k(df: pd.DataFrame, output_path: Path | str, alpha: float, tracker: Any = None) -> None:
    """Write a 2×3 chart of coverage (top) and mean width (bottom) against ensemble size K.

    Args:
        df:           Sensitivity rows with columns K, quantity, alpha, method, coverage, mean_width.
        output_path:  Destination PNG path.
        alpha:        Miscoverage level to show; the dashed line is ``1 - alpha``.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_sensitivity_vs_k"
    _require(df, {"K", "quantity", "alpha", "method", "coverage", "mean_width"}, figure)
    rows = _select(df, figure, alpha=alpha).sort_values("K")
    t = THEME.uq
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(2, len(_QUANTITY_LABELS), t.figsize_grid, sharex=True)
        for col, (quantity, label) in enumerate(_QUANTITY_LABELS.items()):
            sub = rows[rows["quantity"] == quantity]
            for method, color in t.method_colors.items():
                part = sub[sub["method"] == method]
                style = dict(color=color, marker=t.method_markers[method], markersize=t.marker_size,
                             linewidth=t.line_width_main)  # fmt: skip
                axes[0, col].plot(part["K"], part["coverage"], **style)
                # An infinite quantile has no finite width to draw.
                finite = part[np.isfinite(part["mean_width"])]
                axes[1, col].plot(finite["K"], finite["mean_width"], **style)
            _ideal_h(axes[0, col], 1 - alpha)
            axes[0, col].set_ylim(*t.coverage_ylim)
            axes[1, col].set_xticks(sorted(rows["K"].unique()))
            _decorate(axes[0, col], label, "", f"Coverage (dashed: {1 - alpha:.2f})", grid_axis="both")
            _decorate(axes[1, col], f"{label}: Mean Width", "Ensemble size K", "Mean interval width", grid_axis="both")
        handles = [
            Line2D([], [], color=c, marker=t.method_markers[m], markersize=t.marker_size, label=_METHOD_LABELS[m])
            for m, c in t.method_colors.items()
        ]
        _suptitle(fig, f"Sensitivity to Ensemble Size, Primary Model: α = {alpha}")
        _figure_legend(fig, handles, ncol=len(handles))
        _finish(fig, output_path, tracker, bottom=0.05)


def _draw_leakage_quantity(
    axes: Any,
    quantity: str,
    label: str,
    rows: pd.DataFrame,
    errors: pd.DataFrame,
    colors: dict[str, str],
    alpha: float,
) -> None:
    """Draw one quantity's coverage (``axes[0]``) and MAE (``axes[1]``) for every model."""
    figure = "plot_leakage_demo"
    t = THEME.uq
    for x, (model, color) in enumerate(colors.items()):
        row = _select(rows, figure, model=model, quantity=quantity).iloc[0]
        axes[0].errorbar(
            x,
            row["coverage"],
            yerr=[[max(row["coverage"] - row["boot_lo"], 0)], [max(row["boot_hi"] - row["coverage"], 0)]],
            fmt=t.marker_default,
            markersize=t.marker_size,
            color=color,
            capsize=t.capsize,
            linewidth=t.line_width_sub,
            zorder=4,
        )
        mae = _select(errors, figure, model=model, quantity=quantity).iloc[0]["mae"]
        axes[1].bar(x, mae, color=color, alpha=t.alpha_bar)
    _ideal_h(axes[0], 1 - alpha)
    axes[0].set_ylim(*t.coverage_ylim)
    _decorate(axes[0], f"{label}: Coverage", "", f"Coverage (dashed: {1 - alpha:.2f})")
    _decorate(axes[1], f"{label}: MAE", "", "MAE of ensemble mean")
    for ax in axes:
        ax.set_xlim(-0.6, len(colors) - 0.4)
        ax.set_xticks(range(len(colors)), [str(i + 1) for i in range(len(colors))])


def plot_leakage_demo(
    coverage: pd.DataFrame,
    errors: pd.DataFrame,
    output_path: Path | str,
    method: str,
    alpha: float,
    tracker: Any = None,
) -> None:
    """Write a 2×3 chart of coverage (top) and ensemble-mean MAE (bottom) per model and leakage label.

    Models keep the order of the saved rows and carry their run-disjoint label, so this does not rank them.

    Args:
        coverage:     Coverage rows of all models, with columns model and label.
        errors:       Point errors of all models with columns model, quantity, mae.
        output_path:  Destination PNG path.
        method:       Conformal method to show.
        alpha:        Miscoverage level to show; the dashed line is ``1 - alpha``.
        tracker:      Optional object with ``log_artifact``; if given, the figure is logged instead of saved.
    """
    figure = "plot_leakage_demo"
    _require(coverage, _COVERAGE_COLUMNS | {"model", "label", "boot_lo", "boot_hi"}, figure)
    _require(errors, {"model", "quantity", "mae"}, figure)
    rows = _select(coverage, figure, stratum_type="all", method=method, alpha=alpha)
    t = THEME.uq
    models = list(dict.fromkeys(rows["model"]))
    colors = {m: t.color_palette[i % len(t.color_palette)] for i, m in enumerate(models)}
    labels = dict(zip(rows["model"], rows["label"], strict=False))
    with plt.style.context("dark_background"):
        fig, axes = _new_figure(2, len(_QUANTITY_LABELS), t.figsize_grid)
        for col, (quantity, label) in enumerate(_QUANTITY_LABELS.items()):
            _draw_leakage_quantity(axes[:, col], quantity, label, rows, errors, colors, alpha)
        handles = [
            Line2D([], [], color=colors[m], marker=t.marker_default, linestyle="none", markersize=t.marker_size,
                   label=f"{i + 1}: {m} [{labels[m]}]")
            for i, m in enumerate(models)
        ]  # fmt: skip
        _suptitle(fig, f"Leakage Demonstration: {_selection_title(method, alpha)}")
        _figure_legend(fig, handles, ncol=1)
        _finish(fig, output_path, tracker, bottom=0.04 + 0.02 * len(models))


def _read(directory: Path, name: str) -> Any:
    return pd.read_csv(directory / name, dtype={"stratum": str, "run_id": str})


def _figure_jobs(data_dir: Path, method: str, alpha: float, tracker: Any) -> list[tuple[str, Callable[[Path], None]]]:
    """Load the saved tables and list every figure as ``(file name, render function)``."""
    coverage = _read(data_dir, COVERAGE_FILE)
    points = _read(data_dir, POINTS_FILE)
    sensitivity = _read(data_dir, SENSITIVITY_FILE)
    errors = _read(data_dir, ERRORS_FILE)
    _require(coverage, {"model", "role"}, "render_uq_figures")
    primaries = list(dict.fromkeys(coverage.loc[coverage["role"] == "primary", "model"]))
    if len(primaries) != 1:
        raise ValueError(f"{COVERAGE_FILE}: expected one primary model, got {primaries}")
    primary = primaries[0]

    def model_jobs(model: str, nominal: str, spread: str, material: str) -> list[tuple[str, Callable[[Path], None]]]:
        rows = coverage[coverage["model"] == model]
        sample = points[points["model"] == model]
        return [
            (nominal, lambda p: plot_coverage_vs_nominal(rows, p, tracker)),
            (spread, lambda p: plot_spread_vs_error(sample, p, tracker)),
            (material, lambda p: plot_coverage_per_material(rows, p, method, alpha, tracker)),
        ]

    primary_rows = coverage[coverage["model"] == primary]
    jobs = model_jobs(primary, NOMINAL_FILE, SPREAD_FILE, MATERIAL_FILE)
    for model in dict.fromkeys(coverage["model"]):
        jobs += model_jobs(model, *(model_figure_name(model, kind) for kind in MODEL_FIGURES))
    return [
        *jobs,
        (RUN_SPREAD_FILE, lambda p: plot_run_coverage_spread(primary_rows, p, method, alpha, tracker)),
        (SENSITIVITY_FIG_FILE, lambda p: plot_sensitivity_vs_k(sensitivity, p, alpha, tracker)),
        (LEAKAGE_FILE, lambda p: plot_leakage_demo(coverage, errors, p, method, alpha, tracker)),
    ]


def render_uq_figures(
    data_dir: Path | str,
    fig_dir: Path | str,
    primary_method: str,
    primary_alpha: float,
    tracker: Any = None,
) -> dict[str, str]:
    """Render every UQ figure from saved tables, or log them all to ``tracker`` without touching saved PNGs.

    A failed figure never stops the others, and when saving, its old PNG is removed so the report cannot show
    outdated content. Returns ``{file name: error}`` for figures that failed.
    """
    fig_dir = Path(fig_dir)
    try:
        jobs = _figure_jobs(Path(data_dir), primary_method, primary_alpha, tracker)
    except Exception as exc:  # A bad table must not end the stage; the caller reports this.
        logger.exception("Could not read UQ figure data from %s", data_dir)
        return {"all figures": f"{type(exc).__name__}: {exc}"}

    failures: dict[str, str] = {}
    for name, job in jobs:
        try:
            if tracker is None:
                (fig_dir / name).unlink(missing_ok=True)
            job(fig_dir / name)
        except Exception as exc:  # One broken figure must not cost the others or the stage.
            logger.exception("UQ figure %s failed", name)
            failures[name] = f"{type(exc).__name__}: {exc}"
            plt.close("all")
    return failures
