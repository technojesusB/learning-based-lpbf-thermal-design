"""Loss analytics plots: training/validation curves with optional lambda_phys panel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

from neural_pbf.schemas.viz import THEME

if matplotlib.get_backend().lower() in {"tkagg", "qt5agg", "qt4agg", "wxagg", "macosx"}:
    import os as _os

    if not _os.environ.get("DISPLAY") and not _os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")


def _moving_avg(values: Sequence[float], window: int) -> list[float]:
    """Compute centered moving average, padding edges by replication."""
    n = len(values)
    result: list[float] = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def _apply_dark_theme(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> None:
    fig.patch.set_facecolor(THEME.base.bg_figure)
    ax.set_facecolor(THEME.base.bg_axis)
    if THEME.base.grid.enabled:
        ax.grid(
            True,
            which=THEME.base.grid.which,
            axis=THEME.base.grid.axis,
            linestyle=THEME.base.grid.linestyle,
            alpha=THEME.base.grid.alpha,
            color=THEME.base.grid.color,
        )


def loss_panel(
    train_losses: list[float],
    val_losses: list[float],
    *,
    fm_losses: list[float] | None = None,
    pde_losses: list[float] | None = None,
    lambda_hist: list[float] | None = None,
    title: str = "",
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Multi-panel loss analytics plot.

    Two rendering modes determined by the presence of sub-loss arguments:

    **Standard mode** (v1/v2/v3, Section 2 of plot_specs.md):
        Called when ``fm_losses`` and ``pde_losses`` are both ``None``.
        One panel (+ optional lambda panel), log-scale, MA-5 overlay,
        figsize ``(10, 6)``, DPI 120.

    **Detailed mode** (v4/v5, Section 3 of plot_specs.md):
        Called when both ``fm_losses`` *and* ``pde_losses`` are provided.
        Always two panels: sub-loss breakdown (top) + lambda_phys (bottom),
        figsize ``(14, 12)``, DPI 200.  ``lambda_hist`` is required in this mode.

    Args:
        train_losses:  Per-epoch total training loss.
        val_losses:    Per-epoch validation loss (must match ``train_losses`` length).
        fm_losses:     Per-epoch FM data-loss component (detailed mode only).
        pde_losses:    Per-epoch PDE residual component (detailed mode only).
        lambda_hist:   Per-epoch lambda_phys values.
        title:         Optional figure suptitle.
        savepath:      If provided, the figure is saved to this path.

    Returns:
        A :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: On length mismatches, empty inputs, or partial sub-loss args.
    """
    if len(train_losses) != len(val_losses):
        raise ValueError(
            "train_losses and val_losses must have the same length; "
            f"got {len(train_losses)} and {len(val_losses)}."
        )
    if len(train_losses) == 0:
        raise ValueError("train_losses and val_losses must not be empty.")
    if lambda_hist is not None and len(lambda_hist) != len(train_losses):
        raise ValueError(
            "lambda_hist must have the same length as train_losses; "
            f"got {len(lambda_hist)} vs {len(train_losses)}."
        )

    # Validate sub-loss args: must be both or neither
    if (fm_losses is None) != (pde_losses is None):
        raise ValueError(
            "fm_losses and pde_losses must both be provided or both be None."
        )
    if fm_losses is not None and len(fm_losses) != len(train_losses):
        raise ValueError(
            "fm_losses must have the same length as train_losses; "
            f"got {len(fm_losses)} vs {len(train_losses)}."
        )
    if pde_losses is not None and len(pde_losses) != len(train_losses):
        raise ValueError(
            "pde_losses must have the same length as train_losses; "
            f"got {len(pde_losses)} vs {len(train_losses)}."
        )

    detailed = fm_losses is not None  # both fm_losses and pde_losses are set

    if detailed:
        return _loss_panel_detailed(
            train_losses,
            val_losses,
            fm_losses,
            pde_losses,  # type: ignore[arg-type]
            lambda_hist=lambda_hist,
            title=title,
            savepath=savepath,
        )
    return _loss_panel_standard(
        train_losses,
        val_losses,
        lambda_hist=lambda_hist,
        title=title,
        savepath=savepath,
    )


# ── rendering helpers ──────────────────────────────────────────────────────────


def _loss_panel_standard(
    train_losses: list[float],
    val_losses: list[float],
    *,
    lambda_hist: list[float] | None,
    title: str,
    savepath: Path | str | None,
) -> matplotlib.figure.Figure:
    """Section 2 standard style — one panel + optional lambda panel."""
    epochs = list(range(len(train_losses)))
    train_ma5 = _moving_avg(train_losses, 5)
    val_ma5 = _moving_avg(val_losses, 5)

    with plt.style.context("dark_background"):
        if lambda_hist is not None:
            fig, (ax_loss, ax_lam) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(10, 6))

        _apply_dark_theme(fig, ax_loss)

        # Raw lines (background)
        ax_loss.plot(
            epochs,
            train_losses,
            color="#2e7d32",
            alpha=THEME.base.alpha_raw,
            linewidth=1,
            label="Train (raw)",
        )
        ax_loss.plot(
            epochs,
            val_losses,
            color="#c62828",
            alpha=THEME.base.alpha_raw,
            linewidth=1,
            label="Val (raw)",
        )
        # MA-5 (foreground)
        ax_loss.plot(
            epochs,
            train_ma5,
            color=THEME.base.color_train,
            linewidth=THEME.base.line_width_main,
            label="Train (MA-5)",
        )
        ax_loss.plot(
            epochs,
            val_ma5,
            color=THEME.base.color_val,
            linewidth=THEME.base.line_width_main,
            label="Val (MA-5)",
        )

        ax_loss.set_yscale("log")
        ax_loss.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=10)
        )
        ax_loss.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(
                base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12
            )
        )
        ax_loss.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_loss.grid(True, which="both", linestyle="--", alpha=0.4, color="#666666")

        ax_loss.set_xlabel("Epoch", fontsize=10)
        ax_loss.set_ylabel("Loss (log)", fontsize=11, fontweight="bold")
        ax_loss.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax_loss.legend(
            facecolor=THEME.base.bg_axis,
            edgecolor=THEME.base.grid.color,
            loc="upper right",
        )
        if title:
            ax_loss.set_title(title)

        if lambda_hist is not None:
            _apply_dark_theme(fig, ax_lam)
            ax_lam.plot(
                epochs,
                lambda_hist,
                color="#9b59b6",
                linewidth=2,
                label="Physics Weight lambda_phys",
            )
            ax_lam.fill_between(epochs, 0, lambda_hist, color="#9b59b6", alpha=0.2)
            ax_lam.set_yscale("log")
            ax_lam.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10.0, numticks=10)
            )
            ax_lam.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if THEME.base.grid.enabled:
                ax_lam.grid(
                    True,
                    which=THEME.base.grid.which,
                    axis=THEME.base.grid.axis,
                    linestyle=THEME.base.grid.linestyle,
                    alpha=THEME.base.grid.alpha,
                    color=THEME.base.grid.color,
                )
            ax_lam.set_xlabel("Epoch", fontsize=10)
            ax_lam.set_ylabel(
                r"ReLoBRaLo $\lambda_{phys}$ (log)", fontsize=11, fontweight="bold"
            )
            ax_lam.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))

        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=120, bbox_inches="tight")

    return fig


def _loss_panel_detailed(
    train_losses: list[float],
    val_losses: list[float],
    fm_losses: list[float],
    pde_losses: list[float],
    *,
    lambda_hist: list[float] | None,
    title: str,
    savepath: Path | str | None,
) -> matplotlib.figure.Figure:
    """Section 3 detailed physics style — always two panels."""
    epochs = list(range(len(train_losses)))
    train_ma5 = _moving_avg(train_losses, 5)
    val_ma5 = _moving_avg(val_losses, 5)
    fm_ma5 = _moving_avg(fm_losses, 5)
    pde_ma5 = _moving_avg(pde_losses, 5)

    with plt.style.context("dark_background"):
        fig, (ax_loss, ax_lam) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        fig.patch.set_facecolor(THEME.base.bg_figure)

        # --- Panel 1: sub-loss breakdown ---
        ax_loss.set_facecolor(THEME.base.bg_axis)
        if THEME.base.grid.enabled:
            ax_loss.grid(
                True,
                which=THEME.base.grid.which,
                axis=THEME.base.grid.axis,
                linestyle=THEME.base.grid.linestyle,
                alpha=THEME.base.grid.alpha,
                color=THEME.base.grid.color,
            )

        ax_loss.plot(
            epochs,
            train_ma5,
            color="#f1c40f",
            linewidth=2,
            zorder=5,
            label="Total Train (MA-5)",
        )
        ax_loss.plot(
            epochs,
            fm_ma5,
            color=THEME.base.color_fm,
            linewidth=THEME.base.line_width_sub,
            alpha=0.8,
            label="FM Data Loss (MA-5)",
        )
        ax_loss.plot(
            epochs,
            pde_ma5,
            color=THEME.base.color_pde,
            linewidth=THEME.base.line_width_sub,
            alpha=0.8,
            label="PDE Residual (MA-5)",
        )
        ax_loss.plot(
            epochs,
            val_ma5,
            color="#ecf0f1",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label="Val Loss (MA-5)",
        )

        ax_loss.set_yscale("log")
        ax_loss.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=10)
        )
        ax_loss.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(
                base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12
            )
        )
        ax_loss.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_loss.grid(True, which="both", linestyle="--", alpha=0.4, color="#666666")

        ax_loss.set_ylabel("Loss (log)", fontsize=11, fontweight="bold")
        ax_loss.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax_loss.legend(
            facecolor=THEME.base.bg_axis,
            edgecolor=THEME.base.grid.color,
            loc="upper right",
        )
        if title:
            ax_loss.set_title(
                title,
                fontsize=THEME.base.font_size_title,
                fontweight=THEME.base.font_weight_title,
                pad=THEME.base.title_pad,
            )

        # --- Panel 2: lambda_phys ---
        ax_lam.set_facecolor(THEME.base.bg_axis)
        if THEME.base.grid.enabled:
            ax_lam.grid(
                True,
                which=THEME.base.grid.which,
                axis=THEME.base.grid.axis,
                linestyle=THEME.base.grid.linestyle,
                alpha=THEME.base.grid.alpha,
                color=THEME.base.grid.color,
            )

        lam_vals = lambda_hist if lambda_hist is not None else [1.0] * len(epochs)
        ax_lam.plot(
            epochs,
            lam_vals,
            color=THEME.base.color_lambda,
            linewidth=THEME.base.line_width_sub + 0.5,
            label="Physics Weight lambda_phys",
        )
        ax_lam.fill_between(
            epochs, 0, lam_vals, color=THEME.base.color_lambda, alpha=0.2
        )
        ax_lam.set_yscale("log")
        ax_lam.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=10)
        )
        ax_lam.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax_lam.set_xlabel("Epoch", fontsize=10)
        ax_lam.set_ylabel(
            r"ReLoBRaLo $\lambda_{phys}$ (log)", fontsize=11, fontweight="bold"
        )
        ax_lam.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))

        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig
