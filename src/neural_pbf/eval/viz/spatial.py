"""Spatial comparison plots: triple-view, profile slices, isotherm overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.figure
import numpy as np
import torch

from neural_pbf.schemas.viz import THEME

# Only switch to Agg when no interactive display is available and the backend
# hasn't already been configured by the caller (e.g. a GUI or notebook session).
if matplotlib.get_backend().lower() in {"tkagg", "qt5agg", "qt4agg", "wxagg", "macosx"}:
    import os as _os

    if not _os.environ.get("DISPLAY") and not _os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")

import matplotlib.pyplot as plt


def triple_view(
    T_gt: torch.Tensor,
    T_pred: torch.Tensor,
    *,
    title: str = "",
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Side-by-side GT | Pred | |Error| panel.

    Args:
        T_gt:     Ground-truth temperature field (any LPBF shape).
        T_pred:   Predicted temperature field (same shape as T_gt).
        title:    Optional figure title.
        savepath: If provided, save the figure to this path.
    """
    gt = _to_2d(T_gt)
    pred = _to_2d(T_pred)
    err = np.abs(pred - gt)

    vmin = float(min(gt.min(), pred.min()))
    vmax = float(max(gt.max(), pred.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _imshow(axes[0], gt, "GT", vmin, vmax, cmap="inferno")
    _imshow(axes[1], pred, "Pred", vmin, vmax, cmap="inferno")
    im = axes[2].imshow(err, cmap="hot", origin="lower")
    axes[2].set_title("|Error| [K]")
    plt.colorbar(im, ax=axes[2])

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


def profile_slices(
    T_gt: torch.Tensor,
    T_pred: torch.Tensor,
    *,
    title: str = "",
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """1-D temperature profiles through the GT peak temperature location.

    Plots profiles along X (row through peak) and Y (column through peak).
    """
    gt = _to_2d(T_gt)
    pred = _to_2d(T_pred)

    flat_idx = int(gt.argmax())
    iy, ix = divmod(flat_idx, gt.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gt[iy, :], label="GT")
    axes[0].plot(pred[iy, :], label="Pred", linestyle="--")
    axes[0].set_title(f"Profile along X (y={iy})")
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].legend()

    axes[1].plot(gt[:, ix], label="GT")
    axes[1].plot(pred[:, ix], label="Pred", linestyle="--")
    axes[1].set_title(f"Profile along Y (x={ix})")
    axes[1].set_xlabel("Y index")
    axes[1].set_ylabel("Temperature [K]")
    axes[1].legend()

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


def isotherm_overlay(
    T_gt: torch.Tensor,
    T_pred: torch.Tensor,
    T_iso: float,
    *,
    title: str = "",
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Overlay the T_iso contour of GT (blue) and Pred (red)."""
    gt = _to_2d(T_gt)
    pred = _to_2d(T_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(gt, cmap="inferno", origin="lower", alpha=0.4)
    try:
        ax.contour(
            gt, levels=[T_iso], colors=["blue"], linestyles=["-"], linewidths=[2]
        )
        ax.contour(
            pred, levels=[T_iso], colors=["red"], linestyles=["--"], linewidths=[2]
        )
    except ValueError:
        pass  # contour() raises ValueError when no level intersects the field

    ax.set_title(title or f"Isotherm T={T_iso:.0f} K — blue=GT, red=Pred")
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


def gallery_evolution(
    T_gt: torch.Tensor,
    T_pred_history: list[torch.Tensor],
    *,
    epoch_labels: list[str] | None = None,
    title: str = "",
    savepath: Path | str | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """2-row × 10-col learning-progress gallery for a single sample.

    Column layout: GT | pred_0 | pred_1 | … | pred_8  (10 cols total)
    Row 0: Surface (XY, last depth slice).
    Row 1: Depth   (XZ, mid-Y slice).

    Milestone selection:
        len ≤ 9  → use all entries as prediction columns.
        len > 9  → pick 9 via ``linspace(0, N-1, 9)`` (includes both endpoints).

    Args:
        T_gt:            Ground-truth field (any LPBF shape).
        T_pred_history:  Ordered list of predicted fields, one per training milestone.
        epoch_labels:    Optional list of column header strings (len == len history).
        title:           Optional figure suptitle.
        savepath:        If given, save the figure to this path.
        dpi:             Save resolution (default 300).

    Note:
        The returned figure is owned by the caller. Call ``plt.close(fig)`` to
        avoid resource leaks.

    Raises:
        ValueError: If *T_pred_history* is empty or *epoch_labels* has wrong length.
    """
    if len(T_pred_history) == 0:
        raise ValueError("T_pred_history must not be empty.")
    if epoch_labels is not None and len(epoch_labels) != len(T_pred_history):
        raise ValueError(
            f"epoch_labels length ({len(epoch_labels)}) must match "
            f"T_pred_history length ({len(T_pred_history)})."
        )

    N = len(T_pred_history)
    if N > 9:
        indices = [int(round(i * (N - 1) / 8)) for i in range(9)]
    else:
        indices = list(range(N))

    selected_preds = [T_pred_history[i] for i in indices]
    n_cols = 1 + len(selected_preds)

    gt = _squeeze_to_3d(T_gt)
    mid_y = gt.shape[1] // 2
    gt_xy, gt_xz = gt[-1, :, :], gt[:, mid_y, :]

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, n_cols, figsize=(22, 7))
        plt.subplots_adjust(left=0.06, bottom=0.08, top=0.90, right=0.98)

        axes[0, 0].imshow(gt_xy, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[0, 0].set_title("GT Surface", fontsize=10)
        axes[0, 0].axis("off")
        axes[1, 0].imshow(gt_xz, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[1, 0].set_title("GT Depth", fontsize=10)
        axes[1, 0].axis("off")

        for col_offset, (pred_tensor, hist_idx) in enumerate(
            zip(selected_preds, indices, strict=False)
        ):
            pred = _squeeze_to_3d(pred_tensor)
            pred_xy = pred[-1, :, :]
            pred_xz = pred[:, pred.shape[1] // 2, :]
            col = col_offset + 1
            label = (
                epoch_labels[hist_idx] if epoch_labels is not None else f"Ep {hist_idx}"
            )
            axes[0, col].imshow(pred_xy, vmin=0, vmax=1, cmap="magma", origin="lower")
            axes[0, col].set_title(label, fontsize=10)
            axes[0, col].axis("off")
            axes[1, col].imshow(pred_xz, vmin=0, vmax=1, cmap="magma", origin="lower")
            axes[1, col].set_title(label, fontsize=10)
            axes[1, col].axis("off")

        fig.text(
            0.03,
            0.71,
            "SURFACE",
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="#3498db",
        )
        fig.text(
            0.03,
            0.27,
            "DEPTH",
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="#e67e22",
        )

        if title:
            fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches=None)

    return fig


def gallery_test(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    titles: list[str] | None = None,
    title: str = "",
    savepath: Path | str | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """2-row × 10-col test gallery for up to 5 distinct test samples.

    Column layout: GT₁ Pred₁ GT₂ Pred₂ GT₃ Pred₃ GT₄ Pred₄ GT₅ Pred₅
    Row 0: Surface (XY), Row 1: Depth (XZ).
    Unused columns (when fewer than 5 samples are provided) are hidden.

    Args:
        samples: List of 1–5 ``(T_gt, T_pred)`` tensor pairs.
        titles:  Optional list of sample-name strings for column headers.
        title:   Optional figure suptitle.
        savepath: If given, save the figure to this path.
        dpi:     Save resolution (default 300).

    Note:
        The returned figure is owned by the caller. Call ``plt.close(fig)`` to
        avoid resource leaks.

    Raises:
        ValueError: If *samples* is empty or contains more than 5 pairs.
    """
    n = len(samples)
    if n == 0 or n > 5:
        raise ValueError(f"gallery_test requires 1–5 samples, got {n}.")

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 10, figsize=(22, 7))
        fig.patch.set_facecolor(THEME.spatial.bg_figure)
        plt.subplots_adjust(left=0.06, bottom=0.08, top=0.90, right=0.98)

        for i, (T_gt_s, T_pred_s) in enumerate(samples):
            gt = _squeeze_to_3d(T_gt_s)
            pred = _squeeze_to_3d(T_pred_s)
            mid_y = gt.shape[1] // 2
            gt_xy, gt_xz = gt[-1, :, :], gt[:, mid_y, :]
            pred_xy, pred_xz = pred[-1, :, :], pred[:, pred.shape[1] // 2, :]
            label = titles[i] if titles else f"S{i + 1}"
            gt_col, pred_col = 2 * i, 2 * i + 1

            axes[0, gt_col].imshow(
                gt_xy, vmin=0, vmax=1, cmap=THEME.spatial.cmap, origin="lower"
            )
            axes[0, gt_col].set_title(
                f"GT {label}",
                fontsize=THEME.spatial.font_size_gallery,
                pad=THEME.spatial.title_pad_gallery,
                color=THEME.spatial.color_gt,
                fontweight=THEME.spatial.font_weight_title,
            )
            axes[0, gt_col].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            if THEME.spatial.grid.enabled:
                axes[0, gt_col].grid(
                    True,
                    which=THEME.spatial.grid.which,
                    axis=THEME.spatial.grid.axis,
                    linestyle=THEME.spatial.grid.linestyle,
                    alpha=THEME.spatial.grid.alpha,
                    color=THEME.spatial.grid.color,
                )

            axes[0, pred_col].imshow(
                pred_xy, vmin=0, vmax=1, cmap=THEME.spatial.cmap, origin="lower"
            )
            axes[0, pred_col].set_title(
                f"Pred {label}",
                fontsize=THEME.spatial.font_size_gallery,
                pad=THEME.spatial.title_pad_gallery,
                color=THEME.spatial.color_gt,
                fontweight=THEME.spatial.font_weight_title,
            )
            axes[0, pred_col].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            if THEME.spatial.grid.enabled:
                axes[0, pred_col].grid(
                    True,
                    which=THEME.spatial.grid.which,
                    axis=THEME.spatial.grid.axis,
                    linestyle=THEME.spatial.grid.linestyle,
                    alpha=THEME.spatial.grid.alpha,
                    color=THEME.spatial.grid.color,
                )

            axes[1, gt_col].imshow(
                gt_xz, vmin=0, vmax=1, cmap=THEME.spatial.cmap, origin="lower"
            )
            axes[1, gt_col].set_title(
                f"GT {label}",
                fontsize=THEME.spatial.font_size_gallery,
                pad=THEME.spatial.title_pad_gallery,
                color=THEME.spatial.color_gt,
                fontweight=THEME.spatial.font_weight_title,
            )
            axes[1, gt_col].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            if THEME.spatial.grid.enabled:
                axes[1, gt_col].grid(
                    True,
                    which=THEME.spatial.grid.which,
                    axis=THEME.spatial.grid.axis,
                    linestyle=THEME.spatial.grid.linestyle,
                    alpha=THEME.spatial.grid.alpha,
                    color=THEME.spatial.grid.color,
                )

            axes[1, pred_col].imshow(
                pred_xz, vmin=0, vmax=1, cmap=THEME.spatial.cmap, origin="lower"
            )
            axes[1, pred_col].set_title(
                f"Pred {label}",
                fontsize=THEME.spatial.font_size_gallery,
                pad=THEME.spatial.title_pad_gallery,
                color=THEME.spatial.color_gt,
                fontweight=THEME.spatial.font_weight_title,
            )
            axes[1, pred_col].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            if THEME.spatial.grid.enabled:
                axes[1, pred_col].grid(
                    True,
                    which=THEME.spatial.grid.which,
                    axis=THEME.spatial.grid.axis,
                    linestyle=THEME.spatial.grid.linestyle,
                    alpha=THEME.spatial.grid.alpha,
                    color=THEME.spatial.grid.color,
                )

        # Hide unused columns when fewer than 5 samples provided
        for i in range(n, 5):
            for col in (2 * i, 2 * i + 1):
                axes[0, col].axis("off")
                axes[1, col].axis("off")

        fig.text(
            0.03,
            0.71,
            "SURFACE",
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="#3498db",
        )
        fig.text(
            0.03,
            0.27,
            "DEPTH",
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="#e67e22",
        )

        if title:
            fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches=None)

    return fig


def val_grid_2x2(
    T_gt: torch.Tensor,
    T_pred: torch.Tensor,
    *,
    epoch: int = 0,
    title: str = "",
    savepath: Path | str | None = None,
) -> matplotlib.figure.Figure:
    """Gold-standard 2x2 GT vs Pred validation grid.

    Layout:
        [0,0] GT Surface (XY)   | [0,1] Pred Surface (Ep N)
        [1,0] GT Depth (XZ)     | [1,1] Pred Depth (XZ)

    Args:
        T_gt:     Ground-truth temperature field (any LPBF shape).
        T_pred:   Predicted temperature field (same shape).
        epoch:    Current training epoch (for title).
        title:    Optional figure suptitle.
        savepath: If provided, save the figure to this path.

    Note:
        The returned figure is owned by the caller. Call ``plt.close(fig)``
        when done to avoid resource leaks.
    """
    gt = _squeeze_to_3d(T_gt)
    pred = _squeeze_to_3d(T_pred)

    mid_y = gt.shape[1] // 2

    gt_xy = gt[-1, :, :]
    gt_xz = gt[:, mid_y, :]
    pred_xy = pred[-1, :, :]
    pred_xz = pred[:, mid_y, :]

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(gt_xy, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[0, 0].set_title("GT Surface (XY)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pred_xy, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[0, 1].set_title(f"Pred Surface (Ep {epoch})")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(gt_xz, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[1, 0].set_title("GT Depth (XZ)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_xz, vmin=0, vmax=1, cmap="magma", origin="lower")
        axes[1, 1].set_title(f"Pred Depth (Ep {epoch})")
        axes[1, 1].axis("off")

        if title:
            fig.suptitle(title)

        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=120, bbox_inches="tight")

    return fig


# ── internal helpers ──────────────────────────────────────────────────────────


def _squeeze_to_3d(T: torch.Tensor) -> np.ndarray:
    """Squeeze any LPBF tensor shape down to a (D, H, W) numpy array.

    Handles shapes such as:
        (B, C, D, H, W)   — standard 3D batch
        (B, C, 1, D, H, W) — extra batch/channel dim (RoPE models)
        (1, D, H, W)       — single-sample 3D without channel
        (1, 1, H, W)       — 2D batch: treated as (1, H, W) i.e. D=1
    """
    if T.numel() == 0:
        raise ValueError(f"Cannot plot empty tensor (shape {tuple(T.shape)})")
    arr = T.detach().cpu().float().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # treat as (1, H, W)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array after squeezing, got {arr.ndim}D from {tuple(T.shape)}"
        )
    return arr


def _to_2d(T: torch.Tensor) -> np.ndarray:
    """Reduce any LPBF tensor shape to a 2-D numpy array for plotting."""
    arr = T.detach().cpu().float().numpy()
    while arr.ndim > 2:
        mid = arr.shape[0] // 2
        arr = arr[mid]
    return arr


def _imshow(
    ax: matplotlib.axes.Axes,
    arr: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "inferno",
) -> None:
    im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
