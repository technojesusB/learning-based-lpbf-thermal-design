"""Spatial comparison plots: triple-view, profile slices, isotherm overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


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
    except Exception:
        pass  # no contour when isotherm not present in field

    ax.set_title(title or f"Isotherm T={T_iso:.0f} K — blue=GT, red=Pred")
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=100, bbox_inches="tight")

    return fig


# ── internal helpers ──────────────────────────────────────────────────────────


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
