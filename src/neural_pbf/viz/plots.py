from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — for type annotation only
from plotly.subplots import make_subplots

# Data convention throughout this module:
#   T.shape == (NZ, NY, NX)  — axis-0 is depth (Z), axis-2 is scan direction (X).
# Slicing: T[zi, :, :] → XY plane, T[:, yi, :] → XZ, T[:, :, xi] → YZ.


def plot_interactive_volume(
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    step: int,
    max_pts: int = 200000,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Generate Plotly Figure for 3D Volume with adaptive downsampling."""
    T = np.asarray(T)
    nz, ny, nx = T.shape
    total_pts = nx * ny * nz
    stride = 1
    if total_pts > max_pts:
        stride = int(np.ceil((total_pts / max_pts) ** (1 / 3)))

    T_sub = T[::stride, ::stride, ::stride]
    nz_s, ny_s, nx_s = T_sub.shape

    # Use linspace so coordinates span the full physical domain (not (N-1)*d).
    x_lin = np.linspace(0, nx * dx * 1000, nx_s)
    y_lin = np.linspace(0, ny * dy * 1000, ny_s)
    z_lin = np.linspace(0, nz * dz * 1000, nz_s)
    Z_g, Y_g, X_g = np.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    x_pts, y_pts, z_pts = X_g, Y_g, Z_g

    fig = go.Figure(
        data=go.Volume(
            x=x_pts.flatten(),
            y=y_pts.flatten(),
            z=z_pts.flatten(),
            value=T_sub.flatten(),
            isomin=vmin if vmin is not None else np.min(T_sub),
            isomax=vmax if vmax is not None else np.max(T_sub),
            opacity=0.1,
            surface_count=20,
            colorscale="Jet",
            caps=dict(x_show=False, y_show=False, z_show=True),
        )
    )
    fig.update_layout(title=f"3D Interactive Volume - Step {step}")
    return fig


def plot_interactive_composite(
    T: np.ndarray, dx: float, dy: float, dz: float, step: int
):
    """
    Generate an interactive Plotly dashboard with 4 panels:
    - 3D Volume (Top Left)
    - XY Slice (Top Right)
    - XZ Slice (Bottom Left)
    - YZ Slice (Bottom Right)
    """
    T = np.asarray(T)
    nz, ny, nx = T.shape
    idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
    zi, yi, xi = int(idx[0]), int(idx[1]), int(idx[2])
    scale = 1000.0

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
        subplot_titles=(
            "3D Volume (Subsampled)",
            "XY Plane (Top)",
            "XZ Plane (Side)",
            "YZ Plane (Front)",
        ),
    )

    # 1. 3D Volume
    v_fig = plot_interactive_volume(T, dx, dy, dz, step, max_pts=100000)
    fig.add_trace(v_fig.data[0], row=1, col=1)

    x_coords = np.linspace(0, nx * dx * scale, nx)
    y_coords = np.linspace(0, ny * dy * scale, ny)
    z_coords = np.linspace(0, nz * dz * scale, nz)

    # 2. XY Heatmap: T[zi, :, :] shape (NY, NX) → x=X, y=Y
    fig.add_trace(
        go.Heatmap(
            x=x_coords, y=y_coords, z=T[zi, :, :], colorscale="Jet", showscale=False
        ),
        row=1,
        col=2,
    )

    # 3. XZ Heatmap: T[:, yi, :] shape (NZ, NX) → x=X, y=Z
    fig.add_trace(
        go.Heatmap(
            x=x_coords, y=z_coords, z=T[:, yi, :], colorscale="Jet", showscale=False
        ),
        row=2,
        col=1,
    )

    # 4. YZ Heatmap: T[:, :, xi] shape (NZ, NY) → x=Y, y=Z
    fig.add_trace(
        go.Heatmap(
            x=y_coords, y=z_coords, z=T[:, :, xi], colorscale="Jet", showscale=True
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, width=1000, title_text=f"Thermal Composite Dashboard - Step {step}"
    )
    return fig


def plot_interactive_heatmap(
    T: np.ndarray,
    dx: float,
    dy: float,
    step: int,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Generate Plotly Figure for a 2D Heatmap.

    Args:
        T:  2D array with shape ``(NX, NY)`` — Dim0 maps to the horizontal
            x-axis (spacing dx) and Dim1 to the vertical y-axis (spacing dy).
            To pass an XY-plane slice from a 3D field stored as (NZ, NY, NX),
            transpose first: ``plot_interactive_heatmap(T[zi, :, :].T, dx, dy, ...)``.
        dx: Grid spacing along x (horizontal axis) in metres.
        dy: Grid spacing along y (vertical axis) in metres.
    """
    T = np.asarray(T)
    nx, ny = T.shape
    x = np.linspace(0, nx * dx * 1000, nx)
    y = np.linspace(0, ny * dy * 1000, ny)
    fig = go.Figure(
        data=go.Heatmap(
            x=x, y=y, z=T.T, colorscale="Jet", zmin=vmin, zmax=vmax, showscale=True
        )
    )
    fig.update_layout(title=f"2D Interactive Heatmap - Step {step}")
    return fig


def plot_surface_heatmap_mpl(
    ax: Axes,
    T: np.ndarray,
    dx: float,
    dy: float,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Any = "jet",
    unit: str = "m",
    show_colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """
    Plot a 2D heatmap on a Matplotlib Axes with physical units.

    Args:
        T: 2D array [Dim1, Dim2]. Dim1 maps to the horizontal (x) axis with
           spacing dx; Dim2 maps to the vertical (y) axis with spacing dy.
        unit: 'm' or 'mm'.
    """
    T = np.asarray(T)
    scale = 1000.0 if unit == "mm" else 1.0
    lx = T.shape[0] * dx * scale
    ly = T.shape[1] * dy * scale

    # imshow expects [Row, Col] = [Dim2, Dim1] → transpose so Dim1 is on x-axis.
    im = ax.imshow(
        T.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0, lx, 0, ly),
        aspect="auto",
    )
    ax.set_xlabel(xlabel if xlabel else f"X [{unit}]")
    ax.set_ylabel(ylabel if ylabel else f"Y [{unit}]")
    if title:
        ax.set_title(title)

    if show_colorbar:
        cb_ax = inset_axes(
            ax,
            width="3%",
            height="40%",
            loc="upper right",
            bbox_to_anchor=(0.02, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        plt.colorbar(im, cax=cb_ax)
        cb_ax.tick_params(labelsize=8)

    return im


def _slice_xy(T: np.ndarray, zi: int) -> np.ndarray:
    """Return XY plane slice with shape (NX, NY) ready for plot_surface_heatmap_mpl."""
    return T[zi, :, :].T  # (NY, NX).T → (NX, NY): Dim1=NX on x, Dim2=NY on y


def _slice_xz(T: np.ndarray, yi: int) -> np.ndarray:
    """Return XZ plane slice with shape (NX, NZ) ready for plot_surface_heatmap_mpl."""
    return T[:, yi, :].T  # (NZ, NX).T → (NX, NZ): Dim1=NX on x, Dim2=NZ on y


def _slice_yz(T: np.ndarray, xi: int) -> np.ndarray:
    """Return YZ plane slice with shape (NY, NZ) ready for plot_surface_heatmap_mpl."""
    return T[:, :, xi].T  # (NZ, NY).T → (NY, NZ): Dim1=NY on x, Dim2=NZ on y


def plot_cross_sections(
    fig: Figure,
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    slice_indices: tuple[int, int, int] | None = None,
    unit: str = "mm",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    show_colorbar: bool = True,
):
    """
    Create a 3-view cross-section plot (XY, XZ, YZ planes).

    Args:
        slice_indices: Voxel indices as ``(xi, yi, zi)`` — X-index first, Z-index last.
            Note: this is the *physical* axis order (X, Y, Z), opposite to the
            storage order (NZ, NY, NX). ``np.unravel_index`` on a (NZ, NY, NX) tensor
            returns ``(zi, yi, xi)`` — transpose before passing here.
    """
    T = np.asarray(T)
    if slice_indices is None:
        idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
        zi, yi, xi = int(idx[0]), int(idx[1]), int(idx[2])
    else:
        xi, yi, zi = slice_indices
    scale = 1000.0 if unit == "mm" else 1.0

    ax1 = fig.add_subplot(131)  # XY
    ax2 = fig.add_subplot(132)  # XZ
    ax3 = fig.add_subplot(133)  # YZ

    im = plot_surface_heatmap_mpl(
        ax1,
        _slice_xy(T, zi),
        dx,
        dy,
        title=f"XY Plane (z={zi * dz * scale:.2f} {unit})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
        xlabel=f"X [{unit}]",
        ylabel=f"Y [{unit}]",
    )

    plot_surface_heatmap_mpl(
        ax2,
        _slice_xz(T, yi),
        dx,
        dz,
        title=f"XZ Plane (y={yi * dy * scale:.2f} {unit})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
        xlabel=f"X [{unit}]",
        ylabel=f"Z [{unit}]",
    )

    plot_surface_heatmap_mpl(
        ax3,
        _slice_yz(T, xi),
        dy,
        dz,
        title=f"YZ Plane (x={xi * dx * scale:.2f} {unit})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
        xlabel=f"Y [{unit}]",
        ylabel=f"Z [{unit}]",
    )

    if show_colorbar:
        fig.subplots_adjust(bottom=0.20, wspace=0.3)
        cbar_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
        cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cb.ax.set_xlabel("Temperature [K]", fontsize=10)

    return (ax1, ax2, ax3)


def plot_composite_thermal_view(
    fig: Figure,
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    step: int,
    unit: str = "mm",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Combined view with 3D block on top and XY/XZ/YZ cross-sections below."""
    T = np.asarray(T)
    gs = GridSpec(
        2, 4, figure=fig, width_ratios=[0.12, 1, 1, 1], height_ratios=[2.5, 1]
    )

    ax_cb = fig.add_subplot(gs[0, 0])
    ax_cb.set_axis_off()

    ax_3d = fig.add_subplot(gs[0, 1:], projection="3d")
    plot_3d_block_mpl_ax(
        ax_3d,
        T,
        dx,
        dy,
        dz,
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        dist=10.0,
    )

    ax_xy = fig.add_subplot(gs[1, 1])
    ax_xz = fig.add_subplot(gs[1, 2])
    ax_yz = fig.add_subplot(gs[1, 3])

    idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
    zi, yi, xi = int(idx[0]), int(idx[1]), int(idx[2])

    im = plot_surface_heatmap_mpl(
        ax_xy,
        _slice_xy(T, zi),
        dx,
        dy,
        unit=unit,
        title="Top (XY)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        xlabel=f"X [{unit}]",
        ylabel=f"Y [{unit}]",
    )
    plot_surface_heatmap_mpl(
        ax_xz,
        _slice_xz(T, yi),
        dx,
        dz,
        unit=unit,
        title="Side (XZ)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        xlabel=f"X [{unit}]",
        ylabel=f"Z [{unit}]",
    )
    plot_surface_heatmap_mpl(
        ax_yz,
        _slice_yz(T, xi),
        dy,
        dz,
        unit=unit,
        title="Front (YZ)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        xlabel=f"Y [{unit}]",
        ylabel=f"Z [{unit}]",
    )

    cbar_ax_ins = inset_axes(
        ax_cb,
        width="40%",
        height="85%",
        loc="center",
        bbox_to_anchor=(0.0, 0.0, 1, 1),
        bbox_transform=ax_cb.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=cbar_ax_ins, orientation="vertical")
    cb.ax.set_ylabel("Temperature [K]", fontsize=12, labelpad=10)
    cb.ax.tick_params(labelsize=10)

    fig.subplots_adjust(
        left=0.02, right=0.95, top=0.98, bottom=0.05, wspace=0.5, hspace=0.1
    )


def plot_3d_block_mpl_ax(  # type: ignore[no-untyped-def]
    ax,  # Axes3D — left untyped; pyright can't model 3D-specific attrs (dist, zaxis)
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    unit: str = "mm",
    title: str | None = None,
    show_colorbar: bool = True,
    dist: float = 10.0,
):
    """Helper to plot 3D block onto an EXISTING 3D axes.

    T must have shape (NZ, NY, NX).
    """
    T = np.asarray(T)
    nz, ny, nx = T.shape
    scale = 1000.0 if unit == "mm" else 1.0
    Lx, Ly, Lz = nx * dx * scale, ny * dy * scale, nz * dz * scale

    norm = Normalize(vmin=vmin if vmin else T.min(), vmax=vmax if vmax else T.max())
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    max_dim = max(nx, ny)
    stride = max(int(max_dim / 150), 1) if max_dim > 150 else 1

    # Top face (Z = Lz): meshgrid is (NY, NX), colors from T[-1, :, :] shape (NY, NX)
    X_t, Y_t = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    ax.plot_surface(
        X_t,
        Y_t,
        np.full_like(X_t, Lz),
        facecolors=m.to_rgba(T[-1, :, :]),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    # Side face (Y = 0): meshgrid is (NZ, NX), colors from T[:, 0, :] shape (NZ, NX)
    X_s, Z_s = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
    ax.plot_surface(
        X_s,
        np.zeros_like(X_s),
        Z_s,
        facecolors=m.to_rgba(T[:, 0, :]),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    # Front face (X = Lx): meshgrid is (NZ, NY), colors from T[:, :, -1] shape (NZ, NY)
    Y_f, Z_f = np.meshgrid(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz))
    ax.plot_surface(
        np.full_like(Y_f, Lx),
        Y_f,
        Z_f,
        facecolors=m.to_rgba(T[:, :, -1]),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.dist = dist
    ax.view_init(elev=30, azim=-60)

    if title:
        ax.set_title(title, pad=10, fontsize=14)
    ax.set_xlabel(f"X [{unit}]", labelpad=15, fontsize=10)
    ax.set_ylabel(f"Y [{unit}]", labelpad=15, fontsize=10)
    ax.text2D(
        -0.08,
        0.5,
        f"Z [{unit}]",
        transform=ax.transAxes,
        rotation="vertical",
        va="center",
        ha="right",
        fontsize=10,
    )

    ax.tick_params(axis="x", pad=5, labelsize=9)
    ax.tick_params(axis="y", pad=5, labelsize=9)
    ax.tick_params(axis="z", pad=4, labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    if show_colorbar:
        cb_ax = inset_axes(
            ax,
            width="3%",
            height="40%",
            loc="upper right",
            bbox_to_anchor=(0.02, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        plt.colorbar(m, cax=cb_ax)
        cb_ax.tick_params(labelsize=8)

    return ax


def get_phase_colormap():
    """Create a custom colormap for material phases: Powder, Solid, Mushy, Liquid."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.4, 0.4, 0.4),  # 0.0: Powder (Grey)
        (0.27, 0.51, 0.71),  # 0.5: Solid (SteelBlue)
        (1.0, 0.84, 0.0),  # 1.0: Liquid (Gold/Yellow)
    ]
    return LinearSegmentedColormap.from_list("phase_map", colors, N=256)


def plot_phase_sections(
    fig: Figure,
    T: np.ndarray,
    mask: np.ndarray,
    T_solidus: float,
    T_liquidus: float,
    dx: float,
    dy: float,
    dz: float,
    gs: GridSpec,
    row: int = 0,
    slice_indices: tuple[int, int, int] | None = None,
    unit: str = "mm",
):
    """
    Plot 3 phase-state cross-sections (XY, XZ, YZ) in a specific GridSpec row.

    Args:
        slice_indices: (xi, yi, zi) voxel indices for the cross-section planes.
    """
    T = np.asarray(T)
    mask = np.asarray(mask)
    if slice_indices is None:
        iz = T.shape[0] // 2
        iy = T.shape[1] // 2
        ix = int(np.argmax(T[iz, iy, :]))
        slice_indices = (ix, iy, iz)

    ix, iy, iz = slice_indices

    phi = np.clip((T - T_solidus) / (T_liquidus - T_solidus + 1e-9), 0, 1)
    phase = np.zeros_like(T)
    phase[mask > 0] = 0.5 + 0.5 * phi[mask > 0]

    cmap = get_phase_colormap()

    # Slices in (NX, N*) order so plot_surface_heatmap_mpl maps Dim1 to x-axis
    slices = [_slice_xy(phase, iz), _slice_xz(phase, iy), _slice_yz(phase, ix)]
    titles = ["XY Phase", "XZ Phase", "YZ Phase"]
    deltas = [(dx, dy), (dx, dz), (dy, dz)]
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

    for i, (s, (d1, d2), (lx, ly)) in enumerate(
        zip(slices, deltas, labels, strict=False)
    ):
        ax = fig.add_subplot(gs[row, i])
        plot_surface_heatmap_mpl(
            ax,
            s,
            d1,
            d2,
            title=titles[i],
            vmin=0,
            vmax=1,
            cmap=cmap,
            unit=unit,
            show_colorbar=(i == 2),
            xlabel=f"{lx} [{unit}]",
            ylabel=f"{ly} [{unit}]",
        )
        if i == 2 and hasattr(ax, "images") and len(ax.images) > 0:
            im = ax.images[-1]
            if hasattr(im, "colorbar") and im.colorbar is not None:
                cax = im.colorbar.ax
                cax.set_yticks([0, 0.5, 1.0])
                cax.set_yticklabels(["Powder", "Solid", "Liquid"])


def plot_dual_thermal_phase_view(
    fig: Figure,
    T: np.ndarray,
    mask: np.ndarray,
    T_solidus: float,
    T_liquidus: float,
    dx: float,
    dy: float,
    dz: float,
    step: int,
    unit: str = "mm",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Combined view:
    Row 0: Temperature (XY, XZ, YZ)
    Row 1: Phase State (XY, XZ, YZ)
    """
    T = np.asarray(T)
    mask = np.asarray(mask)
    fig.clear()
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    idx = np.unravel_index(np.argmax(T), T.shape)
    zi, yi, xi = int(idx[0]), int(idx[1]), int(idx[2])
    slice_indices = (xi, yi, zi)  # (xi, yi, zi) convention for plot_phase_sections

    slices_t = [_slice_xy(T, zi), _slice_xz(T, yi), _slice_yz(T, xi)]
    titles_t = ["XY Temperature", "XZ Temperature", "YZ Temperature"]
    deltas = [(dx, dy), (dx, dz), (dy, dz)]
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

    for i, (s, (d1, d2), (lx, ly)) in enumerate(
        zip(slices_t, deltas, labels, strict=False)
    ):
        ax = fig.add_subplot(gs[0, i])
        plot_surface_heatmap_mpl(
            ax,
            s,
            d1,
            d2,
            title=titles_t[i],
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            unit=unit,
            show_colorbar=(i == 2),
            xlabel=f"{lx} [{unit}]",
            ylabel=f"{ly} [{unit}]",
        )

    plot_phase_sections(
        fig,
        T,
        mask,
        T_solidus,
        T_liquidus,
        dx,
        dy,
        dz,
        gs=gs,
        row=1,
        slice_indices=slice_indices,
        unit=unit,
    )

    fig.suptitle(f"Thermal & Phase State - Step {step}", fontsize=14)
