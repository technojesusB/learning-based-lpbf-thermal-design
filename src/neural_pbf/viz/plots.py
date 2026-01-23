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
from plotly.subplots import make_subplots


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
    nx, ny, nz = T.shape
    total_pts = nx * ny * nz
    stride = 1
    if total_pts > max_pts:
        stride = int(np.ceil((total_pts / max_pts) ** (1 / 3)))

    T_sub = T[::stride, ::stride, ::stride]
    nx_s, ny_s, nz_s = T_sub.shape

    X, Y, Z = np.mgrid[0:nx_s, 0:ny_s, 0:nz_s]
    X = X * (dx * stride) * 1000
    Y = Y * (dy * stride) * 1000
    Z = Z * (dz * stride) * 1000

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
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
    # Find max T for slice indices
    idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
    xi, yi, zi = idx
    nx, ny, nz = T.shape
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

    # 1. 3D Volume (Subsampled for performance)
    # We'll use a very aggressive downsample for the composite view to keep it snappy
    v_fig = plot_interactive_volume(T, dx, dy, dz, step, max_pts=100000)
    fig.add_trace(v_fig.data[0], row=1, col=1)

    # 2. XY Heatmap
    x_coords = np.linspace(0, nx * dx * scale, nx)
    y_coords = np.linspace(0, ny * dy * scale, ny)
    fig.add_trace(
        go.Heatmap(
            x=x_coords, y=y_coords, z=T[:, :, zi].T, colorscale="Jet", showscale=False
        ),
        row=1,
        col=2,
    )

    # 3. XZ Heatmap
    z_coords = np.linspace(0, nz * dz * scale, nz)
    fig.add_trace(
        go.Heatmap(
            x=x_coords, y=z_coords, z=T[:, yi, :].T, colorscale="Jet", showscale=False
        ),
        row=2,
        col=1,
    )

    # 4. YZ Heatmap
    fig.add_trace(
        go.Heatmap(
            x=y_coords, y=z_coords, z=T[xi, :, :].T, colorscale="Jet", showscale=True
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
    """Generate Plotly Figure for 2D Heatmap"""
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
    cmap: str = "jet",
    unit: str = "m",
    show_colorbar: bool = True,
):
    """
    Plot a 2D heatmap on a Matplotlib Axes with physical units.

    Args:
        ax: Matplotlib axes.
        T: 2D array [X, Y].
        dx, dy: Grid spacing in meters.
        unit: 'm' or 'mm'. If 'mm', axis labels and extent are scaled.
    """
    scale = 1000.0 if unit == "mm" else 1.0
    lx = T.shape[0] * dx * scale
    ly = T.shape[1] * dy * scale

    # Extent: [left, right, bottom, top]
    # standard imshow origin='lower' expects [0, Lx, 0, Ly] (transposed T?)
    # T is usually [X, Y]. Imshow expects [Row, Col] -> [Y, X].
    # So we plot T.T

    im = ax.imshow(
        T.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0, lx, 0, ly),
        aspect="auto",
    )
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
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
        slice_indices: (x_idx, y_idx, z_idx) for the slices.
                       If None, uses the center or max T location.
    """
    if slice_indices is None:
        # Use location of Max T as interesting point
        idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
        # Ensure it's a 3-tuple
        xi, yi, zi = int(idx[0]), int(idx[1]), int(idx[2])
    else:
        xi, yi, zi = slice_indices
    nx, ny, nz = T.shape
    scale = 1000.0 if unit == "mm" else 1.0

    # 3 subplots: Top (XY), Side-X (XZ), Side-Y (YZ)
    ax1 = fig.add_subplot(131)  # XY
    ax2 = fig.add_subplot(132)  # XZ
    ax3 = fig.add_subplot(133)  # YZ

    # 1. XY Slice (Top view at zi)
    im = plot_surface_heatmap_mpl(
        ax1,
        T[:, :, zi],
        dx,
        dy,
        title=f"XY Plane (z={zi * dz * scale:.2f})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
    )

    # 2. XZ Slice (Side view at yi)
    # T is [X, Y, Z]. Slice at yi -> [X, Z]
    plot_surface_heatmap_mpl(
        ax2,
        T[:, yi, :],
        dx,
        dz,
        title=f"XZ Plane (y={yi * dy * scale:.2f})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
    )
    ax2.set_ylabel(f"Z [{unit}]")

    # 3. YZ Slice (Front view at xi)
    # T is [X, Y, Z]. Slice at xi -> [Y, Z]
    plot_surface_heatmap_mpl(
        ax3,
        T[xi, :, :],
        dy,
        dz,
        title=f"YZ Plane (x={xi * dx * scale:.2f})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_colorbar=False,
    )
    ax3.set_xlabel(f"Y [{unit}]")
    ax3.set_ylabel(f"Z [{unit}]")

    if show_colorbar:
        # Shared horizontal colorbar at the bottom.
        # Reduced bottom margin and positioned colorbar for better compactness.
        fig.subplots_adjust(bottom=0.20, wspace=0.3)
        cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
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
    """Combined view with 3D block and Cross-sections."""
    # Top half: 3D Block
    # Bottom half: 3 Cross sections

    # Use GridSpec for better control
    # Column 0 will be for the shared colorbar
    # height_ratios: make the 3D plot (top row) much taller
    gs = GridSpec(
        2, 4, figure=fig, width_ratios=[0.12, 1, 1, 1], height_ratios=[2.5, 1]
    )

    # shared colorbar axes (top half, left)
    ax_cb = fig.add_subplot(gs[0, 0])
    ax_cb.set_axis_off()

    ax_3d = fig.add_subplot(gs[0, 1:], projection="3d")
    # Custom 3D block logic inside here...
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

    # Cross sections
    ax_xy = fig.add_subplot(gs[1, 1])
    ax_xz = fig.add_subplot(gs[1, 2])
    ax_yz = fig.add_subplot(gs[1, 3])

    idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
    xi, yi, zi = idx

    im = plot_surface_heatmap_mpl(
        ax_xy,
        T[:, :, zi],
        dx,
        dy,
        unit=unit,
        title="Top (XY)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
    )
    plot_surface_heatmap_mpl(
        ax_xz,
        T[:, yi, :],
        dx,
        dz,
        unit=unit,
        title="Side (XZ)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
    )
    plot_surface_heatmap_mpl(
        ax_yz,
        T[xi, :, :],
        dy,
        dz,
        unit=unit,
        title="Front (YZ)",
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
    )

    # Shared colorbar in the top-left area
    # Create an inset axes within the ax_cb for better control of bar size
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

    # Note: tight_layout might struggle with GridSpec + 3D, so we manual adjust
    fig.subplots_adjust(
        left=0.02, right=0.95, top=0.98, bottom=0.05, wspace=0.5, hspace=0.1
    )


def plot_3d_block_mpl_ax(
    ax,
    T,
    dx,
    dy,
    dz,
    vmin=None,
    vmax=None,
    cmap="jet",
    unit="mm",
    title=None,
    show_colorbar=True,
    dist=10.0,
):
    """Helper to plot 3D block onto an EXISTING axes."""
    nx, ny, nz = T.shape
    scale = 1000.0 if unit == "mm" else 1.0
    Lx, Ly, Lz = nx * dx * scale, ny * dy * scale, nz * dz * scale

    norm = Normalize(vmin=vmin if vmin else T.min(), vmax=vmax if vmax else T.max())
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    # Increase resolution of surfaces by not skipping points
    # (Matplotlib can be slow if we plot every point of 512x256,
    # so maybe rstride/cstride)
    # But usually user wants it sharp.
    stride = 1  # Keep it 1 for resolution

    # Adaptive stride calculation to keep 3D surface plotting fast
    # Matplotlib struggles with >50k points in 3D
    max_dim = max(nx, ny)
    if max_dim > 150:
        # Target roughly 100-150 points along the longest dimension
        stride = int(max_dim / 150)
        stride = max(stride, 1)

    # Top
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    ax.plot_surface(
        X,
        Y,
        np.full_like(X, Lz),
        facecolors=m.to_rgba(T[..., -1].T),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    # Side (Y=0)
    X_s, Z_s = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
    ax.plot_surface(
        X_s,
        np.zeros_like(X_s),
        Z_s,
        facecolors=m.to_rgba(T[:, 0, :].T),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    # Front (X=Lx)
    Y_f, Z_f = np.meshgrid(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz))
    ax.plot_surface(
        np.full_like(Y_f, Lx),
        Y_f,
        Z_f,
        facecolors=m.to_rgba(T[-1, :, :].T),
        shade=False,
        rstride=stride,
        cstride=stride,
    )

    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.dist = dist
    ax.view_init(elev=30, azim=-60)  # Standard perspective to show all labels

    if title:
        ax.set_title(title, pad=10, fontsize=14)
    ax.set_xlabel(f"X [{unit}]", labelpad=15, fontsize=10)
    ax.set_ylabel(f"Y [{unit}]", labelpad=15, fontsize=10)

    # Robust Z-label using 2D axes coordinates
    # (avoids Axes3D clipping/bounding box issues)
    # Positioned to the left of the axes box
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
    # ax.set_zlabel is unreliable with bbox_inches='tight' in 3D

    # Improve tick labels to avoid overlap
    ax.tick_params(axis="x", pad=5, labelsize=9)
    ax.tick_params(axis="y", pad=5, labelsize=9)
    ax.tick_params(axis="z", pad=4, labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    if show_colorbar:
        # Add colorbar for 3D block
        # Assuming inset_axes is imported globally or available
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
