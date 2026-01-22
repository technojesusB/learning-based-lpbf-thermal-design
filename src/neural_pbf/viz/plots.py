# Optional imports handled by caller or try/except, but this module assumes availability for now
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots


def plot_interactive_volume(
    T: np.ndarray, dx: float, dy: float, dz: float, step: int, max_pts: int = 200000
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
            isomin=np.min(T_sub),
            isomax=np.max(T_sub),
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


def plot_interactive_heatmap(T: np.ndarray, dx: float, dy: float, step: int):
    """Generate Plotly Figure for 2D Heatmap"""
    nx, ny = T.shape
    x = np.linspace(0, nx * dx * 1000, nx)
    y = np.linspace(0, ny * dy * 1000, ny)
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=T.T, colorscale="Jet"))
    fig.update_layout(title=f"2D Interactive Heatmap - Step {step}")
    return fig


def plot_surface_heatmap_mpl(
    ax: plt.Axes,
    T: np.ndarray,
    dx: float,
    dy: float,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    unit: str = "m",
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
        extent=[0, lx, 0, ly],
        aspect="auto",
    )
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    if title:
        ax.set_title(title)

    return im


def plot_3d_block_mpl(
    fig: plt.Figure,
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
    unit: str = "m",
):
    """
    Create a 3D Block visualization (Top, Side, Front faces) on a Matplotlib Figure.

    Args:
        fig: Figure to add 3D axes to.
        T: 3D array [X, Y, Z].
    """
    ax = fig.add_subplot(111, projection="3d")

    nx, ny, nz = T.shape
    scale = 1000.0 if unit == "mm" else 1.0
    Lx = nx * dx * scale
    Ly = ny * dy * scale
    Lz = nz * dz * scale

    # Faces data
    # Top: Z = max, XY plane
    T[..., -1].T  # [Y, X]
    # Side: Y = mid?, actually user wants 'exterior' faces usually
    # Figure 12c shows: Top, Side (XZ plane at Y=0 or Y=Ly?), Front (YZ plane at X=Lx?)
    # Let's plot the bounding box faces.
    # Top (Z=Lz), Front (Y=0), Side (X=Lx) ?
    # Let's match typical view:
    # X axis -> right, Y axis -> depth/right-up, Z axis -> up

    # We will project the surfaces onto the bounding box planes.

    # 1. Top Surface (Z = Lz)
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    Z_top = np.full_like(X, Lz)
    # T map needs matching shape [ny, nx] which is T[..., -1].T
    # Use plot_surface

    # Normalizing colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    # Top
    ax.plot_surface(X, Y, Z_top, facecolors=m.to_rgba(T[..., -1].T), shade=False)

    # Side (Y = 0) - XZ plane
    X_side, Z_side = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
    Y_side = np.full_like(X_side, 0)
    # T slice: T[:, 0, :] -> [X, Z]. Transpose for meshgrid [Z, X]?
    # Meshgrid is [nz, nx]. T slice is [nx, nz]. So transpose.
    ax.plot_surface(
        X_side, Y_side, Z_side, facecolors=m.to_rgba(T[:, 0, :].T), shade=False
    )

    # Front (X = Lx) - YZ plane? Or X=0?
    # Usually we want to see the cut. If we show the block, we usually show Top, Right, Front.
    # Let's show X=Lx (Right) and Y=0 (Front) and Z=Lz (Top).

    # Right (X = Lx)
    # Y, Z mesh
    Y_right, Z_right = np.meshgrid(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz))
    X_right = np.full_like(Y_right, Lx)
    # T slice: T[-1, :, :] -> [Y, Z]. Transpose -> [Z, Y]
    ax.plot_surface(
        X_right, Y_right, Z_right, facecolors=m.to_rgba(T[-1, :, :].T), shade=False
    )

    # Beautify
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    ax.set_zlabel(f"Z [{unit}]")
    ax.set_box_aspect((Lx, Ly, Lz))  # Aspect ratio matching physical dims

    if title:
        ax.set_title(title)

    return ax


def plot_cross_sections(
    fig: plt.Figure,
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    slice_indices: tuple[int, int, int] | None = None,
    unit: str = "mm",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "jet",
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
        slice_indices = idx

    xi, yi, zi = slice_indices
    nx, ny, nz = T.shape
    scale = 1000.0 if unit == "mm" else 1.0

    # 3 subplots: Top (XY), Side-X (XZ), Side-Y (YZ)
    ax1 = fig.add_subplot(131)  # XY
    ax2 = fig.add_subplot(132)  # XZ
    ax3 = fig.add_subplot(133)  # YZ

    # 1. XY Slice (Top view at zi)
    plot_surface_heatmap_mpl(
        ax1,
        T[:, :, zi],
        dx,
        dy,
        title=f"XY Plane (z={zi * dz * scale:.2f})",
        unit=unit,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
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
    )
    ax3.set_xlabel(f"Y [{unit}]")
    ax3.set_ylabel(f"Z [{unit}]")

    plt.tight_layout()
    return (ax1, ax2, ax3)


def plot_composite_thermal_view(
    fig: plt.Figure,
    T: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    step: int,
    unit: str = "mm",
):
    """Combined view with 3D block and Cross-sections."""
    # Top half: 3D Block
    # Bottom half: 3 Cross sections

    # Use GridSpec for better control
    gs = GridSpec(2, 3, figure=fig)

    ax_3d = fig.add_subplot(gs[0, :], projection="3d")
    # Custom 3D block logic inside here...
    plot_3d_block_mpl_ax(
        ax_3d, T, dx, dy, dz, unit=unit, title=f"3D Field - Step {step}"
    )

    # Cross sections
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[1, 2])

    idx = np.unravel_index(np.argmax(T, axis=None), T.shape)
    xi, yi, zi = idx

    plot_surface_heatmap_mpl(ax_xy, T[:, :, zi], dx, dy, unit=unit, title="Top (XY)")
    plot_surface_heatmap_mpl(ax_xz, T[:, yi, :], dx, dz, unit=unit, title="Side (XZ)")
    plot_surface_heatmap_mpl(ax_yz, T[xi, :, :], dy, dz, unit=unit, title="Front (YZ)")

    plt.tight_layout()


def plot_3d_block_mpl_ax(
    ax, T, dx, dy, dz, vmin=None, vmax=None, cmap="jet", unit="mm", title=None
):
    """Helper to plot 3D block onto an EXISTING axes."""
    nx, ny, nz = T.shape
    scale = 1000.0 if unit == "mm" else 1.0
    Lx, Ly, Lz = nx * dx * scale, ny * dy * scale, nz * dz * scale

    norm = plt.Normalize(vmin=vmin if vmin else T.min(), vmax=vmax if vmax else T.max())
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    # Increase resolution of surfaces by not skipping points
    # (Matplotlib can be slow if we plot every point of 512x256, so maybe rstride/cstride)
    # But usually user wants it sharp.
    stride = 1  # Keep it 1 for resolution
    if nx > 200 or ny > 200:
        stride = 2  # Adaptive stride for performance

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

    ax.set_box_aspect((Lx, Ly, Lz))
    if title:
        ax.set_title(title)
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    ax.set_zlabel(f"Z [{unit}]")
    return ax
