"""Tests for coordinate system correctness in plots.py.

Data convention: T.shape == (NZ, NY, NX) — Z is depth (axis 0), X is scan
direction (axis 2). All plotting functions must honour this layout.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from neural_pbf.viz.plots import (
    plot_3d_block_mpl_ax,
    plot_composite_thermal_view,
    plot_cross_sections,
    plot_interactive_composite,
    plot_interactive_volume,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NZ, NY, NX = 8, 16, 32
DX, DY, DZ = 1.0e-3 / NX, 0.5e-3 / NY, 0.125e-3 / NZ  # ~15 µm voxels


@pytest.fixture()
def uniform_T():
    """Uniform temperature field — all axes look the same, shape (NZ, NY, NX)."""
    return np.full((NZ, NY, NX), 300.0)


@pytest.fixture()
def hot_corner_T():
    """Single hot voxel at (zi=0, yi=0, xi=NX-1) — last X, first Z, first Y.

    This lets us verify that slicing extracts the right plane.
    """
    T = np.full((NZ, NY, NX), 300.0)
    T[0, 0, NX - 1] = 3500.0  # zi=0, yi=0, xi=NX-1
    return T


# ---------------------------------------------------------------------------
# plot_interactive_volume — shape & coordinate range
# ---------------------------------------------------------------------------


class TestInteractiveVolume:
    def test_shape_unpacked_as_nz_ny_nx(self, uniform_T):
        """Volume trace coordinates must span Lx × Ly × Lz, not Lz × Ly × Lx."""
        fig = plot_interactive_volume(uniform_T, DX, DY, DZ, step=0)
        trace = fig.data[0]
        x_range = float(trace.x.max()) - float(trace.x.min())
        y_range = float(trace.y.max()) - float(trace.y.min())
        z_range = float(trace.z.max()) - float(trace.z.min())
        Lx_mm = NX * DX * 1000
        Ly_mm = NY * DY * 1000
        Lz_mm = NZ * DZ * 1000
        assert x_range == pytest.approx(Lx_mm, rel=0.1), "X axis must span Lx"
        assert y_range == pytest.approx(Ly_mm, rel=0.1), "Y axis must span Ly"
        assert z_range == pytest.approx(Lz_mm, rel=0.1), "Z axis must span Lz"


# ---------------------------------------------------------------------------
# plot_interactive_composite — slice axis labels and hot-spot detection
# ---------------------------------------------------------------------------


class TestInteractiveComposite:
    def test_subplot_titles_present(self, uniform_T):
        fig = plot_interactive_composite(uniform_T, DX, DY, DZ, step=1)
        titles = [a.text for a in fig.layout.annotations]
        assert any("XY" in t for t in titles), "XY plane title missing"
        assert any("XZ" in t for t in titles), "XZ plane title missing"
        assert any("YZ" in t for t in titles), "YZ plane title missing"

    def test_xy_heatmap_shape_matches_nx_ny(self, uniform_T):
        """XY heatmap z-matrix must have shape (NY, NX) — not (NX, NY)."""
        fig = plot_interactive_composite(uniform_T, DX, DY, DZ, step=1)
        # Heatmap traces appear in order: vol(row1col1), XY(r1c2), XZ(r2c1), YZ(r2c2)
        xy_trace = fig.data[1]
        z_matrix = np.array(xy_trace.z)
        assert z_matrix.shape == (
            NY,
            NX,
        ), f"XY heatmap z must be (NY={NY}, NX={NX}), got {z_matrix.shape}"

    def test_xz_heatmap_shape_matches_nx_nz(self, uniform_T):
        """XZ heatmap z-matrix must have shape (NZ, NX)."""
        fig = plot_interactive_composite(uniform_T, DX, DY, DZ, step=1)
        xz_trace = fig.data[2]
        z_matrix = np.array(xz_trace.z)
        assert z_matrix.shape == (
            NZ,
            NX,
        ), f"XZ heatmap z must be (NZ={NZ}, NX={NX}), got {z_matrix.shape}"

    def test_yz_heatmap_shape_matches_ny_nz(self, uniform_T):
        """YZ heatmap z-matrix must have shape (NZ, NY)."""
        fig = plot_interactive_composite(uniform_T, DX, DY, DZ, step=1)
        yz_trace = fig.data[3]
        z_matrix = np.array(yz_trace.z)
        assert z_matrix.shape == (
            NZ,
            NY,
        ), f"YZ heatmap z must be (NZ={NZ}, NY={NY}), got {z_matrix.shape}"

    def test_hot_spot_appears_in_xy_heatmap(self, hot_corner_T):
        """Hot voxel at xi=NX-1, yi=0 must be visible in the XY heatmap."""
        fig = plot_interactive_composite(hot_corner_T, DX, DY, DZ, step=1)
        xy_trace = fig.data[1]
        z_matrix = np.array(xy_trace.z)
        # Hot voxel is at (zi=0, yi=0, xi=NX-1). XY slice at zi=0 → T[0,:,:]
        # In the heatmap z-matrix the value at column NX-1, row 0 must be 3500
        assert z_matrix[0, NX - 1] == pytest.approx(
            3500.0
        ), "Hot voxel at xi=NX-1,yi=0 not found at expected position in XY heatmap"


# ---------------------------------------------------------------------------
# plot_cross_sections — slice dimensions
# ---------------------------------------------------------------------------


class TestCrossSections:
    def test_axes_are_returned(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        axes = plot_cross_sections(fig, uniform_T, DX, DY, DZ, vmin=300, vmax=3500)
        assert len(axes) == 3
        plt.close(fig)

    def test_xy_image_has_correct_extent(self, uniform_T):
        """XY plot extent must be (0, Lx_mm, 0, Ly_mm)."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        ax_xy, _, _ = plot_cross_sections(
            fig, uniform_T, DX, DY, DZ, vmin=300, vmax=3500
        )
        im = ax_xy.images[0]
        extent = im.get_extent()  # (xmin, xmax, ymin, ymax)
        Lx_mm = NX * DX * 1000
        Ly_mm = NY * DY * 1000
        assert extent[1] == pytest.approx(Lx_mm, rel=1e-3), "XY: xmax must be Lx_mm"
        assert extent[3] == pytest.approx(Ly_mm, rel=1e-3), "XY: ymax must be Ly_mm"
        plt.close(fig)

    def test_xz_image_has_correct_extent(self, uniform_T):
        """XZ plot extent must be (0, Lx_mm, 0, Lz_mm)."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        _, ax_xz, _ = plot_cross_sections(
            fig, uniform_T, DX, DY, DZ, vmin=300, vmax=3500
        )
        im = ax_xz.images[0]
        extent = im.get_extent()
        Lx_mm = NX * DX * 1000
        Lz_mm = NZ * DZ * 1000
        assert extent[1] == pytest.approx(Lx_mm, rel=1e-3), "XZ: xmax must be Lx_mm"
        assert extent[3] == pytest.approx(Lz_mm, rel=1e-3), "XZ: ymax must be Lz_mm"
        plt.close(fig)

    def test_yz_image_has_correct_extent(self, uniform_T):
        """YZ plot extent must be (0, Ly_mm, 0, Lz_mm)."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        _, _, ax_yz = plot_cross_sections(
            fig, uniform_T, DX, DY, DZ, vmin=300, vmax=3500
        )
        im = ax_yz.images[0]
        extent = im.get_extent()
        Ly_mm = NY * DY * 1000
        Lz_mm = NZ * DZ * 1000
        assert extent[1] == pytest.approx(Ly_mm, rel=1e-3), "YZ: xmax must be Ly_mm"
        assert extent[3] == pytest.approx(Lz_mm, rel=1e-3), "YZ: ymax must be Lz_mm"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_composite_thermal_view — smoke test + axis label check
# ---------------------------------------------------------------------------


class TestCompositeThermalView:
    def test_runs_without_error(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 8))
        plot_composite_thermal_view(
            fig, uniform_T, DX, DY, DZ, step=0, vmin=300, vmax=3500
        )
        plt.close(fig)

    def test_cross_section_axis_labels(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 8))
        plot_composite_thermal_view(
            fig, uniform_T, DX, DY, DZ, step=0, vmin=300, vmax=3500
        )
        axes = [ax for ax in fig.axes if hasattr(ax, "images") and ax.images]
        xlabels = [ax.get_xlabel() for ax in axes]
        ylabels = [ax.get_ylabel() for ax in axes]
        # XY panel: xlabel must reference X, ylabel must reference Y
        assert any("X" in lbl for lbl in xlabels), "No X-axis label found"
        assert any("Y" in lbl for lbl in ylabels), "No Y-axis label found"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_3d_block_mpl_ax — box aspect ratio reflects (Lx, Ly, Lz) not (Lz,...)
# ---------------------------------------------------------------------------


class TestPlot3dBlock:
    def test_box_aspect_matches_domain(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_block_mpl_ax(ax, uniform_T, DX, DY, DZ, vmin=300, vmax=3500)
        aspect = ax.get_box_aspect()
        Lx_mm = NX * DX * 1000
        Ly_mm = NY * DY * 1000
        Lz_mm = NZ * DZ * 1000
        # Matplotlib may normalize the stored aspect; check ratios, not absolute values.
        assert aspect[0] / aspect[1] == pytest.approx(
            Lx_mm / Ly_mm, rel=1e-2
        ), "box_aspect X:Y ratio must equal Lx:Ly"
        assert aspect[0] / aspect[2] == pytest.approx(
            Lx_mm / Lz_mm, rel=1e-2
        ), "box_aspect X:Z ratio must equal Lx:Lz"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_interactive_volume — downsampling path (stride > 1)
# ---------------------------------------------------------------------------


class TestInteractiveVolumeDownsampling:
    def test_downsampling_reduces_points(self, uniform_T):
        """With a tiny max_pts budget, stride > 1 must be applied."""
        fig = plot_interactive_volume(uniform_T, DX, DY, DZ, step=0, max_pts=10)
        trace = fig.data[0]
        assert len(trace.value) < NZ * NY * NX, "Downsampling must reduce point count"

    def test_downsampled_coords_still_span_domain(self, uniform_T):
        """Even with downsampling, x coordinates must be positive and bounded."""
        fig = plot_interactive_volume(uniform_T, DX, DY, DZ, step=0, max_pts=10)
        trace = fig.data[0]
        Lx_mm = NX * DX * 1000
        x_range = float(trace.x.max()) - float(trace.x.min())
        assert x_range > 0, "X range must be positive after downsampling"
        assert x_range <= Lx_mm + 1e-6, "X range must not exceed domain"


# ---------------------------------------------------------------------------
# plot_interactive_heatmap — smoke test
# ---------------------------------------------------------------------------


class TestInteractiveHeatmap:
    def test_2d_heatmap_smoke(self):
        from neural_pbf.viz.plots import plot_interactive_heatmap

        # T shape is (NX, NY): Dim0 on x-axis with spacing DX
        T2d = np.random.uniform(300, 1500, (NX, NY))
        fig = plot_interactive_heatmap(T2d, DX, DY, step=5, vmin=300, vmax=3500)
        assert fig is not None
        assert len(fig.to_json()) > 0

    def test_2d_heatmap_x_axis_spans_lx(self):
        """x-axis must span Lx = NX * DX * 1000, not Ly."""
        from neural_pbf.viz.plots import plot_interactive_heatmap

        T2d = np.ones((NX, NY)) * 300.0  # shape (NX, NY)
        fig = plot_interactive_heatmap(T2d, DX, DY, step=0)
        trace = fig.data[0]
        Lx_mm = NX * DX * 1000
        x_range = float(np.array(trace.x).max()) - float(np.array(trace.x).min())
        assert x_range == pytest.approx(
            Lx_mm, rel=0.1
        ), f"x-axis must span Lx={Lx_mm:.3f} mm, got {x_range:.3f}"


# ---------------------------------------------------------------------------
# Colorbar branch coverage
# ---------------------------------------------------------------------------


class TestColorbars:
    def test_cross_sections_with_colorbar(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))
        axes = plot_cross_sections(
            fig, uniform_T, DX, DY, DZ, vmin=300, vmax=3500, show_colorbar=True
        )
        assert len(axes) == 3
        plt.close(fig)

    def test_3d_block_with_colorbar(self, uniform_T):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        from neural_pbf.viz.plots import plot_3d_block_mpl_ax

        plot_3d_block_mpl_ax(
            ax, uniform_T, DX, DY, DZ, vmin=300, vmax=3500, show_colorbar=True
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_phase_sections and plot_dual_thermal_phase_view
# ---------------------------------------------------------------------------


class TestPhasePlots:
    def test_phase_sections_smoke(self, uniform_T):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        from neural_pbf.viz.plots import plot_phase_sections

        T_s, T_l = 1648.0, 1723.0
        mask = np.ones_like(uniform_T)
        fig = plt.figure(figsize=(12, 4))
        gs = GridSpec(1, 3, figure=fig)
        plot_phase_sections(
            fig, uniform_T, mask, T_s, T_l, DX, DY, DZ, gs=gs, row=0, unit="mm"
        )
        plt.close(fig)

    def test_dual_thermal_phase_view_smoke(self, uniform_T):
        import matplotlib.pyplot as plt

        from neural_pbf.viz.plots import plot_dual_thermal_phase_view

        T_s, T_l = 1648.0, 1723.0
        mask = np.ones_like(uniform_T)
        fig = plt.figure(figsize=(12, 8))
        plot_dual_thermal_phase_view(
            fig, uniform_T, mask, T_s, T_l, DX, DY, DZ, step=1, vmin=300, vmax=3500
        )
        plt.close(fig)
