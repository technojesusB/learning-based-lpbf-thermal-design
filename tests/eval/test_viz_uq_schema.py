"""THEME.uq reuses the existing look and leaves the other themes untouched."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_pbf.eval.viz import uq as viz
from neural_pbf.schemas.viz import THEME, PlotConfig, ThemeRegistry, UQPlotConfig

METHODS = {"absolute", "normalized", "constant_single_draw", "raw_ensemble"}


@pytest.mark.unit
def test_theme_registry_exposes_uq_plot_config():
    assert isinstance(THEME.uq, UQPlotConfig)
    assert isinstance(THEME.uq, PlotConfig)
    assert isinstance(ThemeRegistry().uq, UQPlotConfig)


@pytest.mark.unit
def test_uq_theme_inherits_base_output_and_background():
    assert (THEME.uq.dpi, THEME.uq.bg_figure, THEME.uq.bg_axis) == (300, "#0a0a0a", "#0a0a0a")


@pytest.mark.unit
def test_uq_fonts_and_grid_match_physical_dashboard():
    for field in ("font_size_title", "font_size_label", "font_size_tick", "font_size_legend"):
        assert getattr(THEME.uq, field) == getattr(THEME.physical, field)
    assert THEME.uq.grid == THEME.physical.grid


@pytest.mark.unit
def test_uq_colors_introduce_no_new_look():
    existing = set(THEME.base.color_palette) | {
        THEME.base.color_train, THEME.base.color_val, THEME.base.color_fm, THEME.base.color_pde,
        THEME.base.color_lambda, THEME.base.color_gt, THEME.base.color_secondary,
    }  # fmt: skip
    assert set(THEME.uq.method_colors) == METHODS
    assert set(THEME.uq.method_colors.values()) <= existing
    assert len(set(THEME.uq.method_colors.values())) == len(METHODS)
    assert THEME.uq.color_ideal in existing
    assert THEME.uq.color_points in existing
    assert THEME.uq.color_marker_edge in existing


@pytest.mark.unit
def test_every_method_has_a_distinct_marker():
    assert set(THEME.uq.method_markers) == METHODS
    assert len(set(THEME.uq.method_markers.values())) == len(METHODS)


@pytest.mark.unit
def test_uq_module_takes_marker_and_edge_style_from_the_theme():
    source = Path(viz.__file__).read_text()
    assert 'edgecolor="w"' not in source and 'fmt="o"' not in source and 'marker="o"' not in source
    assert THEME.uq.marker_default == "o"
    assert THEME.uq.coverage_ylim == (-0.05, 1.05)


@pytest.mark.unit
def test_ideal_line_matches_physical_dashboard():
    assert (THEME.uq.color_ideal, THEME.uq.linestyle_ideal, THEME.uq.line_width_ideal) == ("#ffffff", "--", 1.5)


@pytest.mark.unit
def test_existing_theme_values_are_unchanged():
    assert THEME.physical.figsize == (25, 15)
    assert THEME.system.figsize == (22, 6)
    assert THEME.spectral.figsize == (16.7, 15)
    assert THEME.spatial.cmap == "magma"
