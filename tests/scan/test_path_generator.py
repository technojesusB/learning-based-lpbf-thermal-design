"""Tests for generate_pulsed_path — written TDD-first against the implementation.

Coverage targets:
  - All three patterns: zigzag, raster, island
  - angle_deg != 0 rotation path
  - Domain filtering (boundary epsilon)
  - Return shape and dtype contracts
  - Error paths: unknown pattern, island_size <= 0
  - Edge cases: empty domain, very large point_distance
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_pbf.scan.path_generator import generate_pulsed_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(
    pattern: str = "zigzag",
    Lx: float = 1e-3,
    Ly: float = 1e-3,
    point_distance: float = 0.1e-3,
    hatch_spacing: float = 0.1e-3,
    angle_deg: float = 0.0,
    island_size: float = 2e-3,
) -> np.ndarray:
    """Thin wrapper so tests can override only the relevant argument."""
    return generate_pulsed_path(
        pattern=pattern,
        Lx=Lx,
        Ly=Ly,
        point_distance=point_distance,
        hatch_spacing=hatch_spacing,
        angle_deg=angle_deg,
        island_size=island_size,
    )


# ---------------------------------------------------------------------------
# Return type and shape contracts
# ---------------------------------------------------------------------------


def test_returns_ndarray():
    result = _call()
    assert isinstance(result, np.ndarray), "Expected np.ndarray"


def test_output_is_2d_with_two_columns():
    result = _call()
    assert result.ndim == 2, f"Expected 2-D array, got ndim={result.ndim}"
    assert result.shape[1] == 2, f"Expected shape (N, 2), got {result.shape}"


def test_output_dtype_is_float():
    result = _call()
    assert np.issubdtype(result.dtype, np.floating), (
        f"Expected floating dtype, got {result.dtype}"
    )


# ---------------------------------------------------------------------------
# Points stay inside domain [0, Lx] x [0, Ly]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["zigzag", "raster"])
def test_all_points_inside_domain(pattern: str):
    Lx, Ly = 2e-3, 1.5e-3
    result = _call(pattern=pattern, Lx=Lx, Ly=Ly, point_distance=0.1e-3, hatch_spacing=0.1e-3)
    eps = 1e-9
    assert result.shape[0] > 0, "Expected at least one point"
    assert np.all(result[:, 0] >= -eps), "x coordinate below domain minimum"
    assert np.all(result[:, 0] <= Lx + eps), "x coordinate above domain maximum"
    assert np.all(result[:, 1] >= -eps), "y coordinate below domain minimum"
    assert np.all(result[:, 1] <= Ly + eps), "y coordinate above domain maximum"


def test_island_points_inside_domain():
    Lx, Ly = 4e-3, 4e-3
    result = _call(
        pattern="island",
        Lx=Lx,
        Ly=Ly,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
        island_size=2e-3,
    )
    eps = 1e-9
    assert result.shape[0] > 0
    assert np.all(result[:, 0] >= -eps)
    assert np.all(result[:, 0] <= Lx + eps)
    assert np.all(result[:, 1] >= -eps)
    assert np.all(result[:, 1] <= Ly + eps)


# ---------------------------------------------------------------------------
# Zigzag alternates direction on odd rows
# ---------------------------------------------------------------------------


def test_zigzag_alternates_row_direction():
    """Row 0 should be ascending in x; row 1 should be descending in x."""
    Lx, Ly = 1e-3, 1e-3
    pd = 0.1e-3
    hs = 0.3e-3  # 4 distinct rows: y ≈ 0, 0.3, 0.6, 0.9 (after filter)

    result = _call(
        pattern="zigzag",
        Lx=Lx,
        Ly=Ly,
        point_distance=pd,
        hatch_spacing=hs,
    )
    assert result.shape[0] > 0

    # Group by distinct y values (rounded to avoid float noise)
    y_vals = np.unique(np.round(result[:, 1], decimals=9))
    assert len(y_vals) >= 2, "Need at least 2 rows to test alternation"

    row0_x = result[np.isclose(result[:, 1], y_vals[0], atol=1e-12), 0]
    row1_x = result[np.isclose(result[:, 1], y_vals[1], atol=1e-12), 0]

    assert len(row0_x) > 1 and len(row1_x) > 1, "Rows too short to test direction"

    row0_ascending = row0_x[-1] > row0_x[0]
    row1_ascending = row1_x[-1] > row1_x[0]

    # Row 0 and row 1 must have opposite scan directions
    assert row0_ascending != row1_ascending, (
        f"Zigzag expected opposite directions: row0 ascending={row0_ascending}, "
        f"row1 ascending={row1_ascending}"
    )


# ---------------------------------------------------------------------------
# Raster: all rows same direction
# ---------------------------------------------------------------------------


def test_raster_all_rows_same_direction():
    """All rows in a raster pattern must scan in the same x direction."""
    Lx, Ly = 1e-3, 1e-3
    pd = 0.1e-3
    hs = 0.3e-3

    result = _call(
        pattern="raster",
        Lx=Lx,
        Ly=Ly,
        point_distance=pd,
        hatch_spacing=hs,
    )
    assert result.shape[0] > 0

    y_vals = np.unique(np.round(result[:, 1], decimals=9))
    assert len(y_vals) >= 2

    directions = []
    for y in y_vals:
        row_x = result[np.isclose(result[:, 1], y, atol=1e-12), 0]
        if len(row_x) > 1:
            directions.append(row_x[-1] > row_x[0])

    assert len(set(directions)) == 1, (
        f"Raster should have uniform scan direction, got: {directions}"
    )


# ---------------------------------------------------------------------------
# Raster vs Zigzag: must produce different orderings
# ---------------------------------------------------------------------------


def test_zigzag_differs_from_raster():
    """Zigzag and raster paths differ in point order for multi-row grids."""
    kwargs = dict(Lx=1e-3, Ly=1e-3, point_distance=0.1e-3, hatch_spacing=0.3e-3)
    zz = generate_pulsed_path("zigzag", **kwargs)
    ra = generate_pulsed_path("raster", **kwargs)

    # Same number of points (same grid)
    assert zz.shape == ra.shape, "Zigzag and raster should have the same point count"
    # But at least one row must differ in x-order
    assert not np.allclose(zz, ra), "Zigzag and raster produced identical paths"


# ---------------------------------------------------------------------------
# Island pattern: tiles the domain
# ---------------------------------------------------------------------------


def test_island_returns_points():
    result = _call(
        pattern="island",
        Lx=4e-3,
        Ly=4e-3,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
        island_size=2e-3,
    )
    assert result.shape[0] > 0


def test_island_subdivides_into_tiles():
    """Island output should have roughly the same density as zigzag on the same domain."""
    Lx, Ly = 4e-3, 4e-3
    pd = 0.1e-3
    hs = 0.1e-3

    island_pts = _call(
        pattern="island",
        Lx=Lx, Ly=Ly,
        point_distance=pd,
        hatch_spacing=hs,
        island_size=2e-3,
    )
    zigzag_pts = _call(
        pattern="zigzag",
        Lx=Lx, Ly=Ly,
        point_distance=pd,
        hatch_spacing=hs,
    )

    # Both should produce a comparable number of points (within 10%)
    ratio = island_pts.shape[0] / zigzag_pts.shape[0]
    assert 0.9 <= ratio <= 1.1, (
        f"Island point count ({island_pts.shape[0]}) deviates too much from "
        f"zigzag ({zigzag_pts.shape[0]}), ratio={ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Angle rotation
# ---------------------------------------------------------------------------


def test_angle_zero_points_stay_aligned():
    """With angle_deg=0, x-coords within each row should be monotone multiples of pd."""
    result = _call(angle_deg=0.0, point_distance=0.1e-3, hatch_spacing=0.2e-3)
    y_vals = np.unique(np.round(result[:, 1], decimals=9))
    for y in y_vals:
        row_x = np.sort(result[np.isclose(result[:, 1], y, atol=1e-12), 0])
        if len(row_x) > 1:
            diffs = np.diff(row_x)
            # All diffs should be close to point_distance
            assert np.allclose(diffs, 0.1e-3, rtol=1e-6), (
                f"Row y={y:.3e}: diffs {diffs} not close to point_distance=0.1e-3"
            )


def test_angle_rotated_path_has_different_coords():
    """Rotating 45 degrees should produce a different set of (x, y) coordinates."""
    base = _call(angle_deg=0.0, Lx=2e-3, Ly=2e-3, point_distance=0.1e-3, hatch_spacing=0.2e-3)
    rotated = _call(angle_deg=45.0, Lx=2e-3, Ly=2e-3, point_distance=0.1e-3, hatch_spacing=0.2e-3)

    # Both should still produce points
    assert base.shape[0] > 0
    assert rotated.shape[0] > 0

    # The point sets should differ (rotation changes coordinates)
    assert not np.allclose(
        np.sort(base[:, 0])[:min(len(base), len(rotated))],
        np.sort(rotated[:, 0])[:min(len(base), len(rotated))],
        atol=1e-9,
    ), "Rotating 45 deg should change x-coordinates"


def test_angle_90_produces_points_inside_domain():
    """90-degree rotation should still yield all points inside the domain."""
    Lx, Ly = 2e-3, 2e-3
    result = _call(
        pattern="zigzag",
        Lx=Lx,
        Ly=Ly,
        angle_deg=90.0,
        point_distance=0.1e-3,
        hatch_spacing=0.2e-3,
    )
    eps = 1e-9
    assert result.shape[0] > 0
    assert np.all(result[:, 0] >= -eps)
    assert np.all(result[:, 0] <= Lx + eps)
    assert np.all(result[:, 1] >= -eps)
    assert np.all(result[:, 1] <= Ly + eps)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_pattern_raises_value_error():
    with pytest.raises(ValueError, match="Unknown pattern"):
        generate_pulsed_path(
            pattern="spiral",  # type: ignore[arg-type]
            Lx=1e-3,
            Ly=1e-3,
            point_distance=0.1e-3,
            hatch_spacing=0.1e-3,
        )


def test_island_size_zero_raises_value_error():
    with pytest.raises(ValueError):
        generate_pulsed_path(
            pattern="island",
            Lx=1e-3,
            Ly=1e-3,
            point_distance=0.1e-3,
            hatch_spacing=0.1e-3,
            island_size=0.0,
        )


def test_island_size_negative_raises_value_error():
    with pytest.raises(ValueError):
        generate_pulsed_path(
            pattern="island",
            Lx=1e-3,
            Ly=1e-3,
            point_distance=0.1e-3,
            hatch_spacing=0.1e-3,
            island_size=-1e-3,
        )


# ---------------------------------------------------------------------------
# Edge cases: empty output
# ---------------------------------------------------------------------------


def test_empty_output_when_point_distance_larger_than_domain():
    """If point_distance > Lx no points are generated after rotation+filter."""
    result = generate_pulsed_path(
        pattern="zigzag",
        Lx=1e-6,
        Ly=1e-6,
        point_distance=1e-3,  # Much larger than domain
        hatch_spacing=0.5e-3,
    )
    # The generated "path" in the expanded rotated frame will have a single
    # candidate point per row; after filtering it might be empty or very sparse.
    # The important contract: shape is (N, 2) — never a 1-D array.
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_hatch_spacing_larger_than_domain_returns_single_row_or_empty():
    """When hatch_spacing > Ly, only one or zero hatch lines exist."""
    result = generate_pulsed_path(
        pattern="raster",
        Lx=1e-3,
        Ly=1e-4,
        point_distance=0.1e-3,
        hatch_spacing=1e-3,  # Much larger than Ly
    )
    assert result.ndim == 2
    assert result.shape[1] == 2


# ---------------------------------------------------------------------------
# Point count sanity checks
# ---------------------------------------------------------------------------


def test_point_count_scales_with_domain_area():
    """Doubling both Lx and Ly should roughly quadruple the point count."""
    small = _call(Lx=1e-3, Ly=1e-3, point_distance=0.1e-3, hatch_spacing=0.1e-3)
    large = _call(Lx=2e-3, Ly=2e-3, point_distance=0.1e-3, hatch_spacing=0.1e-3)

    ratio = large.shape[0] / small.shape[0]
    # Expect ~4x; allow generous tolerance for boundary effects
    assert 3.0 <= ratio <= 5.0, (
        f"Expected ~4x more points in 2x domain, got ratio={ratio:.2f} "
        f"(small={small.shape[0]}, large={large.shape[0]})"
    )


def test_finer_point_distance_gives_more_points():
    """Halving point_distance should approximately double the point count per row."""
    coarse = _call(point_distance=0.2e-3, hatch_spacing=0.1e-3)
    fine = _call(point_distance=0.1e-3, hatch_spacing=0.1e-3)

    assert fine.shape[0] > coarse.shape[0], (
        f"Finer point_distance should produce more points: "
        f"fine={fine.shape[0]}, coarse={coarse.shape[0]}"
    )


def test_finer_hatch_spacing_gives_more_rows():
    """Halving hatch_spacing should approximately double the row count."""
    coarse = _call(point_distance=0.1e-3, hatch_spacing=0.2e-3)
    fine = _call(point_distance=0.1e-3, hatch_spacing=0.1e-3)

    assert fine.shape[0] > coarse.shape[0], (
        f"Finer hatch_spacing should produce more points: "
        f"fine={fine.shape[0]}, coarse={coarse.shape[0]}"
    )


# ---------------------------------------------------------------------------
# No NaN / Inf in output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["zigzag", "raster", "island"])
def test_no_nan_in_output(pattern: str):
    result = _call(
        pattern=pattern,
        Lx=2e-3,
        Ly=2e-3,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
        island_size=1e-3,
    )
    assert not np.any(np.isnan(result)), f"NaN detected in {pattern} output"
    assert not np.any(np.isinf(result)), f"Inf detected in {pattern} output"


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 67.5, 90.0])
def test_no_nan_for_various_angles(angle_deg: float):
    result = _call(
        pattern="zigzag",
        Lx=2e-3,
        Ly=2e-3,
        angle_deg=angle_deg,
        point_distance=0.1e-3,
        hatch_spacing=0.2e-3,
    )
    assert not np.any(np.isnan(result)), f"NaN at angle_deg={angle_deg}"
    assert not np.any(np.isinf(result)), f"Inf at angle_deg={angle_deg}"


# ---------------------------------------------------------------------------
# Island size edge case: island_size > domain
# ---------------------------------------------------------------------------


def test_island_size_larger_than_domain_degrades_gracefully():
    """When island_size > Lx and Ly, the island pattern behaves like a single-tile zigzag."""
    result = _call(
        pattern="island",
        Lx=1e-3,
        Ly=1e-3,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
        island_size=10e-3,  # Much larger than the domain
    )
    assert result.ndim == 2
    assert result.shape[1] == 2
    # Points must still be inside the domain
    eps = 1e-9
    if result.shape[0] > 0:
        assert np.all(result[:, 0] >= -eps)
        assert np.all(result[:, 0] <= 1e-3 + eps)
        assert np.all(result[:, 1] >= -eps)
        assert np.all(result[:, 1] <= 1e-3 + eps)


# ---------------------------------------------------------------------------
# Branch coverage: inner _generate_block edge cases
# ---------------------------------------------------------------------------


def test_zero_domain_raster_returns_empty_array():
    """Lx=Ly=0 makes x_min==x_max so no hatch lines are generated anywhere.

    This exercises the inner `return np.empty((0, 2))` branch inside _generate_block
    (when block_points stays empty) and the outer `if not points` early-return path.
    """
    result = generate_pulsed_path(
        pattern="raster",
        Lx=0.0,
        Ly=0.0,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
    )
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert result.shape[0] == 0, "Expected empty array for zero-sized domain"


def test_zero_domain_island_returns_empty_array():
    """Lx=Ly=0 causes all island blocks to be empty; `points` list stays empty.

    This exercises the top-level `if not points: return np.empty((0, 2))` branch
    specifically for the island pattern.
    """
    result = generate_pulsed_path(
        pattern="island",
        Lx=0.0,
        Ly=0.0,
        point_distance=0.1e-3,
        hatch_spacing=0.1e-3,
        island_size=1e-3,
    )
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert result.shape[0] == 0, "Expected empty array for zero-sized island domain"
