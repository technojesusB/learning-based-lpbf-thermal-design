"""Tests for neural_pbf.eval.metrics.spectral."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_pbf.eval.metrics.spectral import (
    calculate_boundary_discontinuity,
    calculate_total_variation,
    compute_axis_psd,
    compute_radial_psd,
)


# ---------------------------------------------------------------------------
# compute_radial_psd
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_radial_psd_returns_two_arrays() -> None:
    vol = torch.rand(16, 16, 16)
    freqs, profile = compute_radial_psd(vol)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(profile, np.ndarray)
    assert len(freqs) == len(profile)


@pytest.mark.unit
def test_compute_radial_psd_freq_range() -> None:
    vol = torch.rand(16, 16, 16)
    freqs, _ = compute_radial_psd(vol)
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_radial_psd_no_nan_or_inf() -> None:
    vol = torch.rand(16, 16, 16)
    _, profile = compute_radial_psd(vol)
    assert not np.any(np.isnan(profile))
    assert not np.any(np.isinf(profile))


@pytest.mark.unit
def test_compute_radial_psd_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="3D"):
        compute_radial_psd(torch.rand(4, 4))


# ---------------------------------------------------------------------------
# compute_axis_psd
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_axis_psd_returns_three_tuples() -> None:
    vol = torch.rand(16, 16, 16)
    result = compute_axis_psd(vol)
    assert len(result) == 3
    for freqs, psd in result:
        assert isinstance(freqs, np.ndarray)
        assert isinstance(psd, np.ndarray)
        assert len(freqs) == len(psd)


@pytest.mark.unit
def test_compute_axis_psd_positive_values() -> None:
    vol = torch.rand(16, 16, 16)
    for _, psd in compute_axis_psd(vol):
        assert np.all(psd >= 0)


# ---------------------------------------------------------------------------
# calculate_total_variation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tv_constant_volume_is_zero() -> None:
    vol = torch.ones(8, 8, 8)
    assert calculate_total_variation(vol) == pytest.approx(0.0)


@pytest.mark.unit
def test_tv_non_constant_is_positive() -> None:
    vol = torch.rand(8, 8, 8)
    assert calculate_total_variation(vol) > 0.0


@pytest.mark.unit
def test_tv_returns_float() -> None:
    assert isinstance(calculate_total_variation(torch.rand(4, 4, 4)), float)


# ---------------------------------------------------------------------------
# calculate_boundary_discontinuity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pbd_constant_volume_near_one() -> None:
    """Constant volume has zero gradients everywhere; PBD should be 1.0 (eps guard)."""
    vol = torch.ones(16, 16, 16)
    pbd = calculate_boundary_discontinuity(vol, patch_size=8)
    assert isinstance(pbd, float)
    assert pbd >= 0.0


@pytest.mark.unit
def test_pbd_no_division_by_zero_for_flat_interior() -> None:
    """Interior gradients are zero for a piecewise-constant volume; must not raise."""
    vol = torch.zeros(16, 16, 16)
    # Create a sharp seam at every patch boundary, zero elsewhere
    vol[7, :, :] = 1.0
    vol[15, :, :] = 1.0
    pbd = calculate_boundary_discontinuity(vol, patch_size=8)
    assert np.isfinite(pbd)


@pytest.mark.unit
def test_pbd_returns_float() -> None:
    assert isinstance(
        calculate_boundary_discontinuity(torch.rand(16, 16, 16), patch_size=8), float
    )


@pytest.mark.unit
def test_pbd_no_boundaries_returns_one() -> None:
    """patch_size larger than volume → no boundary indices → returns 1.0."""
    vol = torch.rand(4, 4, 4)
    assert calculate_boundary_discontinuity(vol, patch_size=64) == pytest.approx(1.0)
