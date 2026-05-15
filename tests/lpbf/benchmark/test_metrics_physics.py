"""Tests for benchmark.metrics_physics — compute_physics_metrics."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.benchmark.metrics_physics import compute_physics_metrics
from neural_pbf.eval.benchmark.constants import T_LIQUIDUS, T_REF, T_AMBIENT


def _make_norm_vol(fill_value: float, size: int = 8) -> torch.Tensor:
    """Return a (size, size, size) normalised volume filled with a constant."""
    return torch.full((size, size, size), fill_value)


@pytest.mark.unit
def test_compute_physics_metrics_returns_dict_with_required_keys():
    pred = _make_norm_vol(0.7)
    tgt = _make_norm_vol(0.7)
    result = compute_physics_metrics(pred, tgt)
    required = {"IoU", "Depth_GT", "Depth_Pred", "T_max_Error", "Offset_vox"}
    assert required.issubset(set(result.keys()))


@pytest.mark.unit
def test_compute_physics_metrics_identical_volumes_iou_one():
    """Identical volumes → IoU == 1.0 (assuming both above liquidus)."""
    # At norm=0.7: T_phys = 0.7 * 2000 + 300 = 1700 K > 1600 K → all liquid
    pred = _make_norm_vol(0.7)
    tgt = _make_norm_vol(0.7)
    result = compute_physics_metrics(pred, tgt)
    assert result["IoU"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_physics_metrics_no_overlap_iou_zero():
    """Pred all below liquidus, GT all above → IoU == 0."""
    # 0.65 * 2000 + 300 = 1600 K  (exactly at threshold — below)
    pred = _make_norm_vol(0.6)   # T = 1500 K < 1600 K → not liquid
    tgt = _make_norm_vol(0.7)    # T = 1700 K > 1600 K → liquid
    result = compute_physics_metrics(pred, tgt)
    assert result["IoU"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_physics_metrics_t_max_error_zero_for_identical():
    """Identical volumes → T_max_Error == 0."""
    vol = _make_norm_vol(0.5)
    result = compute_physics_metrics(vol, vol)
    assert result["T_max_Error"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_compute_physics_metrics_t_max_error_in_kelvin():
    """T_max_Error should be in physical Kelvin units, not normalised."""
    pred = _make_norm_vol(0.8)  # T_max = 0.8 * 2000 + 300 = 1900 K
    tgt = _make_norm_vol(0.9)   # T_max = 0.9 * 2000 + 300 = 2100 K
    result = compute_physics_metrics(pred, tgt)
    # |1900 - 2100| = 200 K
    assert result["T_max_Error"] == pytest.approx(200.0, abs=1.0)


@pytest.mark.unit
def test_compute_physics_metrics_uses_1600k_liquidus():
    """Values between the old 0.6-normalised threshold and 1600 K are treated as solid.

    Before the refactor, the inner loop used _T_LIQUIDUS=0.6 on already-physical
    temperatures, which was a bug.  This test catches any regression back to that.

    norm=0.62 → T_phys = 0.62 * 2000 + 300 = 1540 K
    - Old (buggy) path: 1540 > 0.6 → "melt"
    - Correct path:     1540 < 1600 → "solid"
    When pred=solid vs tgt=melt (norm=0.70 → 1700 K), IoU must be 0.
    """
    below_1600 = _make_norm_vol(0.62)   # T = 1540 K — solid under 1600 K threshold
    above_1600 = _make_norm_vol(0.70)   # T = 1700 K — melt
    result = compute_physics_metrics(below_1600, above_1600)
    assert result["IoU"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_physics_metrics_offset_zero_identical_hotspot():
    """Identical volumes → offset_vox == 0."""
    vol = _make_norm_vol(0.0)
    # Put a hotspot in the same position
    vol[4, 4, 4] = 1.0
    result = compute_physics_metrics(vol, vol)
    assert result["Offset_vox"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_physics_metrics_accepts_5d_input():
    """Accepts (1, 1, D, H, W) batched inputs as well as (D, H, W)."""
    vol_3d = _make_norm_vol(0.7)
    vol_5d = vol_3d.reshape(1, 1, *vol_3d.shape)
    r3 = compute_physics_metrics(vol_3d, vol_3d)
    r5 = compute_physics_metrics(vol_5d, vol_5d)
    assert r3["IoU"] == pytest.approx(r5["IoU"])
    assert r3["T_max_Error"] == pytest.approx(r5["T_max_Error"])
