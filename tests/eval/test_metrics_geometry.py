"""Unit tests for geometry metrics with analytic blobs."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.metrics.geometry import iou_melt_volumes, melt_pool_extent

_T_LIQ = 1673.0


@pytest.mark.unit
def test_melt_pool_extent_no_molten_returns_zeros():
    T = torch.ones(1, 1, 8, 8) * 300.0
    result = melt_pool_extent(T, _T_LIQ)
    assert result == {"W": 0.0, "L": 0.0, "D": 0.0}


@pytest.mark.unit
def test_melt_pool_extent_single_voxel():
    T = torch.ones(1, 1, 8, 8) * 300.0
    T[0, 0, 3, 4] = 2000.0  # one molten voxel
    result = melt_pool_extent(T, _T_LIQ)
    assert result["W"] == 1.0  # 1 row in Y
    assert result["L"] == 1.0  # 1 col in X
    assert result["D"] == 1.0  # 2D → always 1


@pytest.mark.unit
def test_melt_pool_extent_row_blob():
    T = torch.ones(1, 1, 8, 8) * 300.0
    T[0, 0, 3, 2:5] = 2000.0  # 3 voxels in a row
    result = melt_pool_extent(T, _T_LIQ)
    assert result["L"] == pytest.approx(3.0)  # X extent = 3
    assert result["W"] == pytest.approx(1.0)  # Y extent = 1 (one row)


@pytest.mark.unit
def test_melt_pool_extent_3d(sim_cfg_3d):
    T = torch.ones(1, 1, sim_cfg_3d.Nz, sim_cfg_3d.Ny, sim_cfg_3d.Nx) * 300.0
    # A 2×3×4 molten block
    T[0, 0, 0:2, 1:4, 2:6] = 2000.0
    result = melt_pool_extent(T, _T_LIQ)
    assert result["D"] == pytest.approx(2.0)  # Z
    assert result["W"] == pytest.approx(3.0)  # Y
    assert result["L"] == pytest.approx(4.0)  # X


@pytest.mark.unit
def test_iou_both_empty_returns_one():
    T = torch.ones(1, 1, 4, 4) * 300.0
    assert iou_melt_volumes(T, T, _T_LIQ) == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_identical_molten_returns_one():
    T = torch.ones(1, 1, 4, 4) * 300.0
    T[0, 0, 1:3, 1:3] = 2000.0
    assert iou_melt_volumes(T, T, _T_LIQ) == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_no_overlap_returns_zero():
    T_gt = torch.ones(1, 1, 4, 4) * 300.0
    T_gt[0, 0, 0, 0] = 2000.0
    T_pred = torch.ones(1, 1, 4, 4) * 300.0
    T_pred[0, 0, 3, 3] = 2000.0  # disjoint voxel
    assert iou_melt_volumes(T_pred, T_gt, _T_LIQ) == pytest.approx(0.0)


@pytest.mark.unit
def test_iou_partial_overlap():
    # 2x2 GT blob, 2x2 Pred blob shifted by 1 — 1x2 overlap
    T_gt = torch.ones(1, 1, 4, 4) * 300.0
    T_pred = torch.ones(1, 1, 4, 4) * 300.0
    T_gt[0, 0, 0:2, 0:2] = 2000.0   # 4 voxels
    T_pred[0, 0, 0:2, 1:3] = 2000.0  # 4 voxels, overlap at [0:2, 1:2] = 2
    iou = iou_melt_volumes(T_pred, T_gt, _T_LIQ)
    # intersection=2, union=6 → IoU = 1/3
    assert iou == pytest.approx(2 / 6, rel=1e-5)
