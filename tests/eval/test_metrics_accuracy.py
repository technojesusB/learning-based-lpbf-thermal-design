"""Unit tests for accuracy metrics with analytic expected values."""
from __future__ import annotations

import math

import pytest
import torch

from neural_pbf.eval.metrics.accuracy import (
    mae_global,
    mae_melt_pool,
    max_error,
    per_step_mae,
)


@pytest.mark.unit
def test_mae_global_identical_tensors():
    T = torch.ones(1, 1, 4, 4) * 300.0
    assert mae_global(T, T) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_mae_global_uniform_offset():
    T_gt = torch.ones(1, 1, 4, 4) * 300.0
    T_pred = T_gt + 10.0
    assert mae_global(T_pred, T_gt) == pytest.approx(10.0, rel=1e-5)


@pytest.mark.unit
def test_mae_global_mixed_errors():
    T_gt = torch.zeros(1, 1, 2, 2)
    T_pred = torch.tensor([[[[1.0, -1.0], [3.0, -3.0]]]])
    # Abs errors: 1, 1, 3, 3 → mean = 2.0
    assert mae_global(T_pred, T_gt) == pytest.approx(2.0, rel=1e-5)


@pytest.mark.unit
def test_mae_global_nan_in_pred_treated_as_zero():
    T_gt = torch.zeros(1, 1, 2, 2)
    T_pred = torch.tensor([[[[float("nan"), 0.0], [0.0, 0.0]]]])
    # nan diff → treated as 0
    result = mae_global(T_pred, T_gt)
    assert math.isfinite(result)
    assert result == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_mae_melt_pool_no_molten_returns_nan():
    T_gt = torch.ones(1, 1, 4, 4) * 300.0  # all below solidus
    T_pred = T_gt.clone()
    result = mae_melt_pool(T_pred, T_gt, T_solidus=1653.0)
    assert math.isnan(result)


@pytest.mark.unit
def test_mae_melt_pool_only_molten_region():
    # GT has one molten voxel at temperature 2000 K
    T_gt = torch.ones(1, 1, 4, 4) * 300.0
    T_gt[0, 0, 0, 0] = 2000.0
    T_pred = T_gt.clone()
    T_pred[0, 0, 0, 0] = 2100.0  # 100 K error
    T_pred[0, 0, 1, 1] = 400.0   # cold voxel error — should be ignored

    result = mae_melt_pool(T_pred, T_gt, T_solidus=1653.0)
    assert result == pytest.approx(100.0, rel=1e-5)


@pytest.mark.unit
def test_max_error_identical():
    T = torch.ones(1, 1, 4, 4) * 500.0
    assert max_error(T, T) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_max_error_known_value():
    T_gt = torch.zeros(1, 1, 4, 4)
    T_pred = torch.zeros(1, 1, 4, 4)
    T_pred[0, 0, 2, 3] = 500.0
    assert max_error(T_pred, T_gt) == pytest.approx(500.0, rel=1e-5)


@pytest.mark.unit
def test_per_step_mae_length_matches():
    Ts = [torch.ones(1, 1, 4, 4) * float(i) for i in range(5)]
    gts = [torch.zeros(1, 1, 4, 4)] * 5
    result = per_step_mae(Ts, gts)
    assert len(result) == 5
    for i, val in enumerate(result):
        assert val == pytest.approx(float(i), rel=1e-5)
