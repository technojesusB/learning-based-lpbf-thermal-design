"""Tests for pipelines.dataset.prepare_trajectory."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from neural_pbf.physics.material import MaterialConfig
from neural_pbf.pipelines.dataset import TrajectoryPlan, prepare_trajectory


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        seed=42,
        materials="SS316L,IN718",
        nx=32, ny=16, nz=8,
        samples_per_run=10,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _zoo() -> dict[str, MaterialConfig]:
    return {
        "SS316L": MaterialConfig.ss316l_preset(),
        "IN718": MaterialConfig.in718_preset(),
    }


@pytest.mark.unit
class TestPrepareTrajectory:
    def test_returns_trajectory_plan(self):
        plan = prepare_trajectory(_make_args(), run_idx=0, material_zoo=_zoo())
        assert isinstance(plan, TrajectoryPlan)

    def test_deterministic_for_same_seed(self):
        args = _make_args()
        zoo = _zoo()
        p1 = prepare_trajectory(args, run_idx=0, material_zoo=zoo)
        p2 = prepare_trajectory(args, run_idx=0, material_zoo=zoo)
        np.testing.assert_array_equal(p1.path_points, p2.path_points)
        assert p1.run_seed == p2.run_seed
        assert p1.mat_key == p2.mat_key

    def test_different_run_indices_differ(self):
        args = _make_args()
        zoo = _zoo()
        p0 = prepare_trajectory(args, run_idx=0, material_zoo=zoo)
        p1 = prepare_trajectory(args, run_idx=1, material_zoo=zoo)
        assert p0.mat_key != p1.mat_key or not np.array_equal(
            p0.path_points, p1.path_points
        )

    def test_material_alternates_by_index(self):
        args = _make_args(materials="SS316L,IN718")
        zoo = _zoo()
        p0 = prepare_trajectory(args, run_idx=0, material_zoo=zoo)
        p1 = prepare_trajectory(args, run_idx=1, material_zoo=zoo)
        assert p0.mat_key == "SS316L"
        assert p1.mat_key == "IN718"

    def test_sample_indices_within_path_length(self):
        plan = prepare_trajectory(_make_args(), run_idx=0, material_zoo=_zoo())
        n = len(plan.path_points)
        assert all(0 <= i < n for i in plan.sample_indices)

    def test_path_points_shape(self):
        plan = prepare_trajectory(_make_args(), run_idx=0, material_zoo=_zoo())
        assert plan.path_points.ndim == 2
        assert plan.path_points.shape[1] == 2  # (x, y) columns

    def test_plan_is_frozen(self):
        plan = prepare_trajectory(_make_args(), run_idx=0, material_zoo=_zoo())
        with pytest.raises((AttributeError, TypeError)):
            plan.mat_key = "Ti64"  # type: ignore[misc]

    def test_beam_config_power_matches(self):
        plan = prepare_trajectory(_make_args(), run_idx=0, material_zoo=_zoo())
        assert plan.beam_cfg.power == pytest.approx(plan.power)
