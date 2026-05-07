"""Rollout engine integration tests."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.adapters.triton_adapter import TritonAdapter
from neural_pbf.eval.rollout.autoregressive import ar_summary, cumulative_mae, has_diverged
from neural_pbf.eval.rollout.engine import RolloutEngine, RolloutResult


@pytest.mark.unit
def test_rollout_identity_zero_error(constant_trajectory):
    """IdentityAdapter returns GT unchanged — with constant T, MAE must be 0."""
    engine = RolloutEngine()
    adapter = IdentityAdapter()
    result = engine.run(adapter, constant_trajectory, mode="one_step")
    assert isinstance(result, RolloutResult)
    for m in result.per_step_metrics:
        assert m["mae_global"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_rollout_identity_autoregressive(constant_trajectory):
    engine = RolloutEngine()
    result = engine.run(IdentityAdapter(), constant_trajectory, mode="autoregressive")
    assert len(result.pred_T) == len(constant_trajectory) - 1
    for m in result.per_step_metrics:
        assert m["mae_global"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_rollout_latencies_non_negative(trajectory):
    engine = RolloutEngine()
    result = engine.run(IdentityAdapter(), trajectory, mode="one_step")
    for lat in result.latencies_s:
        assert lat >= 0.0


@pytest.mark.unit
def test_rollout_result_length(trajectory):
    engine = RolloutEngine()
    result = engine.run(IdentityAdapter(), trajectory, mode="one_step")
    n = len(trajectory) - 1
    assert len(result.pred_T) == n
    assert len(result.per_step_metrics) == n
    assert len(result.latencies_s) == n


@pytest.mark.unit
def test_rollout_raises_on_short_trajectory(sim_cfg, mat_cfg):
    from neural_pbf.eval.data.snapshot import Snapshot
    from neural_pbf.eval.data.trajectory import Trajectory

    T = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx) * 300.0
    single = Trajectory(
        snapshots=[Snapshot(T=T, t=0.0, dt=1e-5)],
        sim_cfg=sim_cfg,
        mat_cfg=mat_cfg,
    )
    with pytest.raises(ValueError, match="2 snapshots"):
        RolloutEngine().run(IdentityAdapter(), single)


@pytest.mark.unit
def test_rollout_raises_on_invalid_mode(trajectory):
    with pytest.raises(ValueError, match="mode"):
        RolloutEngine().run(IdentityAdapter(), trajectory, mode="bad_mode")


@pytest.mark.unit
def test_divergence_guard_aborts_early(sim_cfg, mat_cfg):
    """A stepper that produces temperatures > 5000 K should trigger the guard."""
    from neural_pbf.eval.data.snapshot import Snapshot
    from neural_pbf.eval.data.trajectory import Trajectory
    from neural_pbf.core.state import SimulationState
    from typing import Any

    class HotStepper:
        name = "hot"

        def step(self, state, Q_ext, dt, conditioning=None):
            s = state.clone()
            s.T = torch.full_like(s.T, 9999.0)  # above divergence threshold
            return s

    T = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx) * 300.0
    snaps = [Snapshot(T=T.clone(), t=i * 1e-5, dt=1e-5) for i in range(5)]
    traj = Trajectory(snapshots=snaps, sim_cfg=sim_cfg, mat_cfg=mat_cfg)

    result = RolloutEngine().run(HotStepper(), traj)
    assert result.diverged_at_step == 0
    assert len(result.pred_T) == 0  # nothing collected before divergence


@pytest.mark.unit
def test_ar_summary_with_identity(constant_trajectory):
    result = RolloutEngine().run(IdentityAdapter(), constant_trajectory)
    summary = ar_summary(result)
    assert summary["mean_mae"] == pytest.approx(0.0, abs=1e-5)
    assert summary["diverged"] == 0.0


@pytest.mark.unit
def test_cumulative_mae_is_zero_with_constant_traj(constant_trajectory):
    """With identity adapter and constant T, cumulative MAE stays at 0."""
    result = RolloutEngine().run(IdentityAdapter(), constant_trajectory)
    cmae = cumulative_mae(result)
    assert all(v == pytest.approx(0.0, abs=1e-5) for v in cmae)


@pytest.mark.unit
def test_has_diverged_false_for_identity(constant_trajectory):
    result = RolloutEngine().run(IdentityAdapter(), constant_trajectory)
    assert not has_diverged(result)


@pytest.mark.integration
def test_rollout_triton_pytorch_path(sim_cfg, mat_cfg, trajectory):
    """TritonAdapter (PyTorch path) should produce finite, physically reasonable T."""
    engine = RolloutEngine()
    adapter = TritonAdapter(sim_cfg, mat_cfg, use_triton=False)
    result = engine.run(adapter, trajectory, mode="one_step")
    assert len(result.pred_T) > 0
    for T_pred in result.pred_T:
        assert torch.isfinite(T_pred).all()
        assert T_pred.min().item() >= 0.0
