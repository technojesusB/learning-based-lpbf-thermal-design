"""Tests for neural_pbf.training.loops — epoch runners, train loop, test phase, log_val_image_rope."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neural_pbf.training.loops import (
    TrainHistory,
    log_val_image_rope,
    run_test_phase,
    run_train_epoch,
    run_train_loop,
    run_val_epoch,
)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_batches(n: int = 3) -> list[dict]:
    return [{"idx": i} for i in range(n)]


# ---------------------------------------------------------------------------
# run_train_epoch
# ---------------------------------------------------------------------------


class TestRunTrainEpoch:
    def test_calls_batch_fn_once_per_batch(self):
        calls: list = []
        batch_fn = lambda b: (calls.append(b), 1.0)[1]  # noqa: E731
        run_train_epoch(batch_fn, _make_batches(4), epoch=0, tqdm_disable=True)
        assert len(calls) == 4

    def test_returns_average_loss(self):
        losses = iter([1.0, 3.0, 2.0])
        batch_fn = lambda b: next(losses)  # noqa: E731
        avg = run_train_epoch(batch_fn, _make_batches(3), epoch=0, tqdm_disable=True)
        assert avg == pytest.approx(2.0)

    def test_calls_profiler_step_per_batch(self):
        profiler = MagicMock()
        run_train_epoch(lambda b: 0.0, _make_batches(3), epoch=0,
                        profiler=profiler, tqdm_disable=True)
        assert profiler.step.call_count == 3

    def test_no_profiler_does_not_raise(self):
        run_train_epoch(lambda b: 0.0, _make_batches(2), epoch=1,
                        profiler=None, tqdm_disable=True)

    def test_empty_loader_returns_zero(self):
        avg = run_train_epoch(lambda b: 1.0, [], epoch=0, tqdm_disable=True)
        assert avg == pytest.approx(0.0)

    def test_custom_tqdm_desc_does_not_break(self):
        avg = run_train_epoch(lambda b: 5.0, _make_batches(2), epoch=0,
                              tqdm_desc="Custom", tqdm_disable=True)
        assert avg == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# run_val_epoch
# ---------------------------------------------------------------------------


class TestRunValEpoch:
    def test_returns_average_of_step_fn_outputs(self):
        outputs = iter([2.0, 4.0, 6.0])
        avg = run_val_epoch(lambda b: next(outputs), _make_batches(3))
        assert avg == pytest.approx(4.0)

    def test_runs_under_no_grad(self):
        grad_flags: list[bool] = []
        def step_fn(batch):
            grad_flags.append(torch.is_grad_enabled())
            return 0.0
        run_val_epoch(step_fn, _make_batches(2))
        assert all(not flag for flag in grad_flags)

    def test_grad_is_re_enabled_after_return(self):
        assert torch.is_grad_enabled()
        run_val_epoch(lambda b: 0.0, _make_batches(1))
        assert torch.is_grad_enabled()

    def test_empty_loader_returns_zero(self):
        avg = run_val_epoch(lambda b: 99.0, [])
        assert avg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_train_loop
# ---------------------------------------------------------------------------


def _std_callbacks(val_loss_seq: list[float] | None = None):
    """Return (train_fn, val_fn, ckpt_fn, snap_fn, records)."""
    records: dict = {
        "train": [], "val": [], "ckpt": [], "snap": []
    }
    val_iter = iter(val_loss_seq or [0.5] * 100)

    def train_fn(epoch: int) -> float:
        records["train"].append(epoch)
        return 1.0

    def val_fn(epoch: int) -> float:
        records["val"].append(epoch)
        return next(val_iter)

    def ckpt_fn(epoch: int, val_loss: float) -> None:
        records["ckpt"].append((epoch, val_loss))

    def snap_fn(epoch: int):
        records["snap"].append(epoch)
        return None

    return train_fn, val_fn, ckpt_fn, snap_fn, records


class TestRunTrainLoop:
    def test_calls_train_fn_every_epoch(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=4, tqdm_disable=True)
        assert records["train"] == [0, 1, 2, 3]

    def test_val_fn_called_on_val_every_epochs(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                       epochs=6, val_every=2, tqdm_disable=True)
        assert records["val"] == [0, 2, 4]

    def test_val_fn_called_every_epoch_when_val_every_1(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                       epochs=3, val_every=1, tqdm_disable=True)
        assert records["val"] == [0, 1, 2]

    def test_accumulates_train_and_val_losses_in_history(self):
        train_fn, val_fn, ckpt_fn, snap_fn, _ = _std_callbacks([0.4, 0.3, 0.2])
        history = run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                                 epochs=3, tqdm_disable=True)
        assert len(history.train_losses) == 3
        assert len(history.val_losses) == 3

    def test_checkpoint_fn_called_only_on_improvement(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks([0.5, 0.3, 0.4])
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=3, tqdm_disable=True)
        assert len(records["ckpt"]) == 2
        assert records["ckpt"][0] == (0, pytest.approx(0.5))
        assert records["ckpt"][1] == (1, pytest.approx(0.3))

    def test_checkpoint_fn_not_called_when_no_improvement(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks([0.5, 0.6, 0.7])
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=3, tqdm_disable=True)
        # only first epoch improves (from inf)
        assert len(records["ckpt"]) == 1

    def test_snapshot_fn_called_every_val_epoch(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=3, tqdm_disable=True)
        assert records["snap"] == [0, 1, 2]

    def test_non_none_snapshot_populates_pred_history(self):
        T = torch.zeros(1, 1, 4, 4, 4)
        snap_calls: list = []

        def snap_fn(epoch: int):
            snap_calls.append(epoch)
            return (T, T) if epoch == 1 else None

        train_fn, val_fn, ckpt_fn, _, _ = _std_callbacks()
        history = run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=3, tqdm_disable=True)
        assert len(history.pred_history) == 1
        assert history.epoch_labels == ["Ep 1"]

    def test_first_non_none_snapshot_sets_T_gt_fixed(self):
        T = torch.zeros(1, 1, 4, 4, 4)
        snap_fn = lambda epoch: (T, T) if epoch == 0 else None  # noqa: E731
        train_fn, val_fn, ckpt_fn, _, _ = _std_callbacks()
        history = run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=2, tqdm_disable=True)
        assert history.T_gt_fixed is T

    def test_skips_history_and_callbacks_when_not_main(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        history = run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                                 epochs=3, is_main=False, tqdm_disable=True)
        assert history.train_losses == []
        assert history.val_losses == []
        assert records["ckpt"] == []
        assert records["snap"] == []      # snap_fn not called when is_main=False

    def test_train_fn_still_called_when_not_main(self):
        train_fn, val_fn, ckpt_fn, snap_fn, records = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                       epochs=3, is_main=False, tqdm_disable=True)
        assert records["train"] == [0, 1, 2]

    def test_sync_fn_called_after_each_val_epoch(self):
        syncs: list = []
        train_fn, val_fn, ckpt_fn, snap_fn, _ = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                       epochs=3, sync_fn=lambda: syncs.append(1), tqdm_disable=True)
        assert len(syncs) == 3

    def test_sync_fn_called_on_skipped_val_epochs_too(self):
        syncs: list = []
        train_fn, val_fn, ckpt_fn, snap_fn, _ = _std_callbacks()
        run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn,
                       epochs=4, val_every=2,
                       sync_fn=lambda: syncs.append(1), tqdm_disable=True)
        # epochs 0,2 run val (2 syncs), epochs 1,3 skip val (2 syncs) → total 4
        assert len(syncs) == 4

    def test_returns_train_history_instance(self):
        train_fn, val_fn, ckpt_fn, snap_fn, _ = _std_callbacks()
        result = run_train_loop(train_fn, val_fn, ckpt_fn, snap_fn, epochs=1, tqdm_disable=True)
        assert isinstance(result, TrainHistory)


# ---------------------------------------------------------------------------
# run_test_phase
# ---------------------------------------------------------------------------


def _noop_rollout(T_shape=(1, 1, 4, 4, 4)):
    T = torch.zeros(*T_shape)
    return lambda batch: (T, T)


@pytest.fixture()
def mock_run():
    return MagicMock()


@patch("neural_pbf.training.loops.evaluate_physical_metrics", return_value={"mae": 0.0})
@patch("neural_pbf.training.loops.iou_melt_volumes", return_value=0.5)
@patch("neural_pbf.training.loops.mlflow")
@patch("neural_pbf.training.loops.log_figure")
@patch("neural_pbf.training.loops.val_grid_2x2", return_value=MagicMock())
class TestRunTestPhase:
    def test_calls_rollout_fn_once_per_batch(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        calls: list = []
        T = torch.zeros(1, 1, 4, 4, 4)
        def rollout_fn(batch):
            calls.append(batch)
            return T, T
        run_test_phase(rollout_fn, _make_batches(3), tmp_path, mock_run, seed=42)
        assert len(calls) == 3

    def test_writes_mapping_file_with_header(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        run_test_phase(_noop_rollout(), _make_batches(1), tmp_path, mock_run, seed=42)
        lines = (tmp_path / "test_sample_mapping.txt").read_text().splitlines()
        assert lines[0] == "local_idx,h5_path,sample_key"

    def test_writes_mapping_rows_when_keys_in_batch(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        T = torch.zeros(1, 1, 4, 4, 4)
        batches = [
            {"h5_path": ["file.h5"], "sample_key": ["s_001"], "T_target": T},
            {"h5_path": ["file.h5"], "sample_key": ["s_002"], "T_target": T},
        ]
        run_test_phase(lambda b: (T, T), batches, tmp_path, mock_run, seed=42)
        lines = (tmp_path / "test_sample_mapping.txt").read_text().strip().splitlines()
        assert len(lines) == 3   # header + 2 data rows
        assert "file.h5" in lines[1] and "s_001" in lines[1]
        assert "s_002" in lines[2]

    def test_no_mapping_rows_written_without_keys(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        run_test_phase(_noop_rollout(), _make_batches(2), tmp_path, mock_run, seed=42)
        lines = (tmp_path / "test_sample_mapping.txt").read_text().strip().splitlines()
        assert len(lines) == 1  # header only

    def test_calls_tracker_log_artifact_when_tracker_provided(
            self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        tracker = MagicMock()
        run_test_phase(_noop_rollout(), _make_batches(1), tmp_path, mock_run,
                       seed=42, tracker=tracker)
        tracker.log_artifact.assert_called_once()

    def test_no_tracker_does_not_raise(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        run_test_phase(_noop_rollout(), _make_batches(1), tmp_path, mock_run,
                       seed=42, tracker=None)

    def test_returns_at_most_5_gallery_samples(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        gallery = run_test_phase(_noop_rollout(), _make_batches(10), tmp_path, mock_run, seed=42)
        assert len(gallery) <= 5

    def test_returns_gallery_samples_as_tensor_pairs(
            self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        gallery = run_test_phase(_noop_rollout(), _make_batches(2), tmp_path, mock_run, seed=42)
        for T_gt, T_pred in gallery:
            assert isinstance(T_gt, torch.Tensor)
            assert isinstance(T_pred, torch.Tensor)

    def test_seed_is_applied_before_rollout(self, mg, mlf, mmlf, miou, mph, tmp_path, mock_run):
        states: list[int] = []
        T = torch.zeros(1, 1, 4, 4, 4)
        def rollout_fn(batch):
            states.append(torch.randint(0, 1000, (1,)).item())
            return T, T
        p1, p2 = tmp_path / "r1", tmp_path / "r2"
        p1.mkdir(); p2.mkdir()
        run_test_phase(rollout_fn, _make_batches(3), p1, mock_run, seed=99)
        r1 = list(states); states.clear()
        run_test_phase(rollout_fn, _make_batches(3), p2, mock_run, seed=99)
        assert r1 == states


# ---------------------------------------------------------------------------
# log_val_image_rope
# ---------------------------------------------------------------------------


@patch("neural_pbf.training.loops.log_figure")
@patch("neural_pbf.training.loops.val_grid_2x2", return_value=MagicMock())
class TestLogValImageRope:
    def test_sets_eval_mode_during_rollout(self, mock_grid, mock_log, mock_run=None):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        states_during: list = []
        T = torch.zeros(2, 1, 4, 4, 4)

        def rollout_fn(batch):
            states_during.append((model.training, cond_encoder.training))
            return T[:1]

        log_val_image_rope(model, cond_encoder, rollout_fn,
                           {"T_target": T}, epoch=0, run=mock_run)
        assert states_during == [(False, False)]

    def test_restores_train_mode_after_return(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)
        rollout_fn = lambda batch: T[:1]  # noqa: E731
        log_val_image_rope(model, cond_encoder, rollout_fn,
                           {"T_target": T}, epoch=0, run=mock_run)
        assert model.training
        assert cond_encoder.training

    def test_restores_train_mode_even_on_exception(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)

        def bad_rollout(batch):
            raise RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            log_val_image_rope(model, cond_encoder, bad_rollout,
                               {"T_target": T}, epoch=5, run=mock_run)
        assert model.training
        assert cond_encoder.training

    def test_returns_cpu_tensor_pair(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)
        rollout_fn = lambda batch: T[:1]  # noqa: E731
        T_gt, T_pred = log_val_image_rope(
            model, cond_encoder, rollout_fn, {"T_target": T}, epoch=3, run=mock_run
        )
        assert T_gt.device.type == "cpu"
        assert T_pred.device.type == "cpu"

    def test_calls_log_figure_once(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)
        rollout_fn = lambda batch: T[:1]  # noqa: E731
        log_val_image_rope(model, cond_encoder, rollout_fn,
                           {"T_target": T}, epoch=10, run=mock_run)
        mock_log.assert_called_once()

    def test_filename_uses_epoch_number(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)
        rollout_fn = lambda batch: T[:1]  # noqa: E731
        log_val_image_rope(model, cond_encoder, rollout_fn,
                           {"T_target": T}, epoch=42, run=mock_run)
        _args, _kwargs = mock_log.call_args
        filename = _args[2] if len(_args) > 2 else _kwargs.get("artifact_name", "")
        assert "042" in filename

    def test_custom_filename_prefix_used(self, mock_grid, mock_log):
        mock_run = MagicMock()
        model = _TinyNet()
        cond_encoder = _TinyNet()
        T = torch.zeros(2, 1, 4, 4, 4)
        rollout_fn = lambda batch: T[:1]  # noqa: E731
        log_val_image_rope(model, cond_encoder, rollout_fn,
                           {"T_target": T}, epoch=0, run=mock_run,
                           filename_prefix="my_prefix")
        _args, _ = mock_log.call_args
        assert "my_prefix" in _args[2]
