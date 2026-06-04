"""Tests for neural_pbf.training.factories — build_patch_data / setup_run / DataSplit."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_pbf.data.patch_dataset import PatchFMThermalDataset, PatchFMThermalDatasetWithOrigin
from neural_pbf.training.factories import DataSplit, build_patch_data, setup_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFMDataset:
    """Minimal stand-in for FMThermalDataset — no HDF5 needed."""

    def __init__(self, n: int = 100) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:  # pragma: no cover
        raise NotImplementedError("test should not call __getitem__")


@pytest.fixture()
def fake_args(tmp_path):
    class _Args:
        mlflow_uri = "sqlite:///test.db"
        mlflow_experiment = "test_exp"
        checkpoint_dir = str(tmp_path / "checkpoints")
    return _Args()


# ---------------------------------------------------------------------------
# build_patch_data
# ---------------------------------------------------------------------------


class TestBuildPatchData:
    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_default_split_70_20_10(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=42)
        assert len(split.train_ds) == 70
        assert len(split.val_ds) == 20
        assert len(split.test_ds) == 10

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_split_sizes_sum_to_total(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(50)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=0)
        total = len(split.train_ds) + len(split.val_ds) + len(split.test_ds)
        assert total == 50

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_use_origins_false_produces_base_patch_dataset(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(60)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=0, use_origins=False)
        assert isinstance(split.patch_ds, PatchFMThermalDataset)
        assert not isinstance(split.patch_ds, PatchFMThermalDatasetWithOrigin)

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_use_origins_true_produces_origin_dataset(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(60)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=0, use_origins=True)
        assert isinstance(split.patch_ds, PatchFMThermalDatasetWithOrigin)

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_test_loader_always_batch_size_one(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=16, seed=42)
        assert split.test_loader.batch_size == 1

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_train_loader_uses_requested_batch_size(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=8, seed=42)
        assert split.train_loader.batch_size == 8

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_same_seed_produces_same_split(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        s1 = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=7)
        mock_ds_cls.return_value = _FakeFMDataset(100)
        s2 = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=7)
        assert list(s1.train_ds.indices) == list(s2.train_ds.indices)

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_different_seeds_produce_different_splits(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        s1 = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=1)
        mock_ds_cls.return_value = _FakeFMDataset(100)
        s2 = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=2)
        assert list(s1.train_ds.indices) != list(s2.train_ds.indices)

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_returns_data_split_instance(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=42)
        assert isinstance(split, DataSplit)


# ---------------------------------------------------------------------------
# DataSplit mutability
# ---------------------------------------------------------------------------


class TestDataSplit:
    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_loaders_are_mutable(self, mock_cfg_cls, mock_ds_cls):
        """Accelerate replaces DataLoaders after accelerator.prepare(); must be writable."""
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=42)
        sentinel = object()
        split.train_loader = sentinel
        assert split.train_loader is sentinel

    @patch("neural_pbf.training.factories.FMThermalDataset")
    @patch("neural_pbf.training.factories.FMDatasetConfig")
    def test_all_three_loaders_replaceable(self, mock_cfg_cls, mock_ds_cls):
        mock_ds_cls.return_value = _FakeFMDataset(100)
        split = build_patch_data(["fake.h5"], patch_size=8, batch_size=4, seed=42)
        a, b, c = object(), object(), object()
        split.train_loader = a
        split.val_loader = b
        split.test_loader = c
        assert split.train_loader is a
        assert split.val_loader is b
        assert split.test_loader is c


# ---------------------------------------------------------------------------
# setup_run
# ---------------------------------------------------------------------------


class TestSetupRun:
    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_timestamped_creates_subdirectory(self, mock_mlflow, mock_tracker, fake_args):
        _, ckpt_dir = setup_run(fake_args, timestamped=True)
        assert ckpt_dir.parent == Path(fake_args.checkpoint_dir)
        assert ckpt_dir.exists()
        assert re.match(r"\d{8}_\d{6}", ckpt_dir.name), f"Expected timestamp, got: {ckpt_dir.name}"

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_not_timestamped_uses_base_dir_directly(self, mock_mlflow, mock_tracker, fake_args):
        _, ckpt_dir = setup_run(fake_args, timestamped=False)
        assert ckpt_dir == Path(fake_args.checkpoint_dir)
        assert ckpt_dir.exists()

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_default_is_timestamped(self, mock_mlflow, mock_tracker, fake_args):
        _, ckpt_dir = setup_run(fake_args)
        assert ckpt_dir.parent == Path(fake_args.checkpoint_dir)

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_calls_mlflow_set_tracking_uri(self, mock_mlflow, mock_tracker, fake_args):
        setup_run(fake_args)
        mock_mlflow.set_tracking_uri.assert_called_once_with(fake_args.mlflow_uri)

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_calls_mlflow_set_experiment(self, mock_mlflow, mock_tracker, fake_args):
        setup_run(fake_args)
        mock_mlflow.set_experiment.assert_called_once_with(fake_args.mlflow_experiment)

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_returns_tracker_and_path(self, mock_mlflow, mock_tracker, fake_args):
        tracker, ckpt_dir = setup_run(fake_args)
        assert tracker is mock_tracker.return_value
        assert isinstance(ckpt_dir, Path)

    @patch("neural_pbf.training.factories.build_tracker")
    @patch("neural_pbf.training.factories.mlflow")
    def test_ckpt_dir_is_created_on_disk(self, mock_mlflow, mock_tracker, fake_args):
        _, ckpt_dir = setup_run(fake_args, timestamped=False)
        assert ckpt_dir.is_dir()
