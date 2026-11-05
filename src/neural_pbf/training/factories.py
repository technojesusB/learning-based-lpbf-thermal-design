"""Dataset-building and experiment-tracking setup factories."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.patch_dataset import (
    PatchFMThermalDataset,
    PatchFMThermalDatasetWithOrigin,
    _collate_with_strings,
)
from neural_pbf.data.splits import load_manifest, manifest_sample_keys
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.factory import build_tracker

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Holds every data object produced by build_patch_data.

    The loader fields are mutable so callers can replace them after
    accelerator.prepare() without rebuilding the rest of the split.
    """

    full_ds: FMThermalDataset
    patch_ds: Dataset
    ds_cfg: FMDatasetConfig
    train_ds: Subset
    val_ds: Subset
    test_ds: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def build_patch_data(
    h5_paths: list[str],
    patch_size: int,
    batch_size: int,
    seed: int,
    *,
    use_origins: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    Q_ref: float = 1.35e15,
    manifest_path: str | None = None,
    mask_source: Literal["post", "pre"] = "pre",
    allow_random_split: bool = False,
    conditioning_keys: tuple[str, ...] | None = None,
) -> DataSplit:
    """Build patch dataset with 70/20/10 train-val-test split and DataLoaders.

    Args:
        h5_paths:    List of HDF5 data file paths.
        patch_size:  Cubic patch side length in voxels.
        batch_size:  Train/val DataLoader batch size (test always uses batch_size=1).
        seed:        RNG seed for reproducible splitting.
        use_origins: When True uses PatchFMThermalDatasetWithOrigin, which adds
                     patch_origin, sample_key, and h5_path fields to every batch.
                     Also automatically activates _collate_with_strings.  Required
                     for RoPE experiments and for the standardised test mapping file.
        train_frac:  Fraction of data assigned to training.
        val_frac:    Fraction of data assigned to validation.
        Q_ref:       Heat-flux normalisation reference [W/m³].
        manifest_path: Run-level split manifest. Replaces the random split: train and val come
                     from the manifest and test_ds stays empty, so training cannot load
                     calibration or test runs. Fractions are ignored.
        mask_source: "pre" (mask_in, causal) or "post" (legacy, leaks the answer).
        allow_random_split: Without a manifest, a random split over samples is refused: samples of one
                     run are strongly correlated, so it leaks. Set this only for tests or a single run.
        conditioning_keys: Override the dataset's default conditioning keys.

    Raises:
        ValueError: No manifest and ``allow_random_split`` is False.
    """
    if manifest_path is None and not allow_random_split:
        raise ValueError(
            "build_patch_data needs a run-level manifest (manifest_path); a random split over samples leaks "
            "neighbouring samples of one run into val/test. Pass allow_random_split=True to accept that."
        )
    extra: dict[str, Any] = {} if conditioning_keys is None else {"conditioning_keys": conditioning_keys}
    ds_cfg = FMDatasetConfig(h5_paths=h5_paths, Q_ref=Q_ref, mask_source=mask_source, **extra)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds: Dataset
    if use_origins:
        patch_ds = PatchFMThermalDatasetWithOrigin(full_ds, patch_size=patch_size)
    else:
        patch_ds = PatchFMThermalDataset(full_ds, patch_size=patch_size)
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        index = {key: i for i, key in enumerate(full_ds._keys)}
        try:
            train_ds = Subset(patch_ds, [index[k] for k in manifest_sample_keys(manifest, "train")])
            val_ds = Subset(patch_ds, [index[k] for k in manifest_sample_keys(manifest, "val")])
        except KeyError as exc:
            raise ValueError(f"Manifest sample {exc} not found in {h5_paths}") from exc
        test_ds = Subset(patch_ds, [])
        n_train, n_val, n_test = len(train_ds), len(val_ds), 0
    else:
        n = len(patch_ds)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(patch_ds, [n_train, n_val, n_test], generator=generator)
    collate_fn = _collate_with_strings if use_origins else None
    loader_kw: dict[str, Any] = {"collate_fn": collate_fn} if collate_fn is not None else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kw)
    logger.info("Dataset split — train=%d  val=%d  test=%d", n_train, n_val, n_test)
    return DataSplit(
        full_ds=full_ds,
        patch_ds=patch_ds,
        ds_cfg=ds_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def setup_run(
    args: Any,
    *,
    timestamped: bool = True,
) -> tuple[Any, Path]:
    """Initialise MLflow experiment and create the checkpoint directory.

    Args:
        args:        Parsed CLI namespace; must have mlflow_uri, mlflow_experiment,
                     and checkpoint_dir attributes.
        timestamped: When True (default) appends a YYYYMMDD_HHMMSS subdirectory to
                     checkpoint_dir so each run gets an isolated artifact directory.
                     Pass False for scripts that manage the directory themselves
                     (e.g. Triton experiment that uses a fixed top-level path).

    Returns:
        (tracker, ckpt_dir) — the tracker wraps mlflow and ckpt_dir is the
        Path that was created on disk.
    """
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    ckpt_base = Path(args.checkpoint_dir)
    if timestamped:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = ckpt_base / ts
    else:
        ckpt_dir = ckpt_base
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(
        TrackingConfig(
            enabled=True,
            backend="mlflow",
            experiment_name=args.mlflow_experiment,
            mlflow_tracking_uri=args.mlflow_uri,
        )
    )
    return tracker, ckpt_dir
