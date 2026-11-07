"""Held-out evaluation data for the experiment scripts: whole runs, and the inputs the model was trained on."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Subset

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.splits import load_manifest, manifest_sample_keys


def heldout_dataset(ckpt: dict[str, Any], manifest_path: str, split: str = "val", allow_test: bool = False) -> Subset:
    """The ``split`` runs of the manifest, read with the checkpoint's own dataset config.

    The ``test`` runs belong to the UQ pipeline, which logs every access to them; reading them here
    bypasses that log, so it needs ``allow_test``.

    Raises:
        KeyError: The checkpoint does not record its dataset config, so its inputs cannot be rebuilt.
        ValueError: ``split="test"`` without ``allow_test``.
    """
    if split == "test" and not allow_test:
        raise ValueError("the test runs are the UQ pipeline's held-out set; pass allow_test=True to read them")
    if "dataset_cfg" not in ckpt:
        raise KeyError("checkpoint has no dataset_cfg; retrain it with the current scripts")
    manifest = load_manifest(manifest_path)
    keys = manifest_sample_keys(manifest, split)
    paths = sorted({p for p, _ in keys})
    base = FMThermalDataset(FMDatasetConfig.from_checkpoint(ckpt["dataset_cfg"], h5_paths=paths))
    index = {key: i for i, key in enumerate(base._keys)}
    return Subset(base, [index[k] for k in keys])
