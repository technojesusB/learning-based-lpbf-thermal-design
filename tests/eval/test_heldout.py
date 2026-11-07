"""Held-out evaluation data: the checkpoint's own dataset config on whole runs from the manifest."""

from __future__ import annotations

import pytest

from neural_pbf.data.fm_dataset import FMDatasetConfig
from neural_pbf.data.splits import load_manifest, make_manifest, manifest_sample_keys, save_manifest
from neural_pbf.eval.heldout import heldout_dataset
from tests.data.test_mask_source import _file

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


@pytest.fixture
def setup(tmp_path):
    h5 = _file(tmp_path / "d.h5", n_runs=4, per_run=2)
    path = tmp_path / "m.json"
    save_manifest(make_manifest([h5], seed=1, fractions=FRACS), path)
    return h5, str(path)


@pytest.mark.unit
def test_val_runs_with_the_checkpoints_config(setup):
    h5, manifest = setup
    old = FMDatasetConfig(h5_paths=["gone.h5"], conditioning_keys=("power", "sigma")).model_dump()
    old.pop("mask_source")  # a checkpoint from before mask_source
    ds = heldout_dataset({"dataset_cfg": old}, manifest)
    base = ds.dataset
    assert base.cfg.mask_source == "post" and base.cfg.conditioning_keys == ("power", "sigma")
    keys = {base._keys[i] for i in ds.indices}
    assert keys == set(manifest_sample_keys(load_manifest(manifest), "val"))


@pytest.mark.unit
def test_the_uq_test_set_needs_explicit_consent(setup):
    _, manifest = setup
    cfg = {"dataset_cfg": FMDatasetConfig(h5_paths=["x.h5"]).model_dump()}
    with pytest.raises(ValueError, match="test"):
        heldout_dataset(cfg, manifest, split="test")
    assert len(heldout_dataset(cfg, manifest, split="test", allow_test=True)) == 2


@pytest.mark.unit
def test_checkpoint_without_dataset_cfg_fails_loudly(setup):
    _, manifest = setup
    with pytest.raises(KeyError, match="dataset_cfg"):
        heldout_dataset({"model_state": {}}, manifest)
