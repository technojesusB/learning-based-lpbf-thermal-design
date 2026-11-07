"""Shared data arguments of the experiment scripts: run-level manifest by default, T_in on by default."""

from __future__ import annotations

import argparse

import pytest
import torch

from neural_pbf.data.splits import make_manifest, save_manifest
from neural_pbf.training.checkpointing import save_checkpoint
from neural_pbf.training.script_args import add_data_args, resolve_data
from tests.data.test_mask_source import _file

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    return parser.parse_args(argv)


@pytest.mark.unit
def test_defaults_are_the_uq_manifest_and_four_channels():
    args = _parse([])
    assert args.manifest == "artifacts/uq/manifest.json"
    assert args.in_channels == 4 and args.h5 is None and not args.allow_random_split


@pytest.mark.unit
def test_h5_paths_come_from_the_manifest(tmp_path):
    a = _file(tmp_path / "a.h5", n_runs=4)
    path = tmp_path / "m.json"
    save_manifest(make_manifest([a], seed=1, fractions=FRACS), path)
    h5, manifest = resolve_data(_parse(["--manifest", str(path)]))
    assert h5 == [a] and manifest == str(path)


@pytest.mark.unit
def test_missing_manifest_fails_with_a_hint(tmp_path):
    with pytest.raises(SystemExit, match="manifest"):
        resolve_data(_parse(["--manifest", str(tmp_path / "none.json")]))


@pytest.mark.unit
def test_random_split_needs_explicit_h5_and_flag(tmp_path):
    a = _file(tmp_path / "a.h5")
    h5, manifest = resolve_data(_parse(["--h5", a, "--allow-random-split"]))
    assert h5 == [a] and manifest is None
    with pytest.raises(SystemExit, match="allow-random-split"):
        resolve_data(_parse(["--h5", a, "--manifest", ""]))


@pytest.mark.unit
def test_checkpoint_records_dataset_and_model_config(tmp_path):
    m, e = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    save_checkpoint(m, e, tmp_path, 3, 0.5, argparse.Namespace(lr=1.0), dataset_cfg={"mask_source": "pre"},
                    fm_cfg={"in_channels": 4})  # fmt: skip
    ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
    assert ckpt["dataset_cfg"] == {"mask_source": "pre"} and ckpt["fm_cfg"] == {"in_channels": 4}
