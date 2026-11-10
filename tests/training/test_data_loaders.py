"""DataLoader workers are infrastructure: the worker count must not change which batches a run sees, or their order."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_pbf.data.splits import load_manifest
from neural_pbf.training import build_patch_data
from neural_pbf.training.train_run import data_loaders
from tests.scripts.smoke_data import make_smoke_data


def _epoch(loader) -> list[dict]:
    return [{k: v for k, v in b.items() if isinstance(v, torch.Tensor)} for b in loader]


@pytest.mark.integration
def test_the_worker_count_does_not_change_the_batches(tmp_path: Path) -> None:
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    h5 = sorted({p for runs in load_manifest(manifest)["sets"].values() for r in runs for p, _ in r["sample_keys"]})
    split = build_patch_data(h5, manifest_path=str(manifest), allow_random_split=False, patch_size=64, batch_size=2,
                             seed=42, use_origins=True)  # fmt: skip
    serial_train, serial_val = data_loaders(split, batch_size=2, num_workers=0, seed=42)
    workers_train, workers_val = data_loaders(split, batch_size=2, num_workers=4, seed=42)
    assert workers_train.num_workers == 4 and workers_train.persistent_workers
    for serial, workers in ((serial_train, workers_train), (serial_val, workers_val)):
        a, b = _epoch(serial), _epoch(workers)
        assert len(a) == len(b) > 1
        for x, y in zip(a, b, strict=True):
            assert x.keys() == y.keys()
            for k in x:
                assert torch.equal(x[k], y[k]), k
    powers = {tuple(row.tolist()) for b in _epoch(serial_train) for row in b["conditioning"]}
    assert len(powers) > 1  # the train set holds several runs, so a reordering would show


@pytest.mark.integration
def test_the_train_order_depends_only_on_the_seed(tmp_path: Path) -> None:
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    h5 = sorted({p for runs in load_manifest(manifest)["sets"].values() for r in runs for p, _ in r["sample_keys"]})
    split = build_patch_data(h5, manifest_path=str(manifest), allow_random_split=False, patch_size=64, batch_size=2,
                             seed=42, use_origins=True)  # fmt: skip
    first = _epoch(data_loaders(split, batch_size=2, num_workers=0, seed=42)[0])
    torch.manual_seed(12345)  # the global RNG must not matter
    second = _epoch(data_loaders(split, batch_size=2, num_workers=0, seed=42)[0])
    assert all(torch.equal(x["T_in"], y["T_in"]) for x, y in zip(first, second, strict=True))
