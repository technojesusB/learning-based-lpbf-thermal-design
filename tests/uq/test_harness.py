"""Chunked sampling: resumable, atomic, deterministic regardless of interruption."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_pbf.data.splits import discover_runs, make_manifest, save_manifest
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.harness import ModelSpec, legacy_own_split, load_draws, sample_set
from neural_pbf.uq.train import train_baseline
from tests.data.test_mask_source import _file

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("h")
    h5 = _file(tmp / "d.h5", n_runs=4, per_run=3)
    manifest = make_manifest([h5], seed=1, fractions=FRACS)
    path = tmp / "m.json"
    save_manifest(manifest, path)
    cfg = UQConfig.smoke(tmp / "r")
    cfg = cfg.model_copy(update={
        "train": cfg.train.model_copy(update={"base_channels": 4, "cond_embed_dim": 16, "max_epochs": 1}),
        "prereg": cfg.prereg.model_copy(update={"n_euler_steps": 2}),
        "run": cfg.run.model_copy(update={"device": "cpu", "chunk_size": 2, "mlflow_uri": f"sqlite:///{tmp}/m.db"}),
    })
    train_baseline(cfg, path, tmp / "ck")
    return cfg, manifest, ModelSpec("m", str(tmp / "ck" / "best.pt"), "net")


def _sample(cfg, manifest, spec, out, **kw):
    runs = manifest["sets"]["test"]
    return sample_set(spec, runs, out, cfg, base_seed=5, **kw)


@pytest.mark.integration
def test_sampling_writes_chunks_with_expected_shapes(trained, tmp_path):
    cfg, manifest, spec = trained
    _sample(cfg, manifest, spec, tmp_path)
    d = load_draws(tmp_path)
    assert d["draws"].shape == (3, cfg.prereg.K, 3) and d["truth"].shape == (3, 3)
    assert len(list(tmp_path.glob("chunk_*.npz"))) == 2
    assert np.isfinite(d["draws"]).all()
    assert set(d["run_id"]) == {manifest["sets"]["test"][0]["run_id"]}


@pytest.mark.integration
def test_resume_skips_finished_chunks_and_result_is_identical(trained, tmp_path):
    cfg, manifest, spec = trained
    a, b = tmp_path / "a", tmp_path / "b"
    _sample(cfg, manifest, spec, a)
    _sample(cfg, manifest, spec, b)
    first = sorted(b.glob("chunk_*.npz"))[0]
    mtime = first.stat().st_mtime_ns
    sorted(b.glob("chunk_*.npz"))[1].unlink()  # simulated crash before the last chunk
    _sample(cfg, manifest, spec, b)
    assert first.stat().st_mtime_ns == mtime
    assert np.array_equal(load_draws(a)["draws"], load_draws(b)["draws"])


@pytest.mark.integration
def test_no_leftover_temp_files(trained, tmp_path):
    cfg, manifest, spec = trained
    _sample(cfg, manifest, spec, tmp_path)
    assert not list(tmp_path.glob("*tmp*"))


@pytest.mark.unit
def test_load_draws_empty_dir_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_draws(tmp_path)


@pytest.mark.unit
def test_legacy_own_split_matches_recorded_baseline_sizes(tmp_path):
    h5 = _file(tmp_path / "n.h5", n_runs=3, per_run=50)
    cal, test = legacy_own_split([h5], seed=42)
    assert sum(len(r["sample_keys"]) for r in cal) == 30
    assert sum(len(r["sample_keys"]) for r in test) == 15
    runs, _ = discover_runs([h5])
    assert {r["run_id"] for r in test} <= {r.run_id for r in runs}
