"""Baseline training: val-only selection, resume, early stopping, loud failure on NaN."""

from __future__ import annotations

import pytest
import torch

from neural_pbf.data.splits import make_manifest, save_manifest
from neural_pbf.eval.benchmark.model_loader import load_model_adaptive
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.train import TrainingError, train_baseline
from tests.data.test_mask_source import _file

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


def _setup(tmp_path, **train_kw):
    h5 = _file(tmp_path / "d.h5", n_runs=4)
    manifest = make_manifest([h5], seed=1, fractions=FRACS)
    path = tmp_path / "m.json"
    save_manifest(manifest, path)
    cfg = UQConfig.smoke(tmp_path / "r")
    cfg = cfg.model_copy(update={
        "train": cfg.train.model_copy(update={"base_channels": 4, "cond_embed_dim": 16, **train_kw}),
        "run": cfg.run.model_copy(update={"device": "cpu", "mlflow_uri": f"sqlite:///{tmp_path}/m.db"}),
    })
    return cfg, path, manifest, tmp_path / "ckpt"


@pytest.mark.integration
def test_checkpoint_carries_provenance_and_loads(tmp_path):
    cfg, path, manifest, out = _setup(tmp_path)
    result = train_baseline(cfg, path, out)
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["manifest_hash"] == manifest["hash"] and ckpt["manifest_path"] == str(path)
    assert ckpt["dataset_cfg"]["mask_source"] == "pre"
    assert ckpt["val_loss"] == min(result["val_history"])
    load_model_adaptive("t", out / "best.pt", "net", torch.device("cpu"))


@pytest.mark.integration
def test_resume_continues_and_matches_uninterrupted_val_history(tmp_path):
    cfg, path, _, out = _setup(tmp_path, max_epochs=3, patience=10)
    straight = train_baseline(cfg, path, out)
    short = cfg.model_copy(update={"train": cfg.train.model_copy(update={"max_epochs": 2})})
    out2 = tmp_path / "ckpt2"
    first = train_baseline(short, path, out2)
    assert len(first["val_history"]) == 2
    resumed = train_baseline(cfg, path, out2)
    assert resumed["epochs_run_this_call"] == 1
    assert resumed["val_history"] == pytest.approx(straight["val_history"])


@pytest.mark.integration
def test_finished_training_is_a_no_op(tmp_path):
    cfg, path, _, out = _setup(tmp_path)
    train_baseline(cfg, path, out)
    assert train_baseline(cfg, path, out)["epochs_run_this_call"] == 0


@pytest.mark.integration
def test_early_stopping_recorded(tmp_path):
    cfg, path, _, out = _setup(tmp_path, max_epochs=30, patience=1, lr=1.0)
    result = train_baseline(cfg, path, out)
    assert result["stopped_early"] is True and len(result["val_history"]) < 30


@pytest.mark.integration
def test_resume_under_other_manifest_refused(tmp_path):
    cfg, path, _, out = _setup(tmp_path)
    train_baseline(cfg, path, out)
    h5 = _file(tmp_path / "e.h5", n_runs=4)
    other = tmp_path / "o.json"
    save_manifest(make_manifest([h5], seed=2, fractions=FRACS), other)
    with pytest.raises(TrainingError, match="manifest"):
        train_baseline(cfg, other, out)


@pytest.mark.integration
def test_nan_loss_fails_loudly(tmp_path, monkeypatch):
    import neural_pbf.uq.train as mod

    cfg, path, _, out = _setup(tmp_path)
    monkeypatch.setattr(mod, "fm_loss", lambda *a: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(TrainingError, match="non-finite"):
        train_baseline(cfg, path, out)


# --- per-epoch shuffle and pausing ------------------------------------------------------


@pytest.mark.unit
def test_epoch_shuffle_depends_only_on_base_seed_and_epoch():
    from torch.utils.data import DataLoader

    from neural_pbf.uq.train import seed_epoch_shuffle

    loader = DataLoader(list(range(50)), shuffle=True)

    def order(base, epoch):
        seed_epoch_shuffle(loader, base, epoch)
        return list(loader.sampler)

    first = order(7, 3)
    torch.manual_seed(999)  # the global RNG must not matter
    assert order(7, 3) == first
    assert order(7, 4) != first and order(8, 3) != first
    assert sorted(first) == list(range(50))


def _record_batches(monkeypatch):
    import neural_pbf.uq.train as mod

    seen = []
    real = mod._loss

    def spy(batch, model, encoder, device, gen):
        if gen is None:  # training only; validation passes its own generator
            seen.append(batch["conditioning"].tolist())
        return real(batch, model, encoder, device, gen)

    monkeypatch.setattr(mod, "_loss", spy)
    return seen


@pytest.mark.integration
def test_pause_and_resume_match_an_uninterrupted_run(tmp_path, monkeypatch):
    cfg, path, _, out = _setup(tmp_path, max_epochs=4, patience=10)
    seen = _record_batches(monkeypatch)
    straight = train_baseline(cfg, path, out)
    straight_batches = list(seen)
    seen.clear()

    out2 = tmp_path / "ckpt2"
    calls = iter([False, True])
    first = train_baseline(cfg, path, out2, should_pause=lambda: next(calls))
    assert first["paused"] and first["epochs_run"] == 2
    assert (out2 / "latest.pt").exists()
    resumed = train_baseline(cfg, path, out2)
    assert not resumed["paused"] and resumed["epochs_run_this_call"] == 2
    assert seen == straight_batches, "the resumed run must see the same batches in the same order"
    assert resumed["val_history"] == pytest.approx(straight["val_history"])
    assert resumed["train_history"] == pytest.approx(straight["train_history"])


@pytest.mark.integration
def test_deadline_after_the_final_epoch_is_not_a_pause(tmp_path):
    cfg, path, _, out = _setup(tmp_path, max_epochs=1, patience=10)
    assert not train_baseline(cfg, path, out, should_pause=lambda: True)["paused"]


@pytest.mark.integration
def test_deadline_at_early_stop_is_not_a_pause(tmp_path):
    cfg, path, _, out = _setup(tmp_path, max_epochs=5, patience=0)
    result = train_baseline(cfg, path, out, should_pause=lambda: True)
    assert result["stopped_early"] and not result["paused"]


@pytest.mark.integration
def test_restart_after_the_deadline_pauses_before_another_epoch(tmp_path):
    cfg, path, _, out = _setup(tmp_path, max_epochs=4, patience=10)
    calls = iter([True])
    assert train_baseline(cfg, path, out, should_pause=lambda: next(calls))["epochs_run"] == 1
    again = train_baseline(cfg, path, out, should_pause=lambda: True)
    assert again["paused"] and again["epochs_run_this_call"] == 0
