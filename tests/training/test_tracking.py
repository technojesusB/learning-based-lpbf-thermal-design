"""Shared MLflow tracking of every training run: params, tags, step and epoch metrics, artifacts, provenance.

A standalone script run uses the default ``RunHooks`` (MLflow experiment lpbf_experiments, pinned=false); the ablation
layer passes hooks that pin the run. These tests build the hooks directly, without the ablation registry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import Subset

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.splits import load_manifest, make_manifest, manifest_sample_keys, save_manifest
from neural_pbf.training.checkpointing import save_checkpoint
from neural_pbf.training.tracking import (
    ABLATION_EXPERIMENT,
    REQUIRED_EPOCH,
    REQUIRED_STEP,
    STANDALONE_EXPERIMENT,
    RunHooks,
    RunTracker,
    split_runs,
)
from tests.data.test_mask_source import _file

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}
SCRIPT = "train_fm_unet_triton.py"
CONFIGS = {"fm_cfg": {"depth": 4, "base_channels": 32}, "dataset_cfg": {"T_ref": 2000.0, "cond_means": {"rho": 8000.0}}}
PREFLIGHT = {"preflight": "passed", "preflight_max_rel_err": 1e-12}


def _pinned(name: str = "unet_physics_tin", chapter: str = "physics_loss") -> RunHooks:
    """What the ablation layer passes; written out here so the shared layer is tested without it."""
    return RunHooks(name=name, mlflow_experiment=ABLATION_EXPERIMENT, pinned=True,
                    record={"ablation_experiment": name, "chapter": chapter, "recipe": {"depth": 4, "lr": 1e-4}})  # fmt: skip


@pytest.fixture()
def data(tmp_path: Path) -> dict:
    h5 = _file(tmp_path / "a.h5", n_runs=4, per_run=2)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(make_manifest([h5], seed=1, fractions=FRACS), manifest_path)
    manifest = load_manifest(manifest_path)
    full = FMThermalDataset(FMDatasetConfig(h5_paths=[h5]))
    index = {k: i for i, k in enumerate(full._keys)}
    subset = {s: Subset(full, [index[k] for k in manifest_sample_keys(manifest, s)]) for s in FRACS}
    marker = tmp_path / "uq" / "markers" / "S1.done.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"output_hash": "data-hash-123"}))
    return {"manifest_path": str(manifest_path), "manifest": manifest, "subset": subset, "uq_root": tmp_path / "uq"}


def _args(tmp_path: Path, data: dict, **kw) -> argparse.Namespace:
    base = {"physics_form": "tin", "physics_backend": "triton", "lr": 1e-4, "epochs": 100,
            "mlflow_uri": f"file://{tmp_path / 'mlruns'}", "manifest": data["manifest_path"], "log_every": 1,
            "checkpoint_dir": str(tmp_path / "ckpt"), "num_workers": 4}  # fmt: skip
    return argparse.Namespace(**{**base, **kw})


def _start(tmp_path: Path, data: dict, hooks: RunHooks | None = None, **kw) -> RunTracker:
    runs = split_runs(data["manifest_path"], data["subset"]["train"], data["subset"]["val"])
    return RunTracker.start(_args(tmp_path, data, **kw), hooks or _pinned(), script=SCRIPT, split=runs,
                            configs=CONFIGS, preflight=PREFLIGHT, uq_root=data["uq_root"])  # fmt: skip


def _step_metrics() -> dict[str, float]:
    return {k: 0.5 for k in REQUIRED_STEP}


def _epoch_metrics() -> dict[str, float]:
    return {**{k: 0.25 for k in REQUIRED_EPOCH}, "val_phys_tin_melt": math.nan}


# --- split ----------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_split_runs_returns_the_train_and_val_run_ids(data: dict) -> None:
    runs = split_runs(data["manifest_path"], data["subset"]["train"], data["subset"]["val"])
    sets = data["manifest"]["sets"]
    assert runs == {"train_runs": [r["run_id"] for r in sets["train"]], "val_runs": [r["run_id"] for r in sets["val"]]}


@pytest.mark.unit
@pytest.mark.parametrize("leak", ["cal", "test"])
def test_split_runs_refuses_cal_and_test_samples(data: dict, leak: str) -> None:
    train = data["subset"]["train"]
    leaky = Subset(train.dataset, [*train.indices, data["subset"][leak].indices[0]])
    with pytest.raises(ValueError, match=leak):
        split_runs(data["manifest_path"], leaky, data["subset"]["val"])


@pytest.mark.unit
def test_split_runs_refuses_val_samples_in_train(data: dict) -> None:
    train = data["subset"]["train"]
    leaky = Subset(train.dataset, [*train.indices, data["subset"]["val"].indices[0]])
    with pytest.raises(ValueError, match="val"):
        split_runs(data["manifest_path"], leaky, data["subset"]["val"])


# --- tracking -------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_pinned_run_logs_every_required_key(tmp_path: Path, data: dict) -> None:
    tracker = _start(tmp_path, data)
    with tracker:
        tracker.step(_step_metrics(), step=0)
        tracker.epoch(_epoch_metrics(), epoch=0)
        tracker.finish(best=tmp_path / "ckpt" / "best.pt", final=tmp_path / "ckpt" / "latest.pt")
    client = MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")
    run = client.get_run(tracker.run_id)
    assert client.get_experiment(run.info.experiment_id).name == ABLATION_EXPERIMENT
    assert run.info.run_name == "unet_physics_tin"
    tags = run.data.tags
    for key in ("git_commit", "git_dirty", "manifest_path", "manifest_hash", "data_hash", "train_runs", "val_runs",
                "ablation_experiment", "chapter", "script", "pinned", "preflight"):  # fmt: skip
        assert key in tags, key
    assert tags["ablation_experiment"] == "unet_physics_tin" and tags["chapter"] == "physics_loss"
    assert tags["pinned"] == "true" and tags["script"] == SCRIPT
    assert tags["data_hash"] == "data-hash-123" and tags["manifest_hash"] == data["manifest"]["hash"]
    params = run.data.params
    for key in ("physics_form", "lr", "ablation.recipe.depth", "ablation.chapter", "fm_cfg.depth",
                "dataset_cfg.cond_means.rho", "preflight_max_rel_err"):  # fmt: skip
        assert key in params, key
    metrics = run.data.metrics
    assert set(REQUIRED_STEP) | set(REQUIRED_EPOCH) <= set(metrics)
    artifacts = {a.path for a in client.list_artifacts(tracker.run_id)}
    assert "resolved_config.json" in artifacts
    assert tags["best_checkpoint"].endswith("best.pt") and tags["final_checkpoint"].endswith("latest.pt")


@pytest.mark.unit
@pytest.mark.parametrize(("method", "metrics"), [("step", _step_metrics), ("epoch", _epoch_metrics)])
def test_missing_metrics_are_refused(tmp_path: Path, data: dict, method: str, metrics) -> None:
    tracker = _start(tmp_path, data)
    with tracker:
        incomplete = metrics()
        incomplete.pop(next(iter(incomplete)))
        with pytest.raises(ValueError, match="missing"):
            getattr(tracker, method)(incomplete, 0)


@pytest.mark.unit
def test_a_standalone_run_goes_to_lpbf_experiments_unpinned(tmp_path: Path, data: dict) -> None:
    (data["uq_root"] / "markers" / "S1.done.json").unlink()  # a standalone run needs no UQ pipeline
    tracker = _start(tmp_path, data, RunHooks())
    with tracker:
        pass
    client = MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")
    run = client.get_run(tracker.run_id)
    assert client.get_experiment(run.info.experiment_id).name == STANDALONE_EXPERIMENT
    assert run.info.run_name == "train_fm_unet_triton" and run.data.tags["pinned"] == "false"
    assert "ablation_experiment" not in run.data.tags and "ablation_experiment" not in tracker.provenance
    assert tracker.provenance["pinned"] is False and tracker.provenance["data_hash"] == "unknown"


@pytest.mark.unit
def test_a_pinned_run_needs_the_dataset_hash(tmp_path: Path, data: dict) -> None:
    (data["uq_root"] / "markers" / "S1.done.json").unlink()
    with pytest.raises(ValueError, match="S1"):
        _start(tmp_path, data)


@pytest.mark.unit
def test_checkpoint_carries_the_provenance(tmp_path: Path, data: dict) -> None:
    tracker = _start(tmp_path, data)
    with tracker:
        net = torch.nn.Linear(2, 2)
        (tmp_path / "ckpt").mkdir(exist_ok=True)
        save_checkpoint(net, net, tmp_path / "ckpt", 0, 0.1, _args(tmp_path, data), provenance=tracker.provenance)
    ckpt = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=False)
    prov = ckpt["provenance"]
    assert prov["ablation_experiment"] == "unet_physics_tin" and prov["pinned"] is True
    assert prov["data_hash"] == "data-hash-123" and prov["name"] == "unet_physics_tin"
    assert prov["params"]["ablation.recipe.depth"] == 4 and prov["train_runs"]


# --- resume ---------------------------------------------------------------------------------------------------


def _resume(tmp_path: Path, data: dict, provenance: dict, epoch: int, hooks: RunHooks | None = None, **kw):
    return RunTracker.resume(_args(tmp_path, data, **kw), hooks or _pinned(), provenance=provenance, epoch=epoch,
                             script=SCRIPT, configs=CONFIGS, preflight=PREFLIGHT, uq_root=data["uq_root"])  # fmt: skip


@pytest.mark.unit
def test_a_paused_run_is_marked_and_resumes_into_the_same_mlflow_run(tmp_path: Path, data: dict) -> None:
    client = MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}")
    first = _start(tmp_path, data)
    with first:
        first.epoch(_epoch_metrics(), epoch=0)
        first.pause()
    paused = client.get_run(first.run_id)
    assert paused.info.status == "KILLED" and paused.data.tags["state"] == "paused"

    second = _resume(tmp_path, data, first.provenance, epoch=1, num_workers=0)  # infrastructure may change
    with second:
        second.epoch(_epoch_metrics(), epoch=1)
        second.finish(best=tmp_path / "best.pt", final=tmp_path / "latest.pt")
    run = client.get_run(first.run_id)
    assert second.run_id == first.run_id and run.info.status == "FINISHED"
    assert run.data.tags["resumed_at_epoch"] == "1" and run.data.tags["state"] == "finished"
    assert [m.step for m in client.get_metric_history(first.run_id, "val_loss_fm")] == [0, 1]
    assert len(client.search_runs([run.info.experiment_id])) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"hooks": _pinned("unet_physics_coupled")}, "unet_physics_coupled"),
        ({"hooks": _pinned(chapter="other")}, "ablation"),
        ({"lr": 2e-4}, "lr"),
        ({"provenance": {"manifest_hash": "other"}}, "manifest"),
        ({"marker": "other-data"}, "data hash"),
        ({"commit": "0000000"}, "commit"),
        ({"hooks": RunHooks()}, "pinned"),
    ],
)  # fmt: skip
def test_resume_refuses_a_changed_run(tmp_path: Path, data: dict, monkeypatch, change: dict, match: str) -> None:
    first = _start(tmp_path, data)
    with first:
        first.pause()
    provenance = {**first.provenance, **change.pop("provenance", {})}
    if "marker" in change:
        (data["uq_root"] / "markers" / "S1.done.json").write_text(json.dumps({"output_hash": change.pop("marker")}))
    if "commit" in change:
        monkeypatch.setattr("neural_pbf.training.tracking.git_commit", lambda: change.pop("commit"))
    with pytest.raises(ValueError, match=match):
        _resume(tmp_path, data, provenance, epoch=1, **change)


@pytest.mark.unit
def test_a_standalone_run_may_resume_across_commits(tmp_path: Path, data: dict, monkeypatch) -> None:
    first = _start(tmp_path, data, RunHooks())
    with first:
        first.pause()
    monkeypatch.setattr("neural_pbf.training.tracking.git_commit", lambda: "0000000")
    second = _resume(tmp_path, data, first.provenance, epoch=1, hooks=RunHooks())
    with second:
        pass
    run = MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}").get_run(first.run_id)
    assert run.data.tags["resumed_git_commit"] == "0000000"
    assert second.provenance["git_commit"] == first.provenance["git_commit"]  # the run's own commit stays
