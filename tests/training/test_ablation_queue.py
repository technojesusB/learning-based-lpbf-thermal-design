"""The queue runner of the ablation: infrastructure-only options, experiment states, dry run, frozen-code refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_pbf.training import ablation_queue as q
from neural_pbf.training.ablation import EXPERIMENTS, QUEUE


def _paths(tmp_path: Path) -> q.Paths:
    return q.Paths(root=tmp_path / "logs", checkpoint_root=tmp_path / "ckpt", mlflow_uri=f"file://{tmp_path / 'm'}",
                   uq_root=tmp_path / "uq", manifest=tmp_path / "manifest.json", num_workers=4)  # fmt: skip


def _progress(paths: q.Paths, name: str, state: str, next_epoch: int, epochs: int = 100) -> None:
    d = paths.checkpoint_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "progress.json").write_text(json.dumps({"name": name, "state": state, "next_epoch": next_epoch,
                                                 "epochs": epochs, "eta_s": 3600.0, "sec_per_epoch": 36.0}))  # fmt: skip


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted({k for e in EXPERIMENTS.values() for k in (*e.recipe, *e.required)}))
def test_the_runner_takes_no_recipe_value(key: str) -> None:
    """Only infrastructure on the runner's command line, so the pinned recipe cannot be bypassed through it."""
    with pytest.raises(SystemExit):
        q.build_parser().parse_args([f"--{key}", "1"])


@pytest.mark.unit
def test_the_runner_takes_infrastructure_options() -> None:
    args = q.build_parser().parse_args(["--root", "r", "--checkpoint_root", "c", "--mlflow_uri", "sqlite:///x.db",
                                        "--uq_root", "u", "--manifest", "m.json", "--num_workers", "2"])  # fmt: skip
    paths = q.paths_from(args)
    assert paths.root == Path("r") and paths.checkpoint_root == Path("c") and paths.num_workers == 2


@pytest.mark.unit
def test_the_state_of_an_experiment_comes_from_its_progress_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert q.experiment_state(paths, "unet_none") == ("pending", 0, None)
    _progress(paths, "unet_none", "paused", 3)
    assert q.experiment_state(paths, "unet_none") == ("paused", 3, 100)
    _progress(paths, "unet_none", "running", 4)  # a killed run leaves "running": it resumes like a paused one
    assert q.experiment_state(paths, "unet_none") == ("paused", 4, 100)
    _progress(paths, "unet_none", "finished", 100)
    assert q.experiment_state(paths, "unet_none") == ("finished", 100, 100)


@pytest.mark.unit
def test_the_dry_run_lists_every_queued_experiment_and_validates_its_argv(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(q, "dirty_code_paths", lambda: [])
    paths = _paths(tmp_path)
    _progress(paths, "unet_none", "finished", 100)
    _progress(paths, "unet_physics_coupled", "paused", 7)
    assert q.dry_run(paths, list(QUEUE)) == 0
    out = capsys.readouterr().out
    lines = {line.split(":")[0].strip(): line for line in out.splitlines() if line.strip().split(":")[0] in QUEUE}
    assert list(lines) == list(QUEUE)
    assert "finished" in lines["unet_none"] and "paused at epoch 7/100" in lines["unet_physics_coupled"]
    assert all("argv OK" in line for line in lines.values())
    assert "pending" in lines["dit_v1_ape"]


@pytest.mark.unit
def test_the_dry_run_reports_uncommitted_code(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(q, "dirty_code_paths", lambda: ["src/neural_pbf/training/train_run.py"])
    q.dry_run(_paths(tmp_path), ["unet_none"])
    assert "BLOCKED" in capsys.readouterr().out


@pytest.mark.unit
def test_the_queue_refuses_uncommitted_code_before_starting_anything(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(q, "dirty_code_paths", lambda: ["src/neural_pbf/training/train_run.py"])
    monkeypatch.setattr(q, "run_child_with_retries", lambda *a, **k: pytest.fail("a child was started"))
    assert q.drive(_paths(tmp_path), ["unet_none"], overrides=None) == q.EXIT_REFUSED


@pytest.mark.unit
def test_the_queue_refuses_code_imported_from_another_checkout(tmp_path: Path, monkeypatch) -> None:
    """In a worktree, the main repo's virtualenv would import the main checkout: not the frozen code."""
    monkeypatch.setattr(q, "dirty_code_paths", lambda: [])
    monkeypatch.setattr(q, "_checkout_root", lambda: tmp_path / "elsewhere")
    monkeypatch.setattr(q, "run_child_with_retries", lambda *a, **k: pytest.fail("a child was started"))
    assert q.drive(_paths(tmp_path), ["unet_none"], overrides=None) == q.EXIT_REFUSED


@pytest.mark.unit
def test_test_overrides_restrict_the_queue_and_are_never_silent(tmp_path: Path) -> None:
    overrides = {"dit_v3_rope": {"epochs": 2}, "unet_none": {"epochs": 2}}
    assert q.queue_for(None, overrides) == ["unet_none", "dit_v3_rope"]  # queue order, not file order
    assert q.queue_for("dit_v3_rope", overrides) == ["dit_v3_rope"]
    assert q.queue_for(None, None) == list(QUEUE)
    with pytest.raises(SystemExit):
        q.queue_for("dit_v9", None)


@pytest.mark.unit
def test_the_preflight_store_is_a_valid_uri_for_a_relative_root(tmp_path: Path, monkeypatch) -> None:
    """The default --root is relative (logs/ablation); MLflow refuses file://logs/... ."""
    from mlflow.tracking import MlflowClient

    monkeypatch.chdir(tmp_path)
    uri = q.preflight_store(Path("logs/ablation/preflight_work"), "unet_none")
    assert uri.startswith("file:///") and uri.endswith("/logs/ablation/preflight_work/mlruns/unet_none")
    MlflowClient(tracking_uri=uri).search_experiments()  # accepted by MLflow
