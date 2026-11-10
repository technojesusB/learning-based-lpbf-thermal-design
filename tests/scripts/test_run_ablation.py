"""The ablation queue runner end to end, in its test mode: tiny recipe overrides, on the CPU.

Every path the runner and the training runs write points into a temporary "main repository" (checkpoints, logs and
queue state, progress files, the MLflow store, the UQ root with the S1 marker), as when the queue runs from a
worktree; nothing may appear in this checkout. Test mode marks every run with its overrides (provenance key
test_overrides, MLflow experiment lpbf_debug), so the ablation evaluation can tell it apart.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

from tests.scripts.smoke_data import ROOT, make_smoke_data

RUNNER = ROOT / "experiments" / "run_ablation.py"
TINY_DIT = {"epochs": 3, "batch_size": 2, "embed_dim": 24, "depth": 1, "num_heads": 2, "cond_embed_dim": 8,
            "patch_size": 4}  # fmt: skip
# 30 tiny epochs (about 15 s): the stop lands mid-run, so the graceful path (finish the epoch, exit 75) is tested
TINY_UNET = {"epochs": 30, "batch_size": 2, "base_channels": 4, "depth": 1, "cond_embed_dim": 8, "mixed_precision": "no"}
CPU = {"CUDA_VISIBLE_DEVICES": ""}
WATCHED = [ROOT / "checkpoints" / "unet_none", ROOT / "checkpoints" / "dit_v3_rope", ROOT / "logs" / "ablation"]


@pytest.fixture()
def main_repo(tmp_path: Path) -> dict:
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    (tmp_path / "uq" / "markers").mkdir(parents=True)
    (tmp_path / "uq" / "markers" / "S1.done.json").write_text(json.dumps({"output_hash": "data-hash-test"}))
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"unet_none": TINY_UNET, "dit_v3_rope": TINY_DIT}))
    infra = ["--root", str(tmp_path / "logs"), "--checkpoint_root", str(tmp_path / "ckpt"), "--mlflow_uri",
             f"file://{tmp_path / 'mlruns'}", "--uq_root", str(tmp_path / "uq"), "--manifest", str(manifest),
             "--num_workers", "0", "--test-overrides", str(overrides)]  # fmt: skip
    before = {p: p.exists() for p in WATCHED}
    yield {"tmp": tmp_path, "infra": infra}
    assert {p: p.exists() for p in WATCHED} == before, "the runner wrote into this checkout"


def _runner(main_repo: dict, *argv: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNNER), *argv, *main_repo["infra"]], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout, env={**os.environ, **CPU})  # fmt: skip


def _latest(main_repo: dict, name: str) -> dict:
    return torch.load(main_repo["tmp"] / "ckpt" / name / "latest.pt", weights_only=False)


@pytest.mark.integration
def test_only_runs_one_experiment_pinned(main_repo: dict) -> None:
    out = _runner(main_repo, "--only", "dit_v3_rope")
    assert out.returncode == 0, out.stderr[-3000:]
    state = _latest(main_repo, "dit_v3_rope")
    prov = state["provenance"]
    assert state["state"] == "finished" and prov["pinned"] is True and prov["ablation_experiment"] == "dit_v3_rope"
    assert prov["test_overrides"] == TINY_DIT and prov["chapter"] == "position_encoding"
    assert not (main_repo["tmp"] / "ckpt" / "unet_none").exists()
    client = MlflowClient(tracking_uri=f"file://{main_repo['tmp'] / 'mlruns'}")
    run = client.get_run(prov["mlflow_run_id"])
    assert run.data.tags["ablation_experiment"] == "dit_v3_rope" and run.data.tags["pinned"] == "true"
    assert client.get_experiment(run.info.experiment_id).name == "lpbf_debug"  # test mode never reaches the study


@pytest.mark.integration
def test_the_queue_stops_gracefully_and_continues(main_repo: dict) -> None:
    proc = subprocess.Popen([sys.executable, str(RUNNER), *main_repo["infra"]], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, **CPU})  # fmt: skip
    progress = main_repo["tmp"] / "ckpt" / "unet_none" / "progress.json"
    deadline = time.time() + 300
    while not progress.exists() and time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)
    stop = _runner(main_repo, "--stop")
    assert stop.returncode == 0, stop.stderr
    out, _ = proc.communicate(timeout=300)
    assert proc.returncode == 0, out[-3000:]
    states = {n: json.loads((main_repo["tmp"] / "ckpt" / n / "progress.json").read_text())["state"]
              for n in ("unet_none", "dit_v3_rope") if (main_repo["tmp"] / "ckpt" / n / "progress.json").exists()}
    assert list(states.values()).count("paused") == 1, (states, out[-2000:])  # stopped after an epoch, not killed
    assert "dit_v3_rope" not in states or states["unet_none"] == "finished"  # nothing ran after the paused one

    status = _runner(main_repo, "--status")
    assert status.returncode == 0 and "paused" in status.stdout, status.stdout

    again = _runner(main_repo, timeout=900)
    assert again.returncode == 0, again.stderr[-3000:]
    for name in ("unet_none", "dit_v3_rope"):
        assert _latest(main_repo, name)["state"] == "finished"
    paused = next(n for n, s in states.items() if s == "paused")
    run_id = _latest(main_repo, paused)["provenance"]["mlflow_run_id"]
    run = MlflowClient(tracking_uri=f"file://{main_repo['tmp'] / 'mlruns'}").get_run(run_id)
    assert "resumed_at_epoch" in run.data.tags  # the paused experiment continued its own run

    finished = _runner(main_repo)
    assert finished.returncode == 0 and finished.stdout.count("skip") == 2, finished.stdout


@pytest.mark.integration
def test_the_preflight_reports_every_queued_experiment(main_repo: dict) -> None:
    out = _runner(main_repo, "--preflight")
    assert out.returncode == 0, out.stderr[-3000:]
    for name in ("unet_none", "dit_v3_rope"):
        assert any(line.startswith(name) and " OK " in line for line in out.stdout.splitlines()), out.stdout
    report = json.loads((main_repo["tmp"] / "logs" / "preflight.json").read_text())
    assert {r["name"] for r in report["experiments"]} == {"unet_none", "dit_v3_rope"}
    assert all(r["ok"] and r["est_epoch_min"] > 0 and r["est_total_h"] > 0 for r in report["experiments"])
    assert not (main_repo["tmp"] / "ckpt" / "unet_none").exists()  # throwaway checkpoints, not the queue's
