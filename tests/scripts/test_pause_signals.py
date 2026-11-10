"""SIGINT/SIGTERM pause a training run after the current epoch (exit 75); a second one kills it at once, like a kill.

Ctrl-C in a terminal sends SIGINT to the whole process group, and the queue runner's stop sends SIGTERM to it. The
DataLoader workers lead their own process groups, so neither reaches them and they never die mid-epoch; they keep
SIGTERM's default action, which torch's shutdown relies on to end a worker that does not exit by itself.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from neural_pbf.uq.stages import EXIT_PAUSED
from tests.scripts.smoke_data import ROOT, make_smoke_data

TINY = ["--epochs", "3", "--batch_size", "2", "--embed_dim", "24", "--depth", "1", "--num_heads", "2",
        "--cond_embed_dim", "8", "--log_every", "1", "--num_workers", "2", "--device", "cpu"]  # fmt: skip


def _start(tmp_path: Path) -> subprocess.Popen[str]:
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    argv = [*TINY, "--manifest", str(manifest), "--mlflow_uri",
            f"file://{tmp_path / 'mlruns'}", "--checkpoint_dir", str(tmp_path / "ckpt")]  # fmt: skip
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "experiments" / "train_fm_dit_rope.py"), *argv], cwd=ROOT, text=True,
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )  # fmt: skip
    assert proc.stderr is not None
    deadline = time.time() + 300
    for line in proc.stderr:  # the handlers are installed when training starts
        if "training train_fm_dit_rope from epoch" in line or time.time() > deadline:
            break
    return proc


def _finish(proc: subprocess.Popen[str]) -> tuple[int, str]:
    _, err = proc.communicate(timeout=300)
    return proc.returncode, err


@pytest.mark.integration
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_one_signal_pauses_after_the_current_epoch(tmp_path: Path, sig: signal.Signals) -> None:
    proc = _start(tmp_path)
    os.killpg(proc.pid, sig)  # the whole group, as Ctrl-C and the queue runner's stop do
    code, err = _finish(proc)
    assert code == EXIT_PAUSED, err[-3000:]
    state = torch.load(tmp_path / "ckpt" / "latest.pt", weights_only=False)
    assert state["state"] == "paused" and 1 <= state["history"]["next_epoch"] < 3


@pytest.mark.integration
def test_a_second_ctrl_c_stops_at_once(tmp_path: Path) -> None:
    proc = _start(tmp_path)
    os.killpg(proc.pid, signal.SIGINT)
    time.sleep(0.2)
    os.killpg(proc.pid, signal.SIGINT)
    code, err = _finish(proc)
    assert code == -signal.SIGINT, err[-3000:]  # died from the signal, not a swallowed exception
    assert "stopping now" in err
