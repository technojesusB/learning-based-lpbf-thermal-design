"""A training run paused after one epoch and resumed ends bitwise where an uninterrupted run ends (CPU, fp32).

On the GPU in bf16, two uninterrupted runs already differ (atomic and Triton kernels), so there a resumed run is only
required to continue; this test pins the exact case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

from neural_pbf.training.tracking import STANDALONE_EXPERIMENT
from neural_pbf.uq.stages import EXIT_PAUSED
from tests.scripts.smoke_data import make_smoke_data, run_script

TINY = ["--epochs", "2", "--batch_size", "2", "--embed_dim", "24", "--depth", "1", "--num_heads", "2",
        "--cond_embed_dim", "8", "--log_every", "1", "--num_workers", "2"]  # fmt: skip
CASES = {
    "plain": ("train_fm_dit_rope.py", ["--device", "cpu"]),
    "accelerate": ("train_fm_dit_triton.py", ["--mixed_precision", "no"]),
}
CPU = {"CUDA_VISIBLE_DEVICES": ""}


def _run(script: str, argv: list[str], manifest: Path, out: Path) -> int:
    paths = ["--manifest", str(manifest), "--mlflow_uri", f"file://{out / 'mlruns'}", "--checkpoint_dir",
             str(out / "ckpt")]  # fmt: skip
    proc = run_script(script, [*argv, *TINY, *paths], env=CPU)
    assert proc.returncode in (0, EXIT_PAUSED), proc.stderr[-3000:]
    return proc.returncode


def _tensors(obj, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return {prefix: obj}
    if isinstance(obj, dict):
        return {k: v for key, val in obj.items() for k, v in _tensors(val, f"{prefix}.{key}").items()}
    if isinstance(obj, list | tuple):
        return {k: v for i, val in enumerate(obj) for k, v in _tensors(val, f"{prefix}[{i}]").items()}
    return {}


@pytest.mark.integration
@pytest.mark.parametrize("case", sorted(CASES))
def test_pause_and_resume_match_an_uninterrupted_run(tmp_path: Path, case: str) -> None:
    script, args = CASES[case]
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    assert _run(script, args, manifest, straight) == 0

    (resumed / "ckpt").mkdir(parents=True)
    (resumed / "ckpt" / "STOP").touch()
    (resumed / "ckpt" / "tmpabc123.tmp").write_bytes(b"torn")  # left by a SIGKILL during a save
    assert _run(script, args, manifest, resumed) == EXIT_PAUSED
    paused = torch.load(resumed / "ckpt" / "latest.pt", weights_only=False)
    assert paused["state"] == "paused" and paused["history"]["next_epoch"] == 1
    assert not (resumed / "ckpt" / "STOP").exists()  # consumed, so the next start continues
    assert not (resumed / "ckpt" / "tmpabc123.tmp").exists()  # stale temporary files are removed at start
    progress = json.loads((resumed / "ckpt" / "progress.json").read_text())
    assert progress["state"] == "paused" and progress["next_epoch"] == 1 and progress["eta_s"] > 0
    assert _run(script, args, manifest, resumed) == 0

    a = torch.load(straight / "ckpt" / "latest.pt", weights_only=False)
    b = torch.load(resumed / "ckpt" / "latest.pt", weights_only=False)
    assert a["state"] == b["state"] == "finished" and a["global_step"] == b["global_step"] > 0
    for part in ("model_state", "cond_encoder_state", "optimizer_state"):
        ta, tb = _tensors(a[part]), _tensors(b[part])
        assert ta.keys() == tb.keys() and ta
        for k in ta:
            assert torch.equal(ta[k], tb[k]), f"{part}{k}"
    for name in ("best.pt",):
        assert all(torch.equal(x, y) for x, y in zip(
            torch.load(straight / "ckpt" / name, weights_only=False)["model_state"].values(),
            torch.load(resumed / "ckpt" / name, weights_only=False)["model_state"].values(), strict=True))  # fmt: skip

    client_a = MlflowClient(tracking_uri=f"file://{straight / 'mlruns'}")
    client_b = MlflowClient(tracking_uri=f"file://{resumed / 'mlruns'}")
    run_a = client_a.search_runs([client_a.get_experiment_by_name(STANDALONE_EXPERIMENT).experiment_id])[0]
    runs_b = client_b.search_runs([client_b.get_experiment_by_name(STANDALONE_EXPERIMENT).experiment_id])
    assert len(runs_b) == 1 and runs_b[0].data.tags["resumed_at_epoch"] == "1"
    assert runs_b[0].data.tags["state"] == "finished" and runs_b[0].info.status == "FINISHED"
    for key in ("val_loss_fm", "loss_fm"):
        ha = [(m.step, m.value) for m in client_a.get_metric_history(run_a.info.run_id, key)]
        hb = [(m.step, m.value) for m in client_b.get_metric_history(runs_b[0].info.run_id, key)]
        assert ha == hb, key  # the same values at the same steps: global_step continues


@pytest.mark.integration
def test_a_finished_run_is_not_trained_again(tmp_path: Path) -> None:
    script, args = CASES["plain"]
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=3)
    assert _run(script, args, manifest, tmp_path) == 0
    paths = ["--manifest", str(manifest), "--mlflow_uri", f"file://{tmp_path / 'mlruns'}", "--checkpoint_dir",
             str(tmp_path / "ckpt")]  # fmt: skip
    again = run_script(script, [*args, *TINY, *paths], env=CPU)
    assert again.returncode != 0 and "already finished" in again.stderr, again.stderr[-2000:]
