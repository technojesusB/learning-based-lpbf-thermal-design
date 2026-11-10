"""A training run's --seed fixes every RNG: torch, NumPy and Python's random, which ReLoBRaLo draws its lookback from.

CPU and fp32 only. On the GPU with bf16 and the Triton atomics, the same seed means the same configuration and the
same data order, not bitwise-identical weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

from tests.scripts.smoke_data import make_smoke_data, run_script

SCRIPT = "train_fm_dit_accelerate.py"
PHYSICS = ["--physics_form", "coupled", "--physics_backend", "reference"]
# batch 1: 12 steps per epoch, so 23 ReLoBRaLo draws; unseeded runs share them all with a chance of about 1 %
TINY = ["--epochs", "2", "--batch_size", "1", "--embed_dim", "24", "--depth", "1", "--num_heads", "2",
        "--cond_embed_dim", "8", "--log_every", "1", "--num_workers", "0", "--mixed_precision", "no"]  # fmt: skip


def _train(tmp_path: Path, manifest: Path, name: str, seed: int) -> tuple[list[float], dict[str, torch.Tensor]]:
    out = tmp_path / name
    argv = [*PHYSICS, *TINY, "--seed", str(seed), "--manifest", str(manifest), "--mlflow_uri", f"file://{out / 'mlruns'}",
            "--checkpoint_dir", str(out / "ckpt")]  # fmt: skip
    proc = run_script(SCRIPT, argv, env={"CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stderr[-3000:]
    client = MlflowClient(tracking_uri=f"file://{out / 'mlruns'}")
    run = client.search_runs([client.get_experiment_by_name("lpbf_experiments").experiment_id])[0]
    lam = [m.value for m in sorted(client.get_metric_history(run.info.run_id, "lambda_phys"), key=lambda m: m.step)]
    return lam, torch.load(out / "ckpt" / "latest.pt", weights_only=False)["model_state"]


@pytest.mark.integration
def test_the_same_seed_gives_the_same_physics_weights_and_model(tmp_path: Path) -> None:
    manifest = make_smoke_data(tmp_path, n_runs=8, per_run=6)
    lam_a, model_a = _train(tmp_path, manifest, "a", seed=42)
    lam_b, model_b = _train(tmp_path, manifest, "b", seed=42)
    lam_c, _ = _train(tmp_path, manifest, "c", seed=7)
    assert len(lam_a) == 24 and len(set(lam_a)) > 2  # lambda_phys moves, so the comparison means something
    assert lam_a == lam_b
    assert all(torch.equal(model_a[k], model_b[k]) for k in model_a)
    assert lam_a != lam_c
