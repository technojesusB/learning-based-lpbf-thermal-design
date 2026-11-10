"""Smoke tests of train_fm_dit_triton.py: the 3D-RoPE DiT in bf16 under Accelerate trains standalone."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.scripts.smoke_data import make_smoke_data, run_script, smoke

SCRIPT = "train_fm_dit_triton.py"
TINY = ["--epochs", "1", "--batch_size", "2", "--embed_dim", "24", "--depth", "1", "--num_heads", "2",
        "--cond_embed_dim", "8", "--mixed_precision", "no", "--log_every", "1", "--num_workers", "0"]  # fmt: skip


@pytest.mark.integration
def test_trains_standalone_with_workers_and_logs_every_key(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, [], [*TINY, "--num_workers", "2"])
    assert run.data.metrics["loss_phys"] == 0.0 and run.data.params["num_workers"] == "2"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the triton backend needs CUDA")
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_trains_the_physics_loss_with_the_triton_backend(tmp_path: Path, form: str) -> None:
    _, run = smoke(tmp_path, SCRIPT, ["--physics_form", form, "--physics_backend", "triton"], TINY)
    assert run.data.metrics["loss_phys"] > 0.0 and run.data.tags["preflight"] == "passed"


@pytest.mark.integration
def test_a_physics_form_needs_a_backend(tmp_path: Path) -> None:
    out = run_script(SCRIPT, ["--manifest", str(make_smoke_data(tmp_path)), "--physics_form", "coupled", *TINY])
    assert out.returncode != 0 and "needs a physics backend" in out.stderr, out.stderr[-2000:]
