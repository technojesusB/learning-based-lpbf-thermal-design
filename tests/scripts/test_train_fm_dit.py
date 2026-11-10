"""Smoke tests of train_fm_dit.py: the DiT with a learned (APE) or a sinusoidal position embedding trains standalone."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.scripts.smoke_data import make_smoke_data, run_script, smoke

SCRIPT = "train_fm_dit.py"
TINY = ["--epochs", "1", "--batch_size", "2", "--embed_dim", "24", "--depth", "1", "--num_heads", "2",
        "--cond_embed_dim", "8", "--log_every", "1", "--device", "cpu", "--num_workers", "0"]  # fmt: skip


@pytest.mark.integration
def test_the_learned_embedding_trains_and_is_saved(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, ["--pos_embed", "learned"], TINY, depth=64)
    assert math.isfinite(run.data.metrics["loss_total"]) and run.data.params["fm_cfg.model"] == "VelocityDiTAPE"
    state = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=False)["model_state"]
    assert state["pos_embed"].shape == (1, (64 // 8) ** 3, 24)  # the learned embedding is saved


@pytest.mark.integration
def test_the_sinusoidal_embedding_trains(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, ["--pos_embed", "sincos"], TINY, depth=64)
    assert run.data.params["fm_cfg.model"] == "VelocityDiT"


@pytest.mark.integration
def test_the_position_embedding_must_be_chosen(tmp_path: Path) -> None:
    out = run_script(SCRIPT, ["--manifest", str(make_smoke_data(tmp_path)), *TINY])
    assert out.returncode != 0 and "pos_embed" in out.stderr, out.stderr[-2000:]
