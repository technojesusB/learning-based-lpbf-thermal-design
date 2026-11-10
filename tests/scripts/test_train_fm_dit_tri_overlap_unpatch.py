"""Smoke tests of train_fm_dit_tri_overlap_unpatch.py: the DiT trains with overlapping unpatchify."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.scripts.smoke_data import smoke
from tests.scripts.test_train_fm_dit_triton import TINY

SCRIPT = "train_fm_dit_tri_overlap_unpatch.py"


@pytest.mark.integration
def test_trains_with_overlapping_unpatchify(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, [], TINY)
    assert run.data.params["fm_cfg.overlap_unpatchify"] == "True"
    state = torch.load(tmp_path / "ckpt" / "latest.pt", weights_only=False)["model_state"]
    assert "final_layer.weight" in state  # the 3x3x3 overlap convolution


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the triton backend needs CUDA")
def test_trains_the_coupled_loss_with_the_triton_backend(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, ["--physics_form", "coupled", "--physics_backend", "triton"], TINY)
    assert run.data.metrics["loss_phys"] > 0.0
