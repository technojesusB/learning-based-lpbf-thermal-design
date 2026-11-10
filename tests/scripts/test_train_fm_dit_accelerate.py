"""Smoke tests of train_fm_dit_accelerate.py: the coupled physics loss with the PyTorch reference, on any device."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts.smoke_data import smoke
from tests.scripts.test_train_fm_dit_triton import TINY

SCRIPT = "train_fm_dit_accelerate.py"
PHYSICS = ["--physics_form", "coupled", "--physics_backend", "reference"]


@pytest.mark.integration
def test_trains_the_reference_loss_on_any_device(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, PHYSICS, TINY)
    assert run.data.metrics["loss_phys"] > 0.0 and run.data.tags["preflight"] == "not needed"
    assert run.data.params["physics_backend"] == "reference"
