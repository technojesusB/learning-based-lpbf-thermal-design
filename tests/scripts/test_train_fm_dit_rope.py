"""Smoke tests of train_fm_dit_rope.py: the 3D-RoPE DiT trains standalone in fp32 without Accelerate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts.smoke_data import make_smoke_data, run_script, smoke
from tests.scripts.test_train_fm_dit_triton import TINY

SCRIPT = "train_fm_dit_rope.py"
TINY_FP32 = [a for a in TINY if a not in ("--mixed_precision", "no")] + ["--device", "cpu"]


@pytest.mark.integration
def test_trains_standalone_without_accelerate(tmp_path: Path) -> None:
    _, run = smoke(tmp_path, SCRIPT, [], TINY_FP32)
    assert run.data.params["runtime.accelerate"] == "False" and run.data.metrics["loss_phys"] == 0.0


@pytest.mark.integration
def test_has_no_precision_switch(tmp_path: Path) -> None:
    out = run_script(SCRIPT, ["--manifest", str(make_smoke_data(tmp_path)), "--mixed_precision", "bf16"])
    assert out.returncode != 0 and "mixed_precision" in out.stderr, out.stderr[-2000:]
