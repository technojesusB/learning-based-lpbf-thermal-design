"""Every package module imports on its own in a fresh interpreter, whatever was imported before.

An import cycle can hide when another test happens to import one side first; a subprocess has no such history.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "neural_pbf.physics",
    "neural_pbf.physics.fm_physics",
    "neural_pbf.physics.material_vector",
    "neural_pbf.physics.residual_check",
    "neural_pbf.data",
    "neural_pbf.data.material_dataset",
    "neural_pbf.integrator.stepper",
    "neural_pbf.training.ablation",
    "neural_pbf.training.script_args",
    "neural_pbf.training.physics_term",
    "neural_pbf.uq.config",
    "neural_pbf.training.tracking",
    "neural_pbf.training.train_loop",
    "neural_pbf.training.train_run",
    "neural_pbf.training.rope_model",
]


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first(module: str) -> None:
    out = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
