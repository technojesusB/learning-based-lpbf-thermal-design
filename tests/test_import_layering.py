"""The ablation is a layer on top: no experiment script and no shared training module imports training/ablation.py.

Dependencies run one way only: ablation layer -> experiment scripts -> shared training run. A script works as if
the ablation did not exist; the queue runner selects an ablation experiment and passes the pinned arguments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p.name for p in (ROOT / "experiments").glob("train_fm_*.py"))
SHARED = ["train_run", "train_loop", "tracking", "resume", "rope_model", "script_args", "loops", "physics_term",
          "checkpointing", "relobralo"]  # fmt: skip


def _imports_ablation(code: str) -> bool:
    out = subprocess.run([sys.executable, "-c", code + "\nimport sys; print('neural_pbf.training.ablation' in sys.modules)"],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.unit
@pytest.mark.parametrize("script", SCRIPTS)
def test_an_experiment_script_does_not_import_the_ablation(script: str) -> None:
    code = (f"import importlib.util; s = importlib.util.spec_from_file_location('m', 'experiments/{script}'); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)")  # fmt: skip
    assert not _imports_ablation(code)


@pytest.mark.unit
@pytest.mark.parametrize("module", SHARED)
def test_the_shared_training_run_does_not_import_the_ablation(module: str) -> None:
    assert not _imports_ablation(f"import neural_pbf.training.{module}")
