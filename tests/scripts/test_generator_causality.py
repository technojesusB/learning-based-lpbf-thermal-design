"""Generator causality: mask_in is pre-step, Q survives storage, no input derives from state_next."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from neural_pbf.physics.material import MaterialConfig
from neural_pbf.pipelines.dataset import prepare_trajectory

_SCRIPT = Path(__file__).parent.parent.parent / "experiments" / "generate_offline_dataset.py"
_spec = importlib.util.spec_from_file_location("generate_offline_dataset_c", _SCRIPT)
gen = importlib.util.module_from_spec(_spec)
sys.modules["generate_offline_dataset_c"] = gen
_spec.loader.exec_module(gen)  # type: ignore[union-attr]

ZOO = {
    "SS316L": MaterialConfig.ss316l_preset(),
    "Ti64": MaterialConfig.ti64_preset(),
    "IN718": MaterialConfig.in718_preset(),
}


def _args(out: Path, run_index: int = 0, **kw) -> argparse.Namespace:
    base = dict(
        out=str(out), materials="SS316L", nx=32, ny=16, nz=8, viz=False, device="cpu", seed=7,
        samples_per_run=4, wsl_safe=False, mlflow=False, mlflow_experiment="x", no_diag_csv=True,
        run_index=run_index, orchestration_id="t", no_triton=True, patterns=None,
    )
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture(scope="module")
def run_file(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("gen") / "run.h5"
    gen.run_worker(_args(out))
    return out


@pytest.mark.integration
def test_worker_creates_missing_output_file(run_file):
    with h5py.File(run_file, "r") as f:
        assert f.attrs["Nx"] == 32 and len(f["samples"]) == 4


@pytest.mark.integration
def test_q_is_stored_finite_and_in_float32(run_file):
    with h5py.File(run_file, "r") as f:
        for key in f["samples"]:
            q = f["samples"][key]["Q"]
            assert q.dtype == np.float32
            assert np.isfinite(q[()]).all() and q[()].max() > 1e6


@pytest.mark.integration
def test_mask_in_is_subset_of_post_step_mask(run_file):
    with h5py.File(run_file, "r") as f:
        for key in f["samples"]:
            s = f["samples"][key]
            pre, post = s["mask_in"][()], s["mask"][()]
            assert pre.shape == post.shape and pre.dtype == np.uint8
            assert not np.any((pre > 0) & (post == 0))


@pytest.mark.integration
def test_mask_in_empty_at_cold_start(run_file):
    with h5py.File(run_file, "r") as f:
        first = f["samples"][sorted(f["samples"])[0]]
        assert first["T_in"][()].max() < 320.0
        assert first["mask_in"][()].sum() == 0


def _do_run_assignments() -> list[tuple[list[str], set[str]]]:
    tree = ast.parse(_SCRIPT.read_text())
    do_run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_do_run")
    found = []
    for node in ast.walk(do_run):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            found.append((targets, names))
    return found


@pytest.mark.unit
def test_no_input_derives_from_state_next():
    """Source-level audit, followed transitively: state_next reaches only targets and diagnostics."""
    tainted = {"new_state", "state_next"}
    assignments = _do_run_assignments()
    for _ in assignments:  # fixed point; `state` rebinding at loop end is the next step's pre-state
        for targets, names in assignments:
            if names & tainted:
                tainted.update(t for t in targets if t != "state")
    assert tainted <= {"new_state", "state_next", "peak_temperature", "total_substeps", "T_target_np", "mask_np", "timing", "diag_line"}
    assert not tainted & {"Q_vol", "Q_np", "T_in", "T_in_np", "mask_in", "mask_in_np", "x0", "y0"}


@pytest.mark.unit
def test_mask_in_copied_before_step_in_source_order():
    source = _SCRIPT.read_text()
    assert (
        source.index("mask_in = state.material_mask.clone()")
        < source.index("new_state, total_substeps = _chunked_step")
    )


@pytest.mark.unit
def test_patterns_default_keeps_legacy_trajectory():
    a = argparse.Namespace(seed=42, materials="SS316L,Ti64,IN718", samples_per_run=50, nx=512, ny=256, nz=64)
    plan = prepare_trajectory(a, 0, ZOO)
    assert plan.pattern == "raster" and abs(plan.power - 151.30) < 0.01  # matches stored legacy run 0


@pytest.mark.unit
def test_patterns_give_factorial_design_with_unique_parameters():
    a = argparse.Namespace(
        seed=1000, materials="SS316L,Ti64,IN718", samples_per_run=5, nx=128, ny=64, nz=16,
        patterns="raster,island,zigzag",
    )
    plans = [prepare_trajectory(a, r, ZOO) for r in range(45)]
    cells: dict[tuple[str, str], int] = {}
    for p in plans:
        cells[(p.mat_key, p.pattern)] = cells.get((p.mat_key, p.pattern), 0) + 1
    assert len(cells) == 9 and set(cells.values()) == {5}
    assert len({(p.power, p.sigma, p.hatch_spacing, p.exposure_time) for p in plans}) == 45
