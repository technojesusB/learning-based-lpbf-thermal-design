"""
Integration test for the diagnostics features wired into generate_offline_dataset.py.

Runs run_worker in-process on a tiny CPU-only 3D grid (no GPU / Triton required).

TDD RED phase: fails before run_worker is extracted and diagnostics are wired.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_offline_dataset.py"


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location("_gen_dataset", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def dataset_mod():
    return _load_script()


def _tiny_args(tmp_path: Path) -> argparse.Namespace:
    """Minimal Namespace for a tiny CPU-only run with diagnostics enabled."""
    out = tmp_path / "dataset.h5"
    return argparse.Namespace(
        run_index=0,
        seed=0,
        device="cpu",
        samples_per_run=2,
        nx=32,
        ny=32,
        nz=16,
        materials="SS316L",
        viz=False,
        out=str(out),
        wsl_safe=False,
        mlflow=False,
        mlflow_experiment="test-experiment",
        no_diag_csv=False,  # CSV enabled
        orchestration_id="test",
        no_triton=True,  # force pure-PyTorch path; no GPU required
    )


@pytest.mark.integration
def test_run_worker_creates_hdf5_with_samples(tmp_path, dataset_mod):
    """run_worker must write at least one sample to the HDF5 file."""
    args = _tiny_args(tmp_path)
    dataset_mod._init_hdf5(Path(args.out), args)  # type: ignore[attr-defined]
    dataset_mod.run_worker(args)  # type: ignore[attr-defined]

    import h5py

    with h5py.File(args.out, "r") as f:
        assert "samples" in f, "HDF5 missing 'samples' group"
        assert len(f["samples"]) >= 1, "No samples written"
        first_key = sorted(f["samples"].keys())[0]
        for field in ("T_in", "Q", "T_target", "T_lf", "mask"):
            assert field in f["samples"][first_key], f"Sample missing '{field}'"


@pytest.mark.integration
def test_run_worker_creates_diagnostic_csv(tmp_path, dataset_mod):
    """run_worker must create a per-step diagnostic CSV next to the HDF5 file."""
    args = _tiny_args(tmp_path)
    dataset_mod._init_hdf5(Path(args.out), args)  # type: ignore[attr-defined]
    dataset_mod.run_worker(args)  # type: ignore[attr-defined]

    csv_path = Path(args.out).with_suffix(f".run{args.run_index:03d}.diag.csv")
    assert csv_path.exists(), f"Diagnostic CSV not found at {csv_path}"


@pytest.mark.integration
def test_diagnostic_csv_has_valid_content(tmp_path, dataset_mod):
    """CSV rows must contain non-negative t_sim and realistic peak_temperature."""
    import csv

    args = _tiny_args(tmp_path)
    dataset_mod._init_hdf5(Path(args.out), args)  # type: ignore[attr-defined]
    dataset_mod.run_worker(args)  # type: ignore[attr-defined]

    csv_path = Path(args.out).with_suffix(f".run{args.run_index:03d}.diag.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1, "CSV has no data rows"

    for row in rows:
        assert float(row["t_sim"]) >= 0.0, "Negative t_sim"
        assert float(row["peak_temperature"]) >= 293.15, (
            "peak_temperature below ambient — likely uninitialized"
        )
