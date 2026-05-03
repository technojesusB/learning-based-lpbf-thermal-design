"""
Tests for neural_pbf.data.sampling module.

TDD Workflow: Tests written FIRST (RED phase) before implementation.
"""
from __future__ import annotations

import math
import random

import pytest

# ── Unit tests ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_heuristic_sample_indices_default_count():
    """Default call with n_steps=10000 must return exactly 50 indices."""
    from neural_pbf.data.sampling import heuristic_sample_indices

    result = heuristic_sample_indices(10000)
    assert len(result) == 50


@pytest.mark.unit
def test_heuristic_sample_indices_returns_set():
    """Result must be a plain Python set (for O(1) membership tests)."""
    from neural_pbf.data.sampling import heuristic_sample_indices

    result = heuristic_sample_indices(10000)
    assert isinstance(result, set)


@pytest.mark.unit
def test_heuristic_sample_indices_distribution():
    """
    The 10/30/10 distribution must be approximately respected.

    Windows for n_steps=10000:
      early:  [0, 500)
      mid:    [500, 9000)
      late:   [9000, 10000)

    We allow duplicates to land in adjacent windows (set deduplication
    may shift a few indices), so check >= minimum expected rather than exact.
    """
    from neural_pbf.data.sampling import heuristic_sample_indices

    n_steps = 10000
    result = heuristic_sample_indices(n_steps)

    early_count = sum(1 for i in result if 0 <= i < 500)
    mid_count = sum(1 for i in result if 500 <= i < 9000)
    late_count = sum(1 for i in result if 9000 <= i < n_steps)

    # At least half the expected count in each window
    assert early_count >= 5, f"Expected >= 5 early samples, got {early_count}"
    assert mid_count >= 15, f"Expected >= 15 mid samples, got {mid_count}"
    assert late_count >= 5, f"Expected >= 5 late samples, got {late_count}"


@pytest.mark.unit
def test_heuristic_sample_indices_all_in_range():
    """All returned indices must be in [0, n_steps)."""
    from neural_pbf.data.sampling import heuristic_sample_indices

    n_steps = 10000
    result = heuristic_sample_indices(n_steps)
    assert all(0 <= i < n_steps for i in result), "Some indices are out of range"


@pytest.mark.unit
def test_heuristic_sample_indices_small_n():
    """
    When n_steps is just >= n_samples (60 >= 50), the function must not crash
    and return at least 1 valid index.
    """
    from neural_pbf.data.sampling import heuristic_sample_indices

    result = heuristic_sample_indices(60, n_samples=50)
    assert len(result) >= 1
    assert all(0 <= i < 60 for i in result)


@pytest.mark.unit
def test_heuristic_sample_indices_too_few_steps():
    """When n_steps < n_samples, the function returns all available indices."""
    from neural_pbf.data.sampling import heuristic_sample_indices

    result = heuristic_sample_indices(10, n_samples=50)
    assert result == set(range(10))


@pytest.mark.unit
def test_heuristic_sample_indices_non_default_samples():
    """n_samples=10 must return exactly 10 unique indices for large n_steps."""
    from neural_pbf.data.sampling import heuristic_sample_indices

    result = heuristic_sample_indices(10000, n_samples=10)
    assert len(result) == 10


# ── randomize_material tests ──────────────────────────────────────────────────


@pytest.mark.unit
def test_randomize_material_preserves_T_lut():
    """
    T_lut must be IDENTICAL (not just close) before and after randomization.
    Run 20 times to catch any randomness that touches the temperature axis.
    """
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    mat = MaterialConfig.ss316l_preset()
    original_T_lut = list(mat.T_lut)  # type: ignore[arg-type]

    for _ in range(20):
        perturbed = randomize_material(mat)
        assert perturbed.T_lut == original_T_lut, (
            f"T_lut was modified: {perturbed.T_lut} != {original_T_lut}"
        )


@pytest.mark.unit
def test_randomize_material_perturbs_k_lut():
    """
    k_lut values must differ from the originals (perturbation applied) and
    each perturbed value must remain within the ±scale*original range.
    """
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    import random

    random.seed(0)
    mat = MaterialConfig.ss316l_preset()
    original_k_lut = list(mat.k_lut)  # type: ignore[arg-type]
    scale = 0.1

    # Run several times; at least one run should differ
    changed = False
    for _ in range(10):
        perturbed = randomize_material(mat, scale=scale)
        assert perturbed.k_lut is not None
        for orig, new in zip(original_k_lut, perturbed.k_lut):
            assert orig * (1 - scale) <= new <= orig * (1 + scale), (
                f"k_lut value {new} out of perturb bounds [{orig*(1-scale)}, {orig*(1+scale)}]"
            )
        if perturbed.k_lut != original_k_lut:
            changed = True

    assert changed, "k_lut was never changed across 10 calls — perturbation not applied"


@pytest.mark.unit
def test_randomize_material_perturbs_cp_lut():
    """cp_lut values must be perturbed within ±scale bounds."""
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    import random

    random.seed(1)
    mat = MaterialConfig.ss316l_preset()
    original_cp_lut = list(mat.cp_lut)  # type: ignore[arg-type]
    scale = 0.1

    changed = False
    for _ in range(10):
        perturbed = randomize_material(mat, scale=scale)
        assert perturbed.cp_lut is not None
        for orig, new in zip(original_cp_lut, perturbed.cp_lut):
            assert orig * (1 - scale) <= new <= orig * (1 + scale), (
                f"cp_lut value {new} out of perturb bounds [{orig*(1-scale)}, {orig*(1+scale)}]"
            )
        if perturbed.cp_lut != original_cp_lut:
            changed = True

    assert changed, "cp_lut was never changed across 10 calls — perturbation not applied"


@pytest.mark.unit
def test_randomize_material_preserves_solidus_liquidus_order():
    """After perturbation T_liquidus must remain strictly greater than T_solidus."""
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    mat = MaterialConfig.ss316l_preset()
    for _ in range(50):
        perturbed = randomize_material(mat)
        assert perturbed.T_liquidus > perturbed.T_solidus, (
            f"T_liquidus ({perturbed.T_liquidus}) <= T_solidus ({perturbed.T_solidus})"
        )


@pytest.mark.unit
def test_randomize_material_returns_valid_model():
    """
    Core physical quantities must stay positive after perturbation:
    k_powder > 0, rho > 0, latent_heat_L > 0.
    """
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    mat = MaterialConfig.ss316l_preset()
    for _ in range(20):
        perturbed = randomize_material(mat)
        assert perturbed.k_powder > 0, "k_powder became non-positive"
        assert perturbed.rho > 0, "rho became non-positive"
        assert perturbed.latent_heat_L > 0, "latent_heat_L became non-positive"


@pytest.mark.unit
def test_randomize_material_none_lut_material():
    """
    When use_lut=False (k_lut and cp_lut are None), the function must not
    crash and must leave k_lut / cp_lut as None.
    """
    from neural_pbf.data.sampling import randomize_material
    from neural_pbf.physics.material import MaterialConfig

    mat = MaterialConfig(
        k_powder=0.2,
        k_solid=14.0,
        k_liquid=30.0,
        cp_base=500.0,
        rho=8000.0,
        T_solidus=1600.0,
        T_liquidus=1650.0,
        latent_heat_L=2.8e5,
        use_lut=False,
    )
    perturbed = randomize_material(mat)
    assert perturbed.k_lut is None
    assert perturbed.cp_lut is None


# ── _topup_sample_indices tests (Phase 5) ─────────────────────────────────────


@pytest.mark.unit
def test_topup_no_op_when_full():
    """When sample_indices already has n_samples entries, returns same set."""
    from neural_pbf.data.sampling import _topup_sample_indices

    full_set = set(range(50))
    result = _topup_sample_indices(full_set, n_steps=1000, n_samples=50)
    assert result == full_set
    assert len(result) == 50


@pytest.mark.unit
def test_topup_fills_to_target():
    """When sample_indices has 45 entries and n_steps=1000, result has exactly 50."""
    from neural_pbf.data.sampling import _topup_sample_indices

    partial = set(range(45))
    rng = random.Random(0)
    result = _topup_sample_indices(partial, n_steps=1000, n_samples=50, rng=rng)
    assert len(result) == 50
    # Original entries are preserved
    assert partial.issubset(result)


@pytest.mark.unit
def test_topup_deterministic_with_seed():
    """Same RNG seed produces the same top-up result."""
    from neural_pbf.data.sampling import _topup_sample_indices

    partial = set(range(45))
    result_a = _topup_sample_indices(partial, n_steps=1000, n_samples=50, rng=random.Random(99))
    result_b = _topup_sample_indices(partial, n_steps=1000, n_samples=50, rng=random.Random(99))
    assert result_a == result_b


@pytest.mark.unit
def test_topup_graceful_when_impossible():
    """
    When n_steps=48 and all 48 indices are already in sample_indices,
    there is nothing to top up to 50 — returns original set unchanged.
    """
    from neural_pbf.data.sampling import _topup_sample_indices

    full_short = set(range(48))
    result = _topup_sample_indices(full_short, n_steps=48, n_samples=50)
    assert result == full_short
    assert len(result) == 48


@pytest.mark.unit
def test_topup_no_out_of_range():
    """All returned indices must be in [0, n_steps)."""
    from neural_pbf.data.sampling import _topup_sample_indices

    partial = set(range(30))
    rng = random.Random(7)
    result = _topup_sample_indices(partial, n_steps=200, n_samples=50, rng=rng)
    assert all(0 <= i < 200 for i in result), "Some indices are out of [0, n_steps) range"


@pytest.mark.unit
def test_topup_does_not_mutate_input():
    """The input set must not be modified in place."""
    from neural_pbf.data.sampling import _topup_sample_indices

    partial = set(range(45))
    original_copy = set(partial)
    _topup_sample_indices(partial, n_steps=1000, n_samples=50, rng=random.Random(0))
    assert partial == original_copy, "Input set was mutated"


# ── Integration / smoke test (GPU required) ───────────────────────────────────


@pytest.mark.slow
def test_generate_offline_dataset_smoke(tmp_path):
    """
    End-to-end smoke test: run the script with a tiny grid for 1 run and
    verify the HDF5 output has the expected structure.

    Skipped if CUDA is not available.
    """
    import subprocess
    import sys
    from pathlib import Path

    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU required for LPBF simulation")
    except ImportError:
        pytest.skip("torch not available")

    script = Path(__file__).parent.parent / "scripts" / "generate_offline_dataset.py"
    out_file = tmp_path / "test_dataset.h5"

    # --samples-per-run 5: tiny grid (32x16) generates ~48 path points.
    # The Phase 5 top-up guarantees exactly samples_per_run samples are stored
    # as long as the path has at least that many steps.
    samples_per_run = 5
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs",
            "1",
            "--nx",
            "32",
            "--ny",
            "16",
            "--nz",
            "8",
            "--samples-per-run",
            str(samples_per_run),
            "--out",
            str(out_file),
            "--device",
            "cuda",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Script failed with returncode {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    import h5py

    with h5py.File(out_file, "r") as f:
        assert "samples" in f, "HDF5 file missing 'samples' group"
        n_samples = len(f["samples"])
        assert n_samples == samples_per_run, (
            f"Expected {samples_per_run} samples (1 run × {samples_per_run}), got {n_samples}"
        )

        sample = f["samples"]["sample_000000"]
        for key in ["T_in", "Q", "T_target", "T_lf", "mask"]:
            assert key in sample, f"Sample missing dataset '{key}'"

        assert "mat_name" in sample.attrs, "Sample missing 'mat_name' attribute"
        assert "k_s" in sample.attrs, "Sample missing 'k_s' attribute"
        assert "exposure_time" in sample.attrs, "Sample missing 'exposure_time' attribute"

        # Top-level metadata written on first write
        assert "Nx" in f.attrs, "HDF5 file missing top-level 'Nx' attribute"
