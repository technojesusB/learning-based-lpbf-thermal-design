"""Tests for benchmark.constants — all sacred values must be exact."""
from __future__ import annotations

import pytest

from neural_pbf.eval.benchmark.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_EULER_STEPS,
    PATCH_SIZE,
    T_AMBIENT,
    T_LIQUIDUS,
    T_REF,
    TRAINING_RUN_IDS,
)


@pytest.mark.unit
def test_t_liquidus_is_1600():
    assert T_LIQUIDUS == 1600.0


@pytest.mark.unit
def test_t_ref_is_2000():
    assert T_REF == 2000.0


@pytest.mark.unit
def test_t_ambient_is_300():
    assert T_AMBIENT == 300.0


@pytest.mark.unit
def test_n_euler_steps_is_25():
    assert N_EULER_STEPS == 25


@pytest.mark.unit
def test_patch_size_is_64():
    assert PATCH_SIZE == 64


@pytest.mark.unit
def test_mlflow_experiment_name():
    assert MLFLOW_EXPERIMENT_NAME == "Model-Evaluation"


@pytest.mark.unit
def test_mlflow_tracking_uri_default():
    assert MLFLOW_TRACKING_URI == "sqlite:///mlflow.db"


@pytest.mark.unit
def test_training_run_ids_contains_all_models():
    expected_keys = {"Baseline", "DiT v1", "DiT v2", "DiT v3", "DiT v4", "Hero Run (v5)"}
    assert expected_keys == set(TRAINING_RUN_IDS.keys())


@pytest.mark.unit
def test_training_run_ids_are_strings():
    for k, v in TRAINING_RUN_IDS.items():
        assert isinstance(k, str), f"Key {k!r} is not str"
        assert isinstance(v, str), f"Value for {k!r} is not str"
        assert len(v) == 32, f"Run ID for {k!r} should be 32 hex chars, got {len(v)}"


@pytest.mark.unit
def test_training_run_ids_hex_format():
    """Each run ID must be a 32-char lowercase hex string (MLflow UUID without dashes)."""
    for k, v in TRAINING_RUN_IDS.items():
        assert len(v) == 32, f"Run ID for {k!r} should be 32 chars, got {len(v)}"
        assert v == v.lower(), f"Run ID for {k!r} should be lowercase"
        assert all(c in "0123456789abcdef" for c in v), f"Run ID for {k!r} is not hex"


@pytest.mark.unit
def test_normalization_formula():
    """T_phys = T_norm * T_REF + T_AMBIENT must hold for representative values."""
    T_norm = 0.5
    T_phys_expected = T_norm * T_REF + T_AMBIENT
    assert T_phys_expected == 1300.0


@pytest.mark.unit
def test_constants_are_immutable_types():
    """Float and int constants cannot be accidentally mutated."""
    assert isinstance(T_LIQUIDUS, float)
    assert isinstance(T_REF, float)
    assert isinstance(T_AMBIENT, float)
    assert isinstance(N_EULER_STEPS, int)
    assert isinstance(PATCH_SIZE, int)
