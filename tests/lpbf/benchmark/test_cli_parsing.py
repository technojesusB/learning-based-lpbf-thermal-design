"""Tests for benchmark_suite CLI argument parsing."""
from __future__ import annotations

import pytest
import sys
from unittest.mock import patch

# Import the parser factory without triggering a full run
from experiments.benchmark_suite import _build_parser, _parse_run_id_overrides
from neural_pbf.eval.benchmark.constants import TRAINING_RUN_IDS


@pytest.mark.unit
def test_parser_default_metrics():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5"])
    assert set(args.metrics) == {"physics", "spectral", "system"}


@pytest.mark.unit
def test_parser_subset_metrics():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5", "--metrics", "physics"])
    assert args.metrics == ["physics"]


@pytest.mark.unit
def test_parser_multiple_metrics():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5", "--metrics", "physics", "spectral"])
    assert "physics" in args.metrics and "spectral" in args.metrics


@pytest.mark.unit
def test_parser_save_raw_default_false():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5"])
    assert args.save_raw is False


@pytest.mark.unit
def test_parser_save_raw_flag():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5", "--save-raw"])
    assert args.save_raw is True


@pytest.mark.unit
def test_parser_default_output_dir():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5"])
    assert "docs/assets" in str(args.output_dir)


@pytest.mark.unit
def test_parser_custom_device():
    parser = _build_parser()
    args = parser.parse_args(["--dataset", "data/test.h5", "--device", "cpu"])
    assert args.device == "cpu"


# ---------------------------------------------------------------------------
# --run-ids parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_run_ids_empty_string():
    result = _parse_run_id_overrides("")
    assert result == {}


@pytest.mark.unit
def test_parse_run_ids_none():
    result = _parse_run_id_overrides(None)
    assert result == {}


@pytest.mark.unit
def test_parse_run_ids_single():
    result = _parse_run_id_overrides("Baseline:NEWID123")
    assert result == {"Baseline": "NEWID123"}


@pytest.mark.unit
def test_parse_run_ids_multiple():
    result = _parse_run_id_overrides("Baseline:ID1,DiT v3:ID2")
    assert result == {"Baseline": "ID1", "DiT v3": "ID2"}


@pytest.mark.unit
def test_parse_run_ids_overrides_defaults():
    overrides = _parse_run_id_overrides("Baseline:NEWID")
    effective = {**TRAINING_RUN_IDS, **overrides}
    assert effective["Baseline"] == "NEWID"
    assert effective["Hero Run (v5)"] == TRAINING_RUN_IDS["Hero Run (v5)"]


@pytest.mark.unit
def test_parse_run_ids_unknown_key_accepted():
    """Unknown keys in --run-ids are accepted (forward compat)."""
    result = _parse_run_id_overrides("future_model:XYZ")
    assert result == {"future_model": "XYZ"}


@pytest.mark.unit
def test_parser_run_ids_via_cli():
    parser = _build_parser()
    args = parser.parse_args([
        "--dataset", "data/test.h5",
        "--run-ids", "Baseline:NEWID,Hero Run (v5):OTHERID",
    ])
    overrides = _parse_run_id_overrides(args.run_ids)
    assert overrides["Baseline"] == "NEWID"
    assert overrides["Hero Run (v5)"] == "OTHERID"
