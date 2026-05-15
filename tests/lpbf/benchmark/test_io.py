"""Tests for benchmark.io — save_metrics_csv, save_npz, mlflow_log_paths."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from neural_pbf.eval.benchmark.io import mlflow_log_paths, save_metrics_csv, save_npz


@pytest.mark.unit
def test_save_metrics_csv_creates_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    out = tmp_path / "metrics.csv"
    save_metrics_csv(df, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_save_metrics_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"Model": ["A", "B"], "IoU": [0.8, 0.9]})
    out = tmp_path / "out.csv"
    save_metrics_csv(df, out)
    loaded = pd.read_csv(out)
    assert list(loaded["Model"]) == ["A", "B"]
    assert loaded["IoU"].tolist() == pytest.approx([0.8, 0.9])


@pytest.mark.unit
def test_save_metrics_csv_creates_parent_dirs(tmp_path):
    df = pd.DataFrame({"x": [1]})
    nested = tmp_path / "deep" / "nested" / "metrics.csv"
    save_metrics_csv(df, nested)
    assert nested.exists()


@pytest.mark.unit
def test_save_npz_creates_file(tmp_path):
    out = tmp_path / "data.npz"
    save_npz(out, arr=np.array([1.0, 2.0, 3.0]))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_save_npz_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = tmp_path / "data.npz"
    save_npz(out, my_array=arr)
    loaded = np.load(out)
    assert np.allclose(loaded["my_array"], arr)


@pytest.mark.unit
def test_save_npz_multiple_arrays(tmp_path):
    out = tmp_path / "multi.npz"
    save_npz(out, a=np.ones(5), b=np.zeros(3))
    loaded = np.load(out)
    assert "a" in loaded and "b" in loaded


@pytest.mark.unit
def test_save_npz_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "data.npz"
    save_npz(out, x=np.array([1.0]))
    assert out.exists()


@pytest.mark.unit
def test_mlflow_log_paths_calls_log_artifact_for_each_path(tmp_path):
    ctx = MagicMock()
    paths = [tmp_path / "a.png", tmp_path / "b.csv"]
    for p in paths:
        p.write_bytes(b"dummy")
    mlflow_log_paths(ctx, paths, artifact_subdir="eval")
    assert ctx.log_artifact.call_count == 2


@pytest.mark.unit
def test_mlflow_log_paths_skips_missing_files(tmp_path):
    ctx = MagicMock()
    missing = tmp_path / "ghost.png"
    existing = tmp_path / "real.png"
    existing.write_bytes(b"x")
    mlflow_log_paths(ctx, [missing, existing])
    assert ctx.log_artifact.call_count == 1


@pytest.mark.unit
def test_mlflow_log_paths_empty_list(tmp_path):
    ctx = MagicMock()
    mlflow_log_paths(ctx, [])
    ctx.log_artifact.assert_not_called()
