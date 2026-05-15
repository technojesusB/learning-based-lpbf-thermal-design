"""Tests for benchmark.mlflow_data — fetch_mlflow_data."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from neural_pbf.eval.benchmark.mlflow_data import fetch_mlflow_data


def _mock_client(duration_h: float = 1.0, gpu_util: float = 80.0,
                 gpu_mem_mb: float = 8192.0, final_loss: float = 0.01,
                 s_per_it: float | None = 0.5) -> MagicMock:
    """Build a MagicMock MlflowClient that returns sensible data."""
    client = MagicMock()
    run = MagicMock()
    run.info.start_time = 0
    run.info.end_time = int(duration_h * 3_600_000)
    run.data.params = {"batch_size": "4", "n_train": "100"}
    run.data.metrics = {"val_loss": final_loss}
    client.get_run.return_value = run

    m = MagicMock(value=gpu_util, timestamp=1_000, step=1)
    m2 = MagicMock(value=gpu_util, timestamp=2_000, step=2)
    mem = MagicMock(value=gpu_mem_mb, timestamp=1_000, step=1)
    mem2 = MagicMock(value=gpu_mem_mb, timestamp=2_000, step=2)
    loss1 = MagicMock(value=0.05, timestamp=0, step=0)
    loss2 = MagicMock(value=final_loss, timestamp=int(duration_h * 3_600_000 * 0.8), step=99)

    def get_metric_side_effect(run_id, key):
        if "utilization" in key:
            return [m, m2]
        if "memory" in key:
            return [mem, mem2]
        if "train_loss" in key:
            return [loss1, loss2] if s_per_it is not None else []
        return []

    client.get_metric_history.side_effect = get_metric_side_effect
    return client


@pytest.mark.unit
def test_fetch_mlflow_data_returns_dataframe():
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient", return_value=_mock_client()):
        df = fetch_mlflow_data({"v1": "abc123"})
    assert isinstance(df, pd.DataFrame)


@pytest.mark.unit
def test_fetch_mlflow_data_schema():
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient", return_value=_mock_client()):
        df = fetch_mlflow_data({"v1": "abc123"})
    expected = {"Version", "Duration [h]", "GPU Util [%]", "GPU Mem [MB]", "s/sample", "Final Loss"}
    assert expected.issubset(set(df.columns))


@pytest.mark.unit
def test_fetch_mlflow_data_duration_populated():
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient", return_value=_mock_client(duration_h=2.0)):
        df = fetch_mlflow_data({"v1": "abc123"})
    assert df.iloc[0]["Duration [h]"] == pytest.approx(2.0, abs=0.01)


@pytest.mark.unit
def test_fetch_mlflow_data_empty_runs():
    df = fetch_mlflow_data({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.unit
def test_fetch_mlflow_data_falls_back_to_nan_on_error():
    client = MagicMock()
    client.get_run.side_effect = Exception("DB not found")
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient", return_value=client):
        df = fetch_mlflow_data({"v1": "bad_id"})
    assert len(df) == 1
    assert math.isnan(df.iloc[0]["Duration [h]"])


@pytest.mark.unit
def test_fetch_mlflow_data_multiple_runs():
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient", return_value=_mock_client()):
        df = fetch_mlflow_data({"v1": "id1", "v2": "id2", "v3": "id3"})
    assert len(df) == 3
    assert set(df["Version"]) == {"v1", "v2", "v3"}


@pytest.mark.unit
def test_fetch_mlflow_data_custom_tracking_uri_passed_to_client():
    """Ensure the tracking_uri is forwarded to MlflowClient."""
    with patch("neural_pbf.eval.benchmark.mlflow_data.MlflowClient") as MockClient:
        MockClient.return_value = _mock_client()
        fetch_mlflow_data({"v1": "id1"}, tracking_uri="sqlite:///custom.db")
    MockClient.assert_called_once_with(tracking_uri="sqlite:///custom.db")
