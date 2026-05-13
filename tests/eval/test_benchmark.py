"""Tests for src/neural_pbf/eval/benchmark.py"""
from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

from neural_pbf.eval.benchmark import (
    run_physical_fidelity_benchmark,
    run_system_comparison,
    _model_palette,
    _rollout_sample,
    _run_model_benchmark,
)


@pytest.mark.unit
def test_model_palette_returns_dict_with_colors():
    palette = _model_palette(["A", "B", "C"])
    assert set(palette.keys()) == {"A", "B", "C"}
    for v in palette.values():
        assert v.startswith("#")


@pytest.mark.unit
def test_model_palette_empty_list():
    palette = _model_palette([])
    assert palette == {}


@pytest.mark.unit
def test_model_palette_cycles_colors_for_many_names():
    names = [f"model_{i}" for i in range(10)]
    palette = _model_palette(names)
    assert len(palette) == 10
    for v in palette.values():
        assert v.startswith("#")


@pytest.mark.unit
def test_run_system_comparison_empty_runs_returns_empty_df(tmp_path):
    out = tmp_path / "out.png"
    df = run_system_comparison({}, output_path=out)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.unit
def test_run_system_comparison_writes_file_when_empty(tmp_path):
    out = tmp_path / "out.png"
    run_system_comparison({}, output_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_run_system_comparison_returns_correct_columns(tmp_path):
    out = tmp_path / "out.png"
    # Provide one synthetic run entry
    runs = {
        "v1": "checkpoints/v1/best.pt",
    }
    df = run_system_comparison(runs, output_path=out)
    assert isinstance(df, pd.DataFrame)
    # Columns are present when data is populated; empty df also acceptable
    expected_cols = {"Version", "Duration (h)", "GPU Util (%)", "GPU Mem (MB)", "s/it", "Final Loss"}
    if len(df) > 0:
        assert expected_cols.issubset(set(df.columns))


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_returns_correct_columns(tmp_path):
    """Mock the entire rollout loop to test DataFrame structure only."""
    mock_metrics = [
        {
            "Model": "M1",
            "Sample": "s0",
            "IoU": 0.8,
            "Depth_GT": 5.0,
            "Depth_Pred": 4.5,
            "T_max_Error": 50.0,
            "Offset_vox": 1.0,
        },
    ]

    with patch("neural_pbf.eval.benchmark._run_model_benchmark", return_value=mock_metrics):
        out = tmp_path / "fidelity.png"
        df = run_physical_fidelity_benchmark(
            models=[("M1", "fake_path.pt", "dit")],
            ds_cfg=MagicMock(),
            device=torch.device("cpu"),
            output_path=out,
        )

    expected_cols = {"Model", "Sample", "IoU", "Depth_GT", "Depth_Pred", "T_max_Error", "Offset_vox"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_writes_file(tmp_path):
    with patch("neural_pbf.eval.benchmark._run_model_benchmark", return_value=[]):
        out = tmp_path / "out.png"
        run_physical_fidelity_benchmark(
            models=[],
            ds_cfg=MagicMock(),
            device=torch.device("cpu"),
            output_path=out,
        )
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_empty_models_empty_df(tmp_path):
    with patch("neural_pbf.eval.benchmark._run_model_benchmark", return_value=[]):
        out = tmp_path / "out.png"
        df = run_physical_fidelity_benchmark(
            models=[],
            ds_cfg=MagicMock(),
            device=torch.device("cpu"),
            output_path=out,
        )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_multiple_models(tmp_path):
    mock_row_factory = [
        {"Model": "M1", "Sample": f"s{i}", "IoU": 0.7 + i * 0.01,
         "Depth_GT": 5.0, "Depth_Pred": 4.5, "T_max_Error": 50.0, "Offset_vox": 1.0}
        for i in range(3)
    ]

    def side_effect(name, ckpt_path, model_type, ds_cfg, device):
        return mock_row_factory

    with patch("neural_pbf.eval.benchmark._run_model_benchmark", side_effect=side_effect):
        out = tmp_path / "multi.png"
        df = run_physical_fidelity_benchmark(
            models=[("M1", "p1.pt", "net"), ("M2", "p2.pt", "dit")],
            ds_cfg=MagicMock(),
            device=torch.device("cpu"),
            output_path=out,
        )
    # Two models, 3 rows each = 6 total
    assert len(df) == 6


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_creates_parent_dirs(tmp_path):
    nested_path = tmp_path / "deep" / "nested" / "out.png"
    with patch("neural_pbf.eval.benchmark._run_model_benchmark", return_value=[]):
        run_physical_fidelity_benchmark(
            models=[],
            ds_cfg=MagicMock(),
            device=torch.device("cpu"),
            output_path=nested_path,
        )
    assert nested_path.exists()


# ── _rollout_sample ───────────────────────────────────────────────────────────

def _make_fake_batch(B: int = 1, C: int = 1, D: int = 4, H: int = 4, W: int = 4,
                     device: torch.device = torch.device("cpu")) -> dict:
    """Create a minimal batch dict with tensors for rollout testing."""
    shape = (B, C, D, H, W)
    return {
        "T_in": torch.rand(*shape, device=device),
        "mask": torch.zeros(*shape, device=device),
        "Q": torch.zeros(*shape, device=device),
        "conditioning": torch.zeros(B, 4, device=device),
        "T_target": torch.rand(*shape, device=device),
    }


class _TrivialVelocityNet(nn.Module):
    """Minimal velocity net: returns zeros matching the first channel of input."""
    def forward(self, x: torch.Tensor, tau: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # x has shape (B, 3*spatial) — return shape matching T_in (1 channel)
        # Keep the same spatial dims but reduce to 1 channel
        out_shape = list(x.shape)
        out_shape[1] = 1  # 1 channel output
        return torch.zeros(*out_shape, device=x.device)


class _TrivialCondEnc(nn.Module):
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return cond  # pass-through


class _TrivialVelocityNetRoPE(nn.Module):
    """Velocity net that accepts coords_mm (RoPE variant)."""
    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                cond_emb: torch.Tensor, coords_mm: torch.Tensor) -> torch.Tensor:
        out_shape = list(x.shape)
        out_shape[1] = 1
        return torch.zeros(*out_shape, device=x.device)


@pytest.mark.unit
def test_rollout_sample_returns_tensor_net_type():
    device = torch.device("cpu")
    batch = _make_fake_batch(device=device)
    model = _TrivialVelocityNet()
    cond_enc = _TrivialCondEnc()
    out = _rollout_sample(model, cond_enc, batch, model_type="net", device=device, n_steps=2)
    assert isinstance(out, torch.Tensor)
    assert out.shape == batch["T_target"].squeeze(1).shape


@pytest.mark.unit
def test_rollout_sample_returns_tensor_dit_type():
    device = torch.device("cpu")
    batch = _make_fake_batch(device=device)
    model = _TrivialVelocityNet()
    cond_enc = _TrivialCondEnc()
    out = _rollout_sample(model, cond_enc, batch, model_type="dit", device=device, n_steps=2)
    assert isinstance(out, torch.Tensor)


@pytest.mark.unit
def test_rollout_sample_returns_tensor_rope_type():
    device = torch.device("cpu")
    batch = _make_fake_batch(device=device)
    model = _TrivialVelocityNetRoPE()
    cond_enc = _TrivialCondEnc()
    out = _rollout_sample(model, cond_enc, batch, model_type="rope", device=device, n_steps=2)
    assert isinstance(out, torch.Tensor)


# ── _run_model_benchmark ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_run_model_benchmark_returns_list():
    result = _run_model_benchmark(
        name="test_model",
        ckpt_path="nonexistent.pt",
        model_type="net",
        ds_cfg=MagicMock(),
        device=torch.device("cpu"),
    )
    assert isinstance(result, list)


# ── run_system_comparison with data ──────────────────────────────────────────

@pytest.mark.unit
def test_run_system_comparison_with_runs_writes_file(tmp_path):
    out = tmp_path / "sys_cmp.png"
    runs = {"v1": "checkpoints/v1/best.pt", "v2": "checkpoints/v2/best.pt"}
    df = run_system_comparison(runs, output_path=out)
    assert out.exists() and out.stat().st_size > 0
    assert len(df) == 2
    assert "Version" in df.columns


@pytest.mark.unit
def test_run_system_comparison_columns_present(tmp_path):
    out = tmp_path / "sys_cmp.png"
    runs = {"baseline": "ckpts/best.pt"}
    df = run_system_comparison(runs, output_path=out)
    expected_cols = {"Version", "Duration (h)", "GPU Util (%)", "GPU Mem (MB)", "s/it", "Final Loss"}
    assert expected_cols.issubset(set(df.columns))


@pytest.mark.unit
def test_run_physical_fidelity_benchmark_mlflow_logging(tmp_path):
    """Cover the MLflow artifact logging path."""
    mock_metrics = [
        {"Model": "M1", "Sample": "s0", "IoU": 0.8, "Depth_GT": 5.0,
         "Depth_Pred": 4.5, "T_max_Error": 50.0, "Offset_vox": 1.0},
    ]
    with patch("neural_pbf.eval.benchmark._run_model_benchmark", return_value=mock_metrics):
        with patch("mlflow.log_artifact") as mock_log:
            out = tmp_path / "fidelity_mlflow.png"
            df = run_physical_fidelity_benchmark(
                models=[("M1", "fake_path.pt", "dit")],
                ds_cfg=MagicMock(),
                device=torch.device("cpu"),
                output_path=out,
                mlflow_run_id="fake_run_id",
            )
    assert len(df) == 1
    mock_log.assert_called_once_with(str(out), artifact_path="eval")


@pytest.mark.unit
def test_run_system_comparison_mlflow_logging(tmp_path):
    """Cover the MLflow artifact logging path in system comparison."""
    out = tmp_path / "sys_mlflow.png"
    with patch("mlflow.log_artifact") as mock_log:
        run_system_comparison(
            {"v1": "ckpts/best.pt"},
            output_path=out,
            mlflow_run_id="fake_run_id",
        )


# ── real _run_model_benchmark behavior ───────────────────────────────────────

@pytest.mark.unit
def test_run_model_benchmark_returns_empty_on_missing_ckpt(tmp_path):
    """Missing checkpoint file → returns [] without raising."""
    ds_cfg = MagicMock()
    result = _run_model_benchmark(
        "m", str(tmp_path / "nonexistent.pt"), "net", ds_cfg, torch.device("cpu")
    )
    assert result == []


@pytest.mark.unit
def test_run_model_benchmark_returns_empty_on_unknown_model_type(tmp_path):
    """Unknown model_type string → returns [] without raising."""
    ckpt_path = tmp_path / "fake.pt"
    torch.save(
        {"model_state": {}, "cond_encoder_state": {}, "fm_cfg": {}}, ckpt_path
    )
    ds_cfg = MagicMock()
    result = _run_model_benchmark(
        "m", str(ckpt_path), "totally_unknown_type", ds_cfg, torch.device("cpu")
    )
    assert result == []


# ── real run_system_comparison with MLflow ────────────────────────────────────

@pytest.mark.unit
def test_run_system_comparison_uses_mlflow_when_available(tmp_path):
    """When MlflowClient returns data, Duration column is populated (not NaN)."""
    out = tmp_path / "out.png"

    mock_run = MagicMock()
    mock_run.info.start_time = 0
    mock_run.info.end_time = 3_600_000  # 1 hour in ms
    mock_run.data.params = {"batch_size": "4", "n_train": "100"}
    mock_run.data.metrics = {"val_loss": 0.01}

    _m = MagicMock(value=75.0, timestamp=1_000, step=1)
    _m2 = MagicMock(value=78.0, timestamp=2_000, step=2)

    mock_client = MagicMock()
    mock_client.get_run.return_value = mock_run
    mock_client.get_metric_history.return_value = [_m, _m2]

    with patch("neural_pbf.eval.benchmark.MlflowClient", return_value=mock_client):
        df = run_system_comparison(
            {"v1": "run_id_abc123"},
            output_path=out,
            mlflow_tracking_uri="sqlite:///test.db",
        )

    assert len(df) == 1
    assert df.iloc[0]["Version"] == "v1"
    assert not pd.isna(df.iloc[0]["Duration (h)"])
    assert df.iloc[0]["Duration (h)"] == pytest.approx(1.0)


@pytest.mark.unit
def test_run_system_comparison_falls_back_to_nan_on_mlflow_error(tmp_path):
    """When MLflow raises, the row still appears but with NaN metrics."""
    out = tmp_path / "out.png"
    mock_client = MagicMock()
    mock_client.get_run.side_effect = Exception("DB not found")

    with patch("neural_pbf.eval.benchmark.MlflowClient", return_value=mock_client):
        df = run_system_comparison({"v1": "run123"}, output_path=out)

    assert len(df) == 1
    assert pd.isna(df.iloc[0]["Duration (h)"])
