"""Tests for Markdown report builder and MLflow logger."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.reporting.hardware import log_gpu_telemetry, log_step_timing
from neural_pbf.eval.reporting.markdown_report import build_markdown_report, save_markdown_report
from neural_pbf.eval.reporting.mlflow_logger import log_rollout_to_mlflow
from neural_pbf.eval.rollout.engine import RolloutEngine


@pytest.fixture
def rollout_result(trajectory):
    return RolloutEngine().run(IdentityAdapter(), trajectory, mode="one_step")


# ── Markdown ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_markdown_report_contains_stepper_name(rollout_result):
    md = build_markdown_report(rollout_result)
    assert "identity" in md


@pytest.mark.unit
def test_markdown_report_contains_summary_section(rollout_result):
    md = build_markdown_report(rollout_result)
    assert "## Summary Metrics" in md


@pytest.mark.unit
def test_markdown_report_contains_per_step_section(rollout_result):
    md = build_markdown_report(rollout_result)
    assert "## Per-Step Metrics" in md


@pytest.mark.unit
def test_markdown_report_contains_mode(rollout_result):
    md = build_markdown_report(rollout_result)
    assert "one_step" in md


@pytest.mark.unit
def test_markdown_report_includes_figure_paths(tmp_path, rollout_result):
    fake = tmp_path / "some_figure.png"
    fake.touch()
    md = build_markdown_report(rollout_result, figure_paths=[fake])
    assert "some_figure" in md


@pytest.mark.unit
def test_save_markdown_report_creates_file(tmp_path, rollout_result):
    report = save_markdown_report(rollout_result, tmp_path)
    assert report.name == "report.md"
    assert report.exists() and report.stat().st_size > 0


# ── MLflow logger ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_mlflow_logger_calls_log_metrics(rollout_result):
    with patch("mlflow.log_metrics") as mock_metrics, \
         patch("mlflow.log_artifacts"):
        log_rollout_to_mlflow(rollout_result)
        assert mock_metrics.called


@pytest.mark.unit
def test_mlflow_logger_uploads_artifacts_when_dir_given(tmp_path, rollout_result):
    with patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifacts") as mock_art:
        log_rollout_to_mlflow(rollout_result, artifact_dir=tmp_path)
        assert mock_art.called


@pytest.mark.unit
def test_mlflow_logger_no_artifact_upload_when_no_dir(rollout_result):
    with patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifacts") as mock_art:
        log_rollout_to_mlflow(rollout_result, artifact_dir=None)
        assert not mock_art.called


# ── hardware reporter ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_log_step_timing_logs_metric():
    with patch("mlflow.log_metric") as mock_metric:
        log_step_timing(step=5, secs_per_iter=0.5)
        mock_metric.assert_called_once_with("Perf/steps_per_sec", pytest.approx(2.0), step=5)


@pytest.mark.unit
def test_log_step_timing_handles_zero_secs():
    with patch("mlflow.log_metric") as mock_metric:
        log_step_timing(step=0, secs_per_iter=0.0)
        assert mock_metric.called
        key, val = mock_metric.call_args[0][:2]
        assert key == "Perf/steps_per_sec"
        assert val > 0


@pytest.mark.unit
def test_log_gpu_telemetry_no_crash_without_gpu():
    # Should be a silent no-op when pynvml cannot initialise (no GPU in CI).
    log_gpu_telemetry(step=0)  # must not raise
