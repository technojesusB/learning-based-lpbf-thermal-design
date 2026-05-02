"""Tests for MLflowTracker — full coverage via mocked mlflow module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from neural_pbf.tracking.backends.mlflow_backend import MLflowTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    tracking_uri: str = "sqlite:///test.db",
    experiment_name: str = "test-exp",
) -> MLflowTracker:
    """Build a tracker with mlflow fully mocked."""
    with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
        exp = MagicMock()
        exp.experiment_id = "exp-001"
        mock_mlflow.set_experiment.return_value = exp
        mock_mlflow.active_run.return_value = None
        tracker = MLflowTracker(
            tracking_uri=tracking_uri, experiment_name=experiment_name
        )
    return tracker


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestMLflowTrackerInit:
    def test_tracking_uri_set_on_init(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            exp = MagicMock()
            exp.experiment_id = "exp-001"
            mock_mlflow.set_experiment.return_value = exp
            MLflowTracker(tracking_uri="sqlite:///foo.db", experiment_name="test")
            mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///foo.db")

    def test_experiment_created_on_init(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            exp = MagicMock()
            exp.experiment_id = "exp-001"
            mock_mlflow.set_experiment.return_value = exp
            tracker = MLflowTracker(
                tracking_uri="sqlite:///foo.db", experiment_name="my-exp"
            )
            mock_mlflow.set_experiment.assert_called_once_with("my-exp")
            assert tracker.experiment_id == "exp-001"

    def test_experiment_id_none_on_failure(self, caplog):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.set_experiment.side_effect = Exception("conn failed")
            with caplog.at_level(logging.WARNING):
                tracker = MLflowTracker(
                    tracking_uri="sqlite:///foo.db", experiment_name="x"
                )
            assert tracker.experiment_id is None
            assert "Failed to setup MLflow experiment" in caplog.text


# ---------------------------------------------------------------------------
# start_run — zombie run cleanup
# ---------------------------------------------------------------------------


class TestStartRunZombieCleanup:
    def test_zombie_run_ended_before_new_start(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            # First active_run() call: zombie present → trigger cleanup.
            # Second call (in end_run() finally): no active run → skip.
            mock_mlflow.active_run.side_effect = [MagicMock(), None]
            mock_mlflow.start_run.return_value = MagicMock()
            with tracker.start_run("run-1", {}, {}):
                pass
            # end_run (zombie cleanup) must have been called at least once before start_run
            mlflow_calls = [c[0] for c in mock_mlflow.method_calls]
            assert "end_run" in mlflow_calls, "zombie end_run was never called"
            end_idx = mlflow_calls.index("end_run")
            start_idx = mlflow_calls.index("start_run")
            assert end_idx < start_idx, "zombie end_run must precede start_run"

    def test_no_zombie_does_not_call_end_run_before_start(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None  # no zombie
            run = MagicMock()
            mock_mlflow.start_run.return_value = run
            with tracker.start_run("run-1", {}, {}):
                pass
            end_before_start = False
            seen_start = False
            for c in mock_mlflow.method_calls:
                if c[0] == "start_run":
                    seen_start = True
                if c[0] == "end_run" and not seen_start:
                    end_before_start = True
            assert not end_before_start


# ---------------------------------------------------------------------------
# start_run — config param logging
# ---------------------------------------------------------------------------


class TestStartRunConfigLogging:
    def test_flat_params_logged(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            run = MagicMock()
            mock_mlflow.start_run.return_value = run
            config = {"lr": 1e-3, "epochs": 100, "strategy": "direct"}
            with tracker.start_run("run-1", config, {}):
                pass
            mock_mlflow.log_params.assert_called_once_with(
                {"lr": 1e-3, "epochs": 100, "strategy": "direct"}
            )

    def test_nested_dict_dropped_with_warning(self, caplog):
        tracker = _make_tracker()
        with (
            patch(
                "neural_pbf.tracking.backends.mlflow_backend.mlflow"
            ) as mock_mlflow,
            caplog.at_level(logging.WARNING),
        ):
            mock_mlflow.active_run.return_value = None
            run = MagicMock()
            mock_mlflow.start_run.return_value = run
            config = {"lr": 1e-3, "nested": {"a": 1, "b": 2}}
            with tracker.start_run("run-1", config, {}):
                pass
        assert "nested" in caplog.text

    def test_nested_list_dropped_with_warning(self, caplog):
        tracker = _make_tracker()
        with (
            patch(
                "neural_pbf.tracking.backends.mlflow_backend.mlflow"
            ) as mock_mlflow,
            caplog.at_level(logging.WARNING),
        ):
            mock_mlflow.active_run.return_value = None
            run = MagicMock()
            mock_mlflow.start_run.return_value = run
            config = {"layers": [32, 64, 128]}
            with tracker.start_run("run-1", config, {}):
                pass
        assert "layers" in caplog.text

    def test_tags_logged(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            run = MagicMock()
            mock_mlflow.start_run.return_value = run
            with tracker.start_run("run-1", {}, {"env": "test"}):
                pass
            mock_mlflow.set_tags.assert_called_once_with({"env": "test"})


# ---------------------------------------------------------------------------
# log_params / log_metrics / log_text / log_artifact
# ---------------------------------------------------------------------------


class TestLoggingMethods:
    def setup_method(self):
        self.tracker = _make_tracker()

    def test_log_params(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            self.tracker.log_params({"a": 1, "b": 2})
            mock_mlflow.log_params.assert_called_once_with({"a": 1, "b": 2})

    def test_log_params_swallows_exception(self, caplog):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.log_params.side_effect = Exception("conn lost")
            with caplog.at_level(logging.WARNING):
                self.tracker.log_params({"x": 1})
            assert "log_params failed" in caplog.text

    def test_log_metrics(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            self.tracker.log_metrics({"loss": 0.5}, step=10)
            mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_metrics_swallows_exception(self, caplog):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.log_metrics.side_effect = Exception("err")
            with caplog.at_level(logging.WARNING):
                self.tracker.log_metrics({"loss": 0.5})
            assert "log_metrics failed" in caplog.text

    def test_log_text(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            self.tracker.log_text("hello", "notes.txt")
            mock_mlflow.log_text.assert_called_once_with("hello", "notes.txt")

    def test_log_artifact(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            self.tracker.log_artifact("/tmp/model.pt", artifact_path="models")
            mock_mlflow.log_artifact.assert_called_once_with("/tmp/model.pt", "models")


# ---------------------------------------------------------------------------
# log_figure
# ---------------------------------------------------------------------------


class TestLogFigure:
    def setup_method(self):
        self.tracker = _make_tracker()

    def test_plotly_figure_saved_and_logged(self, tmp_path):
        fig = MagicMock()
        fig.write_html = MagicMock()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            self.tracker.log_figure(fig, "my_chart.html")
            fig.write_html.assert_called_once()
            mock_mlflow.log_artifact.assert_called_once()

    def test_html_extension_appended_when_missing(self):
        fig = MagicMock()
        fig.write_html = MagicMock()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow"):
            self.tracker.log_figure(fig, "my_chart")
            # write_html path must end with .html
            call_args = fig.write_html.call_args[0][0]
            assert call_args.endswith(".html")

    def test_html_extension_not_doubled(self):
        fig = MagicMock()
        fig.write_html = MagicMock()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow"):
            self.tracker.log_figure(fig, "my_chart.html")
            call_args = fig.write_html.call_args[0][0]
            assert not call_args.endswith(".html.html")

    def test_unsupported_figure_type_warns(self, caplog):
        class BadFig:
            pass

        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow"):
            with caplog.at_level(logging.WARNING):
                self.tracker.log_figure(BadFig(), "chart")
            assert "not supported" in caplog.text

    def test_write_html_exception_swallowed_with_warning(self, caplog):
        fig = MagicMock()
        fig.write_html.side_effect = OSError("disk full")
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow"):
            with caplog.at_level(logging.WARNING):
                self.tracker.log_figure(fig, "chart.html")
            assert "log_figure failed" in caplog.text


# ---------------------------------------------------------------------------
# end_run
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    def setup_method(self):
        self.tracker = _make_tracker()

    def test_log_text_swallows_exception(self, caplog):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.log_text.side_effect = Exception("err")
            with caplog.at_level(logging.WARNING):
                self.tracker.log_text("content", "file.txt")
            assert "log_text failed" in caplog.text

    def test_log_artifact_swallows_exception(self, caplog):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.log_artifact.side_effect = Exception("err")
            with caplog.at_level(logging.WARNING):
                self.tracker.log_artifact("/tmp/f.pt")
            assert "log_artifact failed" in caplog.text

    def test_flush_is_noop(self):
        self.tracker.flush()  # must not raise

    def test_system_metrics_sampling_failure_is_warned(self, caplog):
        with (
            patch(
                "neural_pbf.tracking.backends.mlflow_backend.mlflow"
            ) as mock_mlflow,
            caplog.at_level(logging.WARNING),
        ):
            mock_mlflow.active_run.return_value = None
            mock_mlflow.set_system_metrics_sampling_interval.side_effect = Exception(
                "unsupported"
            )
            mock_mlflow.start_run.return_value = MagicMock()
            with self.tracker.start_run("run", {}, {}):
                pass
        assert "system metrics" in caplog.text

    def test_exception_in_run_body_is_reraised(self):
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            mock_mlflow.start_run.return_value = MagicMock()
            with pytest.raises(ValueError, match="boom"):
                with self.tracker.start_run("run", {}, {}):
                    raise ValueError("boom")


class TestEndRun:
    def test_end_run_called_when_active(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = MagicMock()
            tracker.end_run()
            mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_end_run_not_called_when_no_active(self):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            tracker.end_run()
            mock_mlflow.end_run.assert_not_called()

    def test_end_run_swallows_exception(self, caplog):
        tracker = _make_tracker()
        with patch("neural_pbf.tracking.backends.mlflow_backend.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = MagicMock()
            mock_mlflow.end_run.side_effect = Exception("gone")
            with caplog.at_level(logging.WARNING):
                tracker.end_run()
            assert "end_run failed" in caplog.text
