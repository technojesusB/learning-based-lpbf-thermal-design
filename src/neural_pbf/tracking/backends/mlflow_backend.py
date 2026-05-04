import logging
import os
from contextlib import contextmanager
from typing import Any, Literal

import mlflow

from ..base import ExperimentTracker

logger = logging.getLogger(__name__)


class MLflowTracker(ExperimentTracker):
    """MLflow implementation of ExperimentTracker."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "lpbf",
        artifact_location: str | None = None,
    ):
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "./mlruns"
        )
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.active_run: Any = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Setup experiment and store ID
        try:
            exp = mlflow.set_experiment(self.experiment_name)
            self.experiment_id = exp.experiment_id
            logger.info(
                f"MLflow tracking initialized for experiment "
                f"'{self.experiment_name}' (ID: {self.experiment_id})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to setup MLflow experiment '{self.experiment_name}': {e}"
            )
            self.experiment_id = None

    @contextmanager
    def start_run(
        self, run_name: str | None, config: dict[str, Any], tags: dict[str, str]
    ):
        # End any existing zombie runs (common in notebooks)
        if mlflow.active_run():
            mlflow.end_run()

        # Set sampling configuration before run start
        try:
            mlflow.set_system_metrics_sampling_interval(5)
        except Exception as e:
            logger.warning(f"Failed to set system metrics sampling interval: {e}")

        # Start run with explicit experiment destination
        self.active_run = mlflow.start_run(
            run_name=run_name, experiment_id=self.experiment_id, log_system_metrics=True
        )
        try:
            self.enable_system_metrics()

            if tags:
                mlflow.set_tags(tags)
            if config:
                # Only scalar values are logged as params; nested dicts/lists
                # exceed MLflow's max param length and must be logged separately
                # as artifacts by the caller.
                flat_params = {}
                for k, v in config.items():
                    if isinstance(v, dict | list):
                        logger.warning(
                            f"MLflow start_run: skipping complex config key '{k}' "
                            f"({type(v).__name__}). Log it manually as an artifact."
                        )
                    else:
                        flat_params[k] = v
                mlflow.log_params(flat_params)

            yield self
        except Exception as e:
            logger.error(f"Error during MLflow run: {e}")
            raise
        finally:
            self.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed: {e}")

    def log_text(self, text: str, artifact_path: str) -> None:
        try:
            mlflow.log_text(text, artifact_path)
        except Exception as e:
            logger.warning(f"MLflow log_text failed: {e}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed: {e}")

    def log_figure(self, fig: Any, artifact_name: str) -> None:
        """Saves plotly figure to HTML and logs to MLflow."""
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                if not artifact_name.endswith(".html"):
                    artifact_name += ".html"
                # Check if it's plotly
                if hasattr(fig, "write_html"):
                    local_path = os.path.join(tmpdir, artifact_name)
                    fig.write_html(local_path, include_plotlyjs="cdn")
                    self.log_artifact(local_path, artifact_path="plots")
                else:
                    logger.warning(
                        f"Figure type {type(fig)} not supported for log_figure "
                        "(no write_html method)."
                    )
        except Exception as e:
            logger.warning(f"MLflow log_figure failed: {e}")

    def enable_system_metrics(self) -> None:
        """Explicitly activate MLflow system-metrics logging for the active run."""
        try:
            mlflow.enable_system_metrics_logging()
        except Exception as exc:
            logger.warning("enable_system_metrics_logging failed: %s", exc)

    def flush(self) -> None:
        # MLflow python client usually handles this, specific implementation
        # might need manual flush
        pass

    def end_run(
        self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED"
    ) -> None:
        try:
            if mlflow.active_run():
                mlflow.end_run(status=status)
                self.active_run = None  # type: ignore
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")
