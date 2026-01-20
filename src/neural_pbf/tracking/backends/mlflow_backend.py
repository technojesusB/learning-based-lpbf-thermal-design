import logging
import os
from contextlib import contextmanager
from typing import Any, Literal

import mlflow

from .base import ExperimentTracker

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
        self.active_run = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Setup experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name, artifact_location=self.artifact_location
                )
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(
                f"Failed to setup MLflow experiment '{self.experiment_name}': {e}"
            )

    @contextmanager
    def start_run(
        self, run_name: str | None, config: dict[str, Any], tags: dict[str, str]
    ):
        self.active_run = mlflow.start_run(run_name=run_name)
        try:
            if tags:
                mlflow.set_tags(tags)
            if config:
                # flatten config if needed or log as params
                # MLflow defines specific max param length, so we might need
                # to be careful with huge configs
                # For now, just log usage.
                # Nested dicts can be logged as JSON artifacts if too large,
                # but here we try simple params.
                flat_params = {}
                # Very basic flattening could be added here if needed, but
                # let's trust the user or Pydantic serialization
                for k, v in config.items():
                    if isinstance(v, (dict, list)):
                        # log complex structures as string or skip?
                        # better logging logic can be added.
                        pass
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
                self.active_run = None
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")
