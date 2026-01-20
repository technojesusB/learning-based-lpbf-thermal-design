import logging

from ..schemas.tracking import TrackingConfig
from .base import ExperimentTracker, NullTracker

logger = logging.getLogger(__name__)


def build_tracker(cfg: TrackingConfig) -> ExperimentTracker:
    """Factory to create an experiment tracker based on configuration."""
    if not cfg.enabled or cfg.backend == "none":
        return NullTracker()

    if cfg.backend == "mlflow":
        try:
            import mlflow  # noqa: F401

            from .backends.mlflow_backend import MLflowTracker

            return MLflowTracker(
                tracking_uri=cfg.mlflow_tracking_uri,
                experiment_name=cfg.experiment_name,
                artifact_location=cfg.mlflow_artifact_location,
            )
        except ImportError as e:
            msg = "MLflow backend requested but 'mlflow' is not installed."
            if cfg.strict:
                raise ImportError(msg) from e

            if cfg.dependency_policy == "warn":
                logger.warning(f"{msg} Falling back to NullTracker.")
            return NullTracker()
        except Exception as e:
            msg = f"Failed to initialize MLflow backend: {e}"
            if cfg.strict:
                raise RuntimeError(msg) from e

            if cfg.dependency_policy == "warn":
                logger.warning(f"{msg} Falling back to NullTracker.")
            return NullTracker()

    return NullTracker()
