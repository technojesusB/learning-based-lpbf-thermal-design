from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field

class TrackingConfig(BaseModel):
    """Configuration for experiment tracking."""
    enabled: bool = False
    backend: Literal["none", "mlflow"] = "none"
    experiment_name: str = "lpbf"
    run_name: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    log_every_n_steps: int = 50
    artifact_every_n_steps: int = 250
    artifact_dir: str = "artifacts"
    strict: bool = False
    dependency_policy: Literal["warn", "silent"] = "warn"
    mlflow_tracking_uri: Optional[str] = None  # Default local ./mlruns if None
    mlflow_artifact_location: Optional[str] = None
