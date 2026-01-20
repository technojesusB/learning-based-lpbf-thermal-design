from typing import Literal

from pydantic import BaseModel, Field


class DiagnosticsConfig(BaseModel):
    """Configuration for simulation diagnostics and health checks."""

    enabled: bool = True
    level: Literal["off", "basic", "verbose"] = "basic"
    log_every_n_steps: int = 50
    snapshot_every_n_steps: int = 250
    check_nan_inf: bool = True
    check_ranges: bool = True
    perf_profile: bool = True
    memory_profile: bool = True
    strict: bool = False
    thresholds: dict[str, float] = Field(default_factory=dict)
