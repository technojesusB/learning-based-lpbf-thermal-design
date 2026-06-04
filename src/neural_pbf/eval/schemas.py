"""Pydantic schemas for evaluation configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProbeSpec(BaseModel):
    """A virtual thermocouple at a fixed voxel location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ix: int
    iy: int
    iz: int | None = None


class RolloutConfig(BaseModel):
    """Configuration for a single rollout evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_steps: int = Field(default=10, gt=0, description="Number of macro-steps to evaluate")
    mode: str = Field(default="one_step", description='"one_step" or "autoregressive"')
    divergence_T_max: float = Field(
        default=5000.0,
        description="Abort rollout if any voxel exceeds this temperature [K]",
    )
    probes: list[ProbeSpec] = Field(default_factory=list)


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    log_mlflow: bool = True
    output_dir: str = "artifacts/eval"
    experiment_name: str = "surrogate_eval"
