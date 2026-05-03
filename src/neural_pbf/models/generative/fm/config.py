"""FMConfig — frozen Pydantic configuration for the Flow Matching surrogate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FMConfig(BaseModel):
    """Frozen configuration for the Flow Matching velocity-field surrogate.

    Input to VelocityNet is 3 spatial channels: (T_τ, mask, Q_norm).
    Output is 1 channel: velocity field for the temperature.
    Conditioning is injected via ConditioningEncoder + sinusoidal τ embedding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arch: Literal["unet3d_fm"] = "unet3d_fm"

    # Spatial channels: T_τ + mask + Q = 3 input; T velocity = 1 output
    in_channels: int = Field(default=3, gt=0)
    out_channels: int = Field(default=1, gt=0)

    base_channels: int = Field(default=32, gt=0)
    depth: int = Field(default=3, gt=0)

    # Conditioning dimensions
    cond_dim: int = Field(default=12, gt=0)
    cond_embed_dim: int = Field(default=128, gt=0)
    tau_embed_dim: int = Field(default=128, gt=0)

    # Normalisation references (must match FMDatasetConfig)
    T_ref: float = Field(default=2000.0, gt=0.0)
    T_ambient: float = Field(default=300.0, ge=0.0)

    # Inference
    n_inference_steps: int = Field(default=25, gt=0)
    sigma_min: float = Field(default=1e-4, gt=0.0)

    @model_validator(mode="after")
    def _validate_embed_dims(self) -> FMConfig:
        if self.cond_embed_dim != self.tau_embed_dim:
            raise ValueError(
                f"cond_embed_dim ({self.cond_embed_dim}) must equal "
                f"tau_embed_dim ({self.tau_embed_dim}) so that additive fusion works."
            )
        return self
