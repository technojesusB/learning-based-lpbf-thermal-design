"""SurrogateConfig — frozen Pydantic configuration for the thermal surrogate model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SurrogateConfig(BaseModel):
    """
    Frozen configuration for the Dual-Strategy Streaming Surrogate.

    Attributes:
        strategy: Prediction strategy. 'direct' predicts dT from (T, Q).
                  'residual' predicts delta from (T, Q, T_lf) and adds T_lf.
        arch: Network architecture identifier.
        in_channels: Number of input channels. Derived automatically from
                     ``strategy`` and ``use_physics_context``.
        out_channels: Number of output channels. 1 normally; 2 when
                     ``use_dual_output=True`` (temperature + mask logits).
        use_physics_context: When True, five extra physics-context channels
                     (k_eff, cp_eff, rho, phi, mask) are concatenated to the
                     UNet input, enabling material-agnostic learning.
        use_dual_output: When True, the network produces a second output head
                     for consolidation-mask logits alongside the temperature
                     increment.
        base_channels: Feature channels in the first UNet encoder block.
        depth: Number of encoder downsampling levels.
        lr: Learning rate for AdamW optimiser.
        pde_weight: Scalar weight on the PDE residual loss term.
        mask_weight: Scalar weight on the mask BCE loss term (only used when
                     ``use_dual_output=True``).
        batch_size: Mini-batch size for surrogate training.
        buffer_capacity: Maximum number of patches in the experience replay buffer.
        patch_size: Spatial side-length of the cubic 3D patch stored in the buffer.
        lf_coarsen_factor: Spatial coarsening factor for the low-fidelity solver.
        lf_substep_factor: Temporal sub-step ratio (HF steps per LF step).
        k_ref: Normalisation reference for thermal conductivity [W/(m·K)].
        cp_ref: Normalisation reference for specific heat capacity [J/(kg·K)].
        rho_ref: Normalisation reference for density [kg/m³].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["direct", "residual"] = "direct"
    arch: Literal["unet3d"] = "unet3d"

    # Physics-context and dual-output flags — must be declared before the
    # validator that reads them.
    use_physics_context: bool = Field(
        default=False,
        description=(
            "Concatenate per-voxel physics context [k_eff, cp_eff, rho, phi, mask] "
            "to the UNet input. Adds 5 input channels."
        ),
    )
    use_dual_output: bool = Field(
        default=False,
        description="Add a second output head for consolidation-mask logits.",
    )

    # Channel dimensions — derived by the validator below; callers should not
    # set these directly.
    in_channels: int = Field(default=2, gt=0)
    out_channels: int = Field(default=1, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _derive_channels(cls, data: Any) -> Any:
        """Derive ``in_channels`` and ``out_channels`` from strategy/flags.

        Input channel count:
          - direct:             T + Q                = 2
          - residual:           T + Q + T_lf         = 3
          - + use_physics_context:  +5 (k_eff, cp_eff, rho, phi, mask)

        Output channel count:
          - default:            1  (temperature increment)
          - use_dual_output:    2  (temperature increment + mask logits)

        Explicit values for ``in_channels`` / ``out_channels`` must match the
        derived values; mismatches raise ``ValueError``.
        """
        if not isinstance(data, dict):
            return data

        strategy = data.get("strategy", "direct")
        use_ctx = data.get("use_physics_context", False)
        use_dual = data.get("use_dual_output", False)

        expected_in = (3 if strategy == "residual" else 2) + (5 if use_ctx else 0)
        expected_out = 2 if use_dual else 1

        provided_in = data.get("in_channels")
        if provided_in is not None and provided_in != expected_in:
            raise ValueError(
                f"in_channels={provided_in} is inconsistent with "
                f"strategy={strategy!r}, use_physics_context={use_ctx} "
                f"(expected {expected_in}). Remove in_channels or correct it."
            )
        data["in_channels"] = expected_in

        provided_out = data.get("out_channels")
        if provided_out is not None and provided_out != expected_out:
            raise ValueError(
                f"out_channels={provided_out} is inconsistent with "
                f"use_dual_output={use_dual} (expected {expected_out}). "
                "Remove out_channels or correct it."
            )
        data["out_channels"] = expected_out

        return data

    base_channels: int = Field(default=32, gt=0)
    depth: int = Field(default=4, gt=0)

    # Optimisation
    lr: float = Field(default=1e-4, gt=0.0)
    pde_weight: float = Field(default=0.1, ge=0.0)
    mask_weight: float = Field(default=1.0, ge=0.0)

    # Training loop
    batch_size: int = Field(default=4, gt=0)
    buffer_capacity: int = Field(default=2048, gt=0)

    # Spatial patch
    patch_size: int = Field(default=64, gt=0)

    # Dual-fidelity settings
    lf_coarsen_factor: int = Field(default=4, gt=0)
    lf_substep_factor: int = Field(default=1, gt=0)

    # Normalisation references (internal to model and loss — public I/O stays in SI)
    T_ref: float = Field(default=2000.0, gt=0.0)
    Q_ref: float = Field(default=1e12, gt=0.0)
    T_ambient: float = Field(default=300.0, ge=0.0)

    # Physics-context normalisation references
    k_ref: float = Field(
        default=200.0, gt=0.0, description="k_eff normalisation reference [W/(m·K)]."
    )
    cp_ref: float = Field(
        default=1000.0, gt=0.0, description="cp_eff normalisation reference [J/(kg·K)]."
    )
    rho_ref: float = Field(
        default=8500.0, gt=0.0, description="Density normalisation reference [kg/m³]."
    )
