"""ThermalSurrogate3D — UNet3D-based surrogate for LPBF thermal field prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from neural_pbf.models.config import SurrogateConfig


def _num_groups(channels: int, desired: int = 8) -> int:
    """Return the largest divisor of ``channels`` that is <= ``desired``."""
    g = min(desired, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class DoubleConv3d(nn.Module):
    """Two successive 3×3×3 conv layers each followed by GroupNorm + GELU.

    Architecture:
        Conv3d(in_ch, out_ch, 3, pad=1) → GroupNorm → GELU
        → Conv3d(out_ch, out_ch, 3, pad=1) → GroupNorm → GELU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ThermalSurrogate3D(nn.Module):
    """3D UNet-based surrogate for predicting temperature increments.

    Supports two strategies:
    - ``direct``:   concat(T, Q)        → dT;  output = T + dT
    - ``residual``: concat(T, Q, T_lf)  → delta;  output = T_lf + delta

    The network always predicts an **increment** which is then added back to
    the appropriate base field, ensuring the network only needs to learn
    corrections rather than absolute temperatures.

    Args:
        cfg: Frozen :class:`SurrogateConfig` controlling architecture.
    """

    def __init__(self, cfg: SurrogateConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # in_channels is derived from strategy by SurrogateConfig's model_validator;
        # use it directly so config and model are always consistent.
        in_ch = cfg.in_channels
        base = cfg.base_channels
        depth = cfg.depth

        # ---- Encoder --------------------------------------------------------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        enc_channels: list[int] = []
        for i in range(depth):
            out_ch = base * (2**i)
            self.encoders.append(DoubleConv3d(ch, out_ch))
            self.pools.append(nn.MaxPool3d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        # ---- Bottleneck -----------------------------------------------------
        bottleneck_ch = base * (2**depth)
        self.bottleneck = DoubleConv3d(ch, bottleneck_ch)
        ch = bottleneck_ch

        # ---- Decoder --------------------------------------------------------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            up_out_ch = skip_ch  # after transpose conv, match skip channel count
            self.upconvs.append(
                nn.ConvTranspose3d(ch, up_out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv3d(up_out_ch + skip_ch, skip_ch))
            ch = skip_ch

        # ---- Temperature output head (1×1×1 conv) --------------------------------
        # out_channels=1 always for temperature; dual-output adds a separate head.
        self.head = nn.Conv3d(ch, 1, kernel_size=1)
        # Small init so delta_norm ≈ 0 at the start → T_pred ≈ T_base.
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        # ---- Consolidation-mask output head (only when use_dual_output) ----------
        if cfg.use_dual_output:
            self.mask_head = nn.Conv3d(ch, 1, kernel_size=1)
            nn.init.xavier_uniform_(self.mask_head.weight, gain=0.01)
            if self.mask_head.bias is not None:
                nn.init.zeros_(self.mask_head.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        T: Tensor,
        Q: Tensor,
        T_lf: Tensor | None = None,
        physics_ctx: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predict next temperature field, and optionally consolidation mask logits.

        Args:
            T:           Current temperature. Shape: (B, 1, D, H, W).
            Q:           Heat source field.   Shape: (B, 1, D, H, W).
            T_lf:        Low-fidelity field.  Shape: (B, 1, D, H, W). Required when
                         ``strategy == 'residual'``.
            physics_ctx: Pre-computed physics context. Shape: (B, 5, D, H, W),
                         channels = [k_eff/k_ref, cp_eff/cp_ref, rho/rho_ref, phi,
                         mask]. Required when ``cfg.use_physics_context=True``.

        Returns:
            When ``cfg.use_dual_output=False`` (default):
                Predicted temperature field of shape (B, 1, D, H, W).
            When ``cfg.use_dual_output=True``:
                Tuple ``(T_pred, mask_logits)`` both of shape (B, 1, D, H, W).
                ``mask_logits`` are raw (pre-sigmoid) consolidation-mask predictions.
        """
        # Internal normalization — maps physical values to O(1) range for the UNet.
        # All public inputs/outputs remain in SI units (Kelvin, W/m³).
        T_hat = (T - self.cfg.T_ambient) / self.cfg.T_ref
        Q_hat = Q / self.cfg.Q_ref

        if self.cfg.strategy == "residual":
            if T_lf is None:
                raise ValueError("T_lf must be provided when strategy == 'residual'.")
            T_lf_hat = (T_lf - self.cfg.T_ambient) / self.cfg.T_ref
            x = torch.cat([T_hat, Q_hat, T_lf_hat], dim=1)
            T_base: Tensor = T_lf
        else:
            x = torch.cat([T_hat, Q_hat], dim=1)
            T_base = T

        if self.cfg.use_physics_context:
            if physics_ctx is None:
                raise ValueError(
                    "physics_ctx must be provided when use_physics_context=True."
                )
            # physics_ctx is already normalised by the caller (k/k_ref, cp/cp_ref, etc.)
            x = torch.cat([x, physics_ctx], dim=1)

        # Encoder pass — save skip connections
        skips: list[Tensor] = []
        for enc, pool in zip(self.encoders, self.pools, strict=False):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass — upsample + skip concatenation
        for upconv, dec, skip in zip(
            self.upconvs, self.decoders, reversed(skips), strict=False
        ):
            x = upconv(x)
            # Handle size mismatches from odd spatial dims via centre-crop
            if x.shape != skip.shape:
                x = _centre_crop(x, skip.shape)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Temperature head predicts a dimensionless normalised increment.
        delta_norm = self.head(x)
        T_pred = T_base + delta_norm * self.cfg.T_ref

        if self.cfg.use_dual_output:
            mask_logits = self.mask_head(x)  # raw logits, sigmoid applied by loss
            return T_pred, mask_logits

        return T_pred

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_autoregressive(
        self,
        T_init: Tensor,
        Q_sequence: list[Tensor],
        T_lf_sequence: list[Tensor] | None = None,
        physics_ctx_sequence: list[Tensor] | None = None,
        device: torch.device | None = None,
    ) -> list[Tensor] | list[tuple[Tensor, Tensor]]:
        """Run autoregressive inference over a sequence of heat sources.

        The caller is responsible for setting the model to eval mode before
        calling this method (e.g. ``model.eval()``).

        Args:
            T_init:               Initial temperature field. Shape: (1, 1, D, H, W).
            Q_sequence:           List of heat-source fields (one per step).
            T_lf_sequence:        Optional list of low-fidelity fields (one per step).
                                  Required when ``strategy == 'residual'``.
            physics_ctx_sequence: Optional list of physics-context tensors, each
                                  shape (1, 5, D, H, W). Required when
                                  ``cfg.use_physics_context=True``.
            device:               Device for inference. Defaults to CPU.

        Returns:
            When ``cfg.use_dual_output=False``:
                List of predicted temperature tensors (one per step).
            When ``cfg.use_dual_output=True``:
                List of ``(T_pred, mask_logits)`` tuples (one per step).
        """
        device = T_init.device if device is None else device
        assert next(self.parameters()).device.type == device.type, (
            f"Model is on {next(self.parameters()).device}, "
            f"but inputs are on {device}. "
            "Move the model to the same device as the inputs before calling "
            "predict_autoregressive."
        )
        T_current = T_init.to(device)
        predictions: list[Tensor] | list[tuple[Tensor, Tensor]] = []

        for step_idx, Q_step in enumerate(Q_sequence):
            Q_step = Q_step.to(device)
            T_lf: Tensor | None = None
            if T_lf_sequence is not None:
                T_lf = T_lf_sequence[step_idx].to(device)
            ctx: Tensor | None = None
            if physics_ctx_sequence is not None:
                ctx = physics_ctx_sequence[step_idx].to(device)

            result = self(T_current, Q_step, T_lf=T_lf, physics_ctx=ctx)
            # Guard: verify the return type matches cfg.use_dual_output so callers
            # that rely on isinstance(result, tuple) never silently see a mismatch.
            if self.cfg.use_dual_output != isinstance(result, tuple):
                raise TypeError(
                    "forward() returned "
                    f"{'tuple' if isinstance(result, tuple) else 'Tensor'}"
                    f" but cfg.use_dual_output={self.cfg.use_dual_output}. "
                    "This is a bug in ThermalSurrogate3D.forward."
                )
            predictions.append(result)  # type: ignore[arg-type]

            # Advance temperature for next step regardless of dual-output mode.
            T_current = result[0] if isinstance(result, tuple) else result

        return predictions


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _centre_crop(src: Tensor, target_shape: torch.Size) -> Tensor:
    """Centre-crop ``src`` to match the last three spatial dims of ``target_shape``."""
    _, _, D_t, H_t, W_t = target_shape
    _, _, D_s, H_s, W_s = src.shape

    assert all(
        s >= t for s, t in zip((D_s, H_s, W_s), (D_t, H_t, W_t), strict=True)
    ), f"_centre_crop: source {tuple(src.shape)} is smaller than target {target_shape}"

    d0 = (D_s - D_t) // 2
    h0 = (H_s - H_t) // 2
    w0 = (W_s - W_t) // 2

    return src[
        :,
        :,
        d0 : d0 + D_t,
        h0 : h0 + H_t,
        w0 : w0 + W_t,
    ]
