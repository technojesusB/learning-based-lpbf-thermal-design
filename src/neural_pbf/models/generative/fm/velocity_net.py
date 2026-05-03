"""VelocityNet: 3D U-Net that predicts the FM velocity field.

Input:  x_τ = (B, 3, Nz, Ny, Nx)  — channels: T_τ (normalised), mask, Q_norm
        τ   = (B,)                  — flow time ∈ [0, 1]
        cond = (B, cond_embed_dim)  — from ConditioningEncoder
Output: velocity = (B, 1, Nz, Ny, Nx)  — velocity for the temperature channel
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.embeddings import (
    AdaGroupNorm3d,
    sinusoidal_time_embedding,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _num_groups(channels: int) -> int:
    """Return the largest divisor of channels that is ≤ 8."""
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ConvAdaNormBlock(nn.Module):
    """Conv3d → AdaGroupNorm3d → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = AdaGroupNorm3d(_num_groups(out_ch), out_ch, cond_dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x), cond))


class _DoubleConvBlock(nn.Module):
    """Two sequential ConvAdaNorm blocks."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.block1 = _ConvAdaNormBlock(in_ch, out_ch, cond_dim)
        self.block2 = _ConvAdaNormBlock(out_ch, out_ch, cond_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.block2(self.block1(x, cond), cond)


class _EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.block = _DoubleConvBlock(in_ch, out_ch, cond_dim)
        self.down = nn.MaxPool3d(2)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.block(x, cond)
        return self.down(skip), skip


class _DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = _DoubleConvBlock(out_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        x = self.up(x)
        # Crop skip to match x if spatial dims differ (odd input sizes)
        x = _centre_crop(x, skip)
        return self.block(torch.cat([x, skip], dim=1), cond)


def _centre_crop(x: Tensor, ref: Tensor) -> Tensor:
    """Crop x spatially to match ref's (D, H, W)."""
    d, h, w = ref.shape[-3], ref.shape[-2], ref.shape[-1]
    return x[..., :d, :h, :w]


# ---------------------------------------------------------------------------
# VelocityNet
# ---------------------------------------------------------------------------


class VelocityNet(nn.Module):
    """3D U-Net with AdaGroupNorm conditioning for FM velocity prediction."""

    def __init__(self, cfg: FMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C = cfg.base_channels
        cond_dim = cfg.cond_embed_dim  # τ-emb and cond share the same dim

        # τ embedding MLP: sin/cos emb → cond_embed_dim
        self.tau_mlp = nn.Sequential(
            nn.Linear(cfg.tau_embed_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = cfg.in_channels
        self._encoder_channels: list[int] = []
        for i in range(cfg.depth):
            out_ch = C * (2**i)
            self.encoders.append(_EncoderLevel(in_ch, out_ch, cond_dim))
            self._encoder_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = C * (2**cfg.depth)
        self.bottleneck = _DoubleConvBlock(in_ch, bottleneck_ch, cond_dim)

        # Decoder
        self.decoders = nn.ModuleList()
        dec_in = bottleneck_ch
        for i in range(cfg.depth - 1, -1, -1):
            skip_ch = self._encoder_channels[i]
            dec_out = C * (2**i)
            self.decoders.append(_DecoderLevel(dec_in, skip_ch, dec_out, cond_dim))
            dec_in = dec_out

        # Output head — small xavier init so predictions ≈ 0 at init
        self.head = nn.Conv3d(dec_in, cfg.out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def forward(self, x_tau: Tensor, tau: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x_tau: (B, 3, Nz, Ny, Nx) — T_τ + mask + Q_norm
            tau:   (B,) ∈ [0, 1]
            cond:  (B, cond_embed_dim) from ConditioningEncoder

        Returns:
            (B, 1, Nz, Ny, Nx) velocity field for the temperature channel
        """
        # Fuse τ embedding additively with conditioning
        tau_emb = sinusoidal_time_embedding(tau, self.cfg.tau_embed_dim)
        tau_emb = self.tau_mlp(tau_emb)  # (B, cond_embed_dim)
        c = cond + tau_emb  # (B, cond_embed_dim)

        # Encoder
        x = x_tau
        skips: list[Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x, c)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x, c)

        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips), strict=False):
            x = dec(x, skip, c)

        return self.head(x)
