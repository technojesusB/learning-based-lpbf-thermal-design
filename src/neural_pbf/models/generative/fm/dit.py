"""DiT (Diffusion Transformer) models for LPBF Flow Matching surrogate."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from neural_pbf.data.patch_dataset import GridAttrs


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------


def make_3d_sinusoidal_pos_embed(d: int, h: int, w: int, embed_dim: int) -> Tensor:
    """Fixed 3D sinusoidal position embeddings.

    Splits embed_dim into three equal per-axis parts (D/H/W), each rounded
    down to the nearest even number for a clean sin/cos split.  Any remainder
    is zero-padded.

    Returns:
        (1, d*h*w, embed_dim) float32 tensor for buffer registration.
    """
    dim3 = (embed_dim // 3) & ~1  # per-axis even dim
    extra = embed_dim - 3 * dim3

    def _sincos(pos: Tensor, dim: int) -> Tensor:
        half = dim // 2
        freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
        angles = pos.float().reshape(-1, 1) * freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    gd, gh, gw = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij")
    n = d * h * w
    parts = [
        _sincos(gd.reshape(-1), dim3),
        _sincos(gh.reshape(-1), dim3),
        _sincos(gw.reshape(-1), dim3),
    ]
    if extra > 0:
        parts.append(torch.zeros(n, extra))
    return torch.cat(parts, dim=-1).unsqueeze(0)  # (1, N, embed_dim)


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Fourier-feature time embedding for flow time ∈ [0, 1].

    Args:
        t:   (B,) flow time.
        dim: Output dimensionality (must be even).

    Returns:
        (B, dim) float32 Fourier features.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    denom = max(half - 1, 1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / denom)
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Physical 3D-RoPE
# ---------------------------------------------------------------------------


def apply_rope_3d(x: Tensor, coords_idx: Tensor, base: float = 10000.0) -> Tensor:
    """Apply Physical 3D Rotary Position Embedding to Q or K tensors.

    head_dim is split into three equal axis partitions (z/y/x).  Within each
    partition the standard RoPE rotation is applied using the physical
    patch-centre coordinate along that axis.

    Args:
        x:          (B, num_heads, N, head_dim).  head_dim must be divisible by 6.
        coords_idx:  (B, N, 3) patch-centre coords (z, y, x).
        base:       RoPE base frequency.

    Returns:
        (B, num_heads, N, head_dim) with 3D-RoPE applied.
    """
    B, H, N, D = x.shape
    if D % 6 != 0:
        raise ValueError(f"head_dim={D} must be divisible by 6 for 3D-RoPE")
    d_axis = D // 3
    half = d_axis // 2

    freqs = 1.0 / (base ** (torch.arange(half, dtype=torch.float32, device=x.device) / max(half, 1)))

    chunks = x.chunk(3, dim=-1)
    rotated: list[Tensor] = []

    for axis_idx, chunk in enumerate(chunks):
        coord = coords_idx[..., axis_idx]  # (B, N)
        angles = coord.unsqueeze(1).unsqueeze(-1) * freqs  # (B, 1, N, half)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        x1, x2 = chunk.chunk(2, dim=-1)
        rotated.append(torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1))

    return torch.cat(rotated, dim=-1)


def patch_center_coords_idx(
    patch_origins: Tensor,
    model_patch_size: int,
    grid_attrs: GridAttrs,
    token_grid_shape: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    """Compute patch-centre coordinates for all tokens in a batch.

    Returns discrete patch indices (not physical mm) to prevent RoPE rotation
    angles from collapsing to ~0 at the micro-scale of LPBF domains.

    Args:
        patch_origins:     (B, 3) long — [z_start, y_start, x_start] in voxels.
        model_patch_size:  DiT patch size (voxels per side).
        grid_attrs:        Output of read_grid_attrs (dz_m, dy_m, dx_m used for shape).
        token_grid_shape:  (Nz_t, Ny_t, Nx_t) — token count per spatial axis.
        device:            Compute device.

    Returns:
        (B, N_tokens, 3) float32, ordered (z_idx, y_idx, x_idx).
    """
    Nz_t, Ny_t, Nx_t = token_grid_shape

    pz = torch.arange(Nz_t, device=device, dtype=torch.float32)
    py = torch.arange(Ny_t, device=device, dtype=torch.float32)
    px = torch.arange(Nx_t, device=device, dtype=torch.float32)

    grid_z, grid_y, grid_x = torch.meshgrid(pz, py, px, indexing="ij")
    local_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)  # (N_t, 3)

    global_origin_idx = patch_origins.to(device).float() / model_patch_size  # (B, 3)

    return local_coords.unsqueeze(0) + global_origin_idx.unsqueeze(1)  # (B, N_t, 3)


# ---------------------------------------------------------------------------
# DiTBlock (adaLN-zero)
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm zero-init (DiT paper §3.3)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        _adaln_linear = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.zeros_(_adaln_linear.weight)
        nn.init.zeros_(_adaln_linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        mods = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
        shift1, scale1, gate1 = (v.unsqueeze(1) for v in (shift1, scale1, gate1))
        shift2, scale2, gate2 = (v.unsqueeze(1) for v in (shift2, scale2, gate2))

        h = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h)
        x = x + gate1 * attn_out

        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# DiTBlockRoPE (per-head Physical 3D-RoPE applied to Q and K)
# ---------------------------------------------------------------------------


class DiTBlockRoPE(nn.Module):
    """DiT block with custom multi-head attention that applies 3D-RoPE to Q and K."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        _adaln_linear = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.zeros_(_adaln_linear.weight)
        nn.init.zeros_(_adaln_linear.bias)

    def forward(self, x: Tensor, c: Tensor, coords_idx: Tensor) -> Tensor:
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim

        mods = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
        shift1, scale1, gate1 = (v.unsqueeze(1) for v in (shift1, scale1, gate1))
        shift2, scale2, gate2 = (v.unsqueeze(1) for v in (shift2, scale2, gate2))

        h = self.norm1(x) * (1.0 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = apply_rope_3d(q, coords_idx)
        k = apply_rope_3d(k, coords_idx)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + gate1 * self.out_proj(attn_out)

        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# VelocityDiT with overlapping unpatchify
# ---------------------------------------------------------------------------


class VelocityDiT(nn.Module):
    """3D patch-based Diffusion Transformer predicting the FM velocity field.

    The unpatchify step uses a two-stage approach to reduce checkerboard
    artifacts: ConvTranspose3d (coarse upsample to embed_dim//4 channels)
    followed by Conv3d with kernel_size=3, padding=1 (overlapping smoothing
    to the final single-channel output).
    """

    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_embed_dim: int = 128,
        time_emb_dim: int = 128,
        input_size: int = 64,
        overlap_unpatchify: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self._time_emb_dim = time_emb_dim
        self.overlap_unpatchify = overlap_unpatchify

        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        grid = input_size // patch_size
        self.register_buffer("pos_embed", make_3d_sinusoidal_pos_embed(grid, grid, grid, embed_dim))

        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads) for _ in range(depth)])

        if self.overlap_unpatchify:
            # Overlapping unpatchify: transposed conv to intermediate channels, then
            # Conv3d 3×3×3 (padding=1) for overlap between adjacent upsampled patches.
            self.unpatchify = nn.ConvTranspose3d(embed_dim, embed_dim // 4, kernel_size=patch_size, stride=patch_size)
            self.final_layer = nn.Conv3d(embed_dim // 4, 1, kernel_size=3, padding=1)
        else:
            # Original zero-overlap unpatchify (for backwards compatibility
            # with checkpoints)
            self.unpatchify = nn.ConvTranspose3d(embed_dim, 1, kernel_size=patch_size, stride=patch_size)
            self.final_layer = nn.Identity()

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    (B, in_channels, D, H, W) packed [x_tau, mask, Q].
            t:    (B,) flow time ∈ [0, 1].
            cond: (B, cond_embed_dim) from ConditioningEncoder.

        Returns:
            (B, 1, D, H, W) predicted velocity field.
        """
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_tokens, embed_dim)
        x = x + 0.02 * cast(Tensor, self.pos_embed)

        t_emb = sinusoidal_time_embedding(t, self._time_emb_dim)
        c = self.time_embed(t_emb) + self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c)

        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.unpatchify(x)  # (B, embed_dim//4, D_full, H_full, W_full)
        return self.final_layer(x)  # (B, 1, D_full, H_full, W_full)


# ---------------------------------------------------------------------------
# VelocityDiTRoPE — Physical 3D-RoPE variant
# ---------------------------------------------------------------------------


class VelocityDiTRoPE(VelocityDiT):
    """VelocityDiT with Physical 3D-RoPE applied inside each attention block.

    Requirements:
        embed_dim % 6 == 0   (three equal axis partitions for z/y/x)
        head_dim % 6 == 0    (same constraint at per-head level)
    """

    def __init__(self, **kwargs: Any) -> None:
        embed_dim = int(kwargs.get("embed_dim", 288))
        num_heads = int(kwargs.get("num_heads", 8))
        head_dim = embed_dim // num_heads
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 6 for 3D-RoPE")
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 6 for 3D-RoPE")
        super().__init__(**kwargs)
        # RoPE encodes positional information; the inherited sinusoidal buffer
        # is unused and wastes memory + pollutes state_dict.
        del self._buffers["pos_embed"]
        depth = len(self.blocks)
        self.blocks = nn.ModuleList([DiTBlockRoPE(embed_dim, num_heads) for _ in range(depth)])

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        t: Tensor,
        cond: Tensor,
        coords_idx: Tensor,
    ) -> Tensor:
        """
        Args:
            x:          (B, in_channels, D, H, W) packed [x_tau, mask, Q].
            t:          (B,) flow time ∈ [0, 1].
            cond:       (B, cond_embed_dim) from ConditioningEncoder.
            coords_idx:  (B, N_tokens, 3) patch-centre coordinates.

        Returns:
            (B, 1, D, H, W) predicted velocity.
        """
        x = self.patch_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        t_emb = sinusoidal_time_embedding(t, self._time_emb_dim)
        c = self.time_embed(t_emb) + self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c, coords_idx)

        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.unpatchify(x)
        return self.final_layer(x)
