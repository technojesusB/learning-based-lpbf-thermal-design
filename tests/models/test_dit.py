"""Unit tests for DiT model classes in neural_pbf.models.generative.fm.dit."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from neural_pbf.models.generative.fm.dit import (
    DiTBlock,
    DiTBlockRoPE,
    VelocityDiT,
    VelocityDiTRoPE,
    apply_rope_3d,
    make_3d_sinusoidal_pos_embed,
    patch_center_coords_idx,
    sinusoidal_time_embedding,
)


# ---------------------------------------------------------------------------
# make_3d_sinusoidal_pos_embed
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pos_embed_shape() -> None:
    d, h, w, dim = 8, 8, 8, 256
    out = make_3d_sinusoidal_pos_embed(d, h, w, dim)
    assert out.shape == (1, d * h * w, dim)


@pytest.mark.unit
def test_pos_embed_output_dim() -> None:
    out = make_3d_sinusoidal_pos_embed(4, 4, 4, 128)
    assert out.shape[-1] == 128


@pytest.mark.unit
def test_pos_embed_non_square_grid() -> None:
    out = make_3d_sinusoidal_pos_embed(2, 4, 8, 64)
    assert out.shape == (1, 2 * 4 * 8, 64)


# ---------------------------------------------------------------------------
# sinusoidal_time_embedding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sinusoidal_time_embedding_shape() -> None:
    t = torch.rand(8)
    out = sinusoidal_time_embedding(t, dim=64)
    assert out.shape == (8, 64)


@pytest.mark.unit
def test_sinusoidal_time_embedding_dtype() -> None:
    t = torch.rand(4)
    out = sinusoidal_time_embedding(t, dim=32)
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_sinusoidal_time_embedding_dim_must_be_even() -> None:
    t = torch.rand(2)
    with pytest.raises(AssertionError):
        sinusoidal_time_embedding(t, dim=7)


# ---------------------------------------------------------------------------
# DiTBlock (adaLN-zero)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dit_block_shape_preserved() -> None:
    B, N, D = 2, 16, 64
    block = DiTBlock(dim=D, num_heads=4)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    out = block(x, c)
    assert out.shape == (B, N, D)


@pytest.mark.unit
def test_dit_block_adaln_zero_init_is_identity() -> None:
    """At init adaLN outputs all-zeros → block acts as identity."""
    B, N, D = 2, 8, 32
    block = DiTBlock(dim=D, num_heads=4)
    block.eval()
    x = torch.randn(B, N, D)
    c = torch.zeros(B, D)
    with torch.no_grad():
        out = block(x, c)
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.unit
def test_dit_block_gradient_flows_through_condition() -> None:
    B, N, D = 2, 8, 32
    block = DiTBlock(dim=D, num_heads=4)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D, requires_grad=True)
    out = block(x, c)
    out.mean().backward()
    assert c.grad is not None
    assert c.grad.shape == c.shape


# ---------------------------------------------------------------------------
# apply_rope_3d
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope3d_shape_preserved() -> None:
    B, H, N, hd = 2, 4, 16, 12  # head_dim=12 divisible by 6
    x = torch.randn(B, H, N, hd)
    coords = torch.randn(B, N, 3)
    out = apply_rope_3d(x, coords)
    assert out.shape == (B, H, N, hd)


@pytest.mark.unit
def test_rope3d_head_dim_divisibility_check() -> None:
    B, H, N, hd = 1, 4, 4, 10  # head_dim=10 NOT divisible by 6
    x = torch.randn(B, H, N, hd)
    coords = torch.randn(B, N, 3)
    with pytest.raises(ValueError):
        apply_rope_3d(x, coords)


@pytest.mark.unit
def test_rope3d_zero_coords_is_identity() -> None:
    B, H, N, hd = 2, 4, 8, 18
    x = torch.randn(B, H, N, hd)
    coords = torch.zeros(B, N, 3)
    out = apply_rope_3d(x, coords)
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.unit
def test_rope3d_norm_preservation() -> None:
    """RoPE is a rotation — must preserve the L2 norm of each token."""
    B, H, N, hd = 3, 8, 10, 24
    x = torch.randn(B, H, N, hd)
    coords = torch.randn(B, N, 3)
    out = apply_rope_3d(x, coords)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


# ---------------------------------------------------------------------------
# patch_center_coords_idx
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_patch_center_coords_idx_shape() -> None:
    B = 3
    model_patch_size = 4
    token_grid = (4, 4, 4)
    patch_origins = torch.zeros(B, 3, dtype=torch.long)
    grid_attrs = {"dz_m": 1e-5, "dy_m": 1e-5, "dx_m": 1e-5}
    out = patch_center_coords_idx(
        patch_origins, model_patch_size, grid_attrs, token_grid, torch.device("cpu")
    )
    assert out.shape == (B, 4 * 4 * 4, 3)


@pytest.mark.unit
def test_patch_center_coords_idx_origin_offset() -> None:
    B = 1
    model_patch_size = 4
    token_grid = (2, 2, 2)
    grid_attrs = {"dz_m": 1e-3, "dy_m": 1e-3, "dx_m": 1e-3}
    origins_zero = torch.zeros(B, 3, dtype=torch.long)
    origins_shifted = torch.tensor([[0, 8, 0]], dtype=torch.long)
    coords_zero = patch_center_coords_idx(origins_zero, model_patch_size, grid_attrs, token_grid, torch.device("cpu"))
    coords_shifted = patch_center_coords_idx(origins_shifted, model_patch_size, grid_attrs, token_grid, torch.device("cpu"))
    expected_shift = 8.0 / model_patch_size
    diff = (coords_shifted - coords_zero)[0, :, 1]
    assert torch.allclose(diff, torch.full_like(diff, expected_shift), atol=1e-5)


# ---------------------------------------------------------------------------
# DiTBlockRoPE
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dit_block_rope_shape_preserved() -> None:
    # dim=48, num_heads=8 → head_dim=6 (div by 6)
    B, N, D = 2, 16, 48
    block = DiTBlockRoPE(dim=D, num_heads=8)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    coords = torch.randn(B, N, 3)
    out = block(x, c, coords)
    assert out.shape == (B, N, D)


@pytest.mark.unit
def test_dit_block_rope_gradient_flows() -> None:
    B, N, D = 1, 8, 48
    block = DiTBlockRoPE(dim=D, num_heads=8)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D, requires_grad=True)
    coords = torch.randn(B, N, 3)
    out = block(x, c, coords)
    out.mean().backward()
    assert c.grad is not None


# ---------------------------------------------------------------------------
# VelocityDiT
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_velocity_dit_forward_output_shape() -> None:
    B, D, H, W = 2, 64, 64, 64
    model = VelocityDiT(
        patch_size=8, in_channels=3, embed_dim=64,
        depth=2, num_heads=4, cond_embed_dim=128,
    )
    x = torch.randn(B, 3, D, H, W)
    t = torch.rand(B)
    cond = torch.randn(B, 128)
    out = model(x, t, cond)
    assert out.shape == (B, 1, D, H, W), f"Expected {(B,1,D,H,W)}, got {out.shape}"


@pytest.mark.unit
def test_velocity_dit_overlapping_unpatchify() -> None:
    """final_layer must be Conv3d (overlapping), unpatchify is ConvTranspose3d."""
    model = VelocityDiT(patch_size=4, in_channels=3, embed_dim=48, depth=1, num_heads=4, overlap_unpatchify=True)
    assert hasattr(model, "unpatchify"), "VelocityDiT must have .unpatchify attribute"
    assert hasattr(model, "final_layer"), "VelocityDiT must have .final_layer attribute"
    assert isinstance(model.unpatchify, nn.ConvTranspose3d)
    assert isinstance(model.final_layer, nn.Conv3d)


@pytest.mark.unit
def test_velocity_dit_no_sinusoidal_pos_embed_in_rope() -> None:
    """VelocityDiTRoPE must NOT have a pos_embed buffer (deleted in __init__)."""
    model_rope = VelocityDiTRoPE(
        patch_size=4, in_channels=3, embed_dim=48,
        depth=1, num_heads=8,
    )
    # pos_embed is deleted in VelocityDiTRoPE.__init__ — RoPE handles positional
    # information and the inherited sinusoidal buffer would waste memory/state_dict.
    assert "pos_embed" not in dict(model_rope.named_buffers())
    assert all(isinstance(b, DiTBlockRoPE) for b in model_rope.blocks)


# ---------------------------------------------------------------------------
# VelocityDiTRoPE
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_velocity_dit_rope_forward_output_shape() -> None:
    B, D, H, W = 2, 64, 64, 64
    patch_size = 4
    model = VelocityDiTRoPE(
        patch_size=patch_size, in_channels=3, embed_dim=48,
        depth=2, num_heads=8, cond_embed_dim=128,
    )
    x = torch.randn(B, 3, D, H, W)
    t = torch.rand(B)
    cond = torch.randn(B, 128)
    Nz = D // patch_size
    Ny = H // patch_size
    Nx = W // patch_size
    coords = torch.randn(B, Nz * Ny * Nx, 3)
    out = model(x, t, cond, coords)
    assert out.shape == (B, 1, D, H, W)


@pytest.mark.unit
def test_velocity_dit_rope_rejects_bad_embed_dim() -> None:
    with pytest.raises(ValueError, match="embed_dim"):
        VelocityDiTRoPE(
            patch_size=4, in_channels=3, embed_dim=64,  # 64 % 6 != 0
            depth=1, num_heads=8,
        )


@pytest.mark.unit
def test_velocity_dit_rope_blocks_are_rope_type() -> None:
    model = VelocityDiTRoPE(
        patch_size=4, in_channels=3, embed_dim=48, depth=3, num_heads=8
    )
    assert all(isinstance(b, DiTBlockRoPE) for b in model.blocks), (
        "All blocks in VelocityDiTRoPE must be DiTBlockRoPE"
    )
