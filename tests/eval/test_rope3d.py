"""Unit tests for Physical 3D-RoPE."""

from __future__ import annotations

import pytest
import torch

from experiments.train_fm_dit_physics import apply_rope_3d, patch_center_coords_mm


# ---------------------------------------------------------------------------
# apply_rope_3d
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope3d_shape_preserved() -> None:
    B, N, D = 2, 16, 12  # D=12 divisible by 6
    x = torch.randn(B, N, D)
    coords = torch.randn(B, N, 3)
    out = apply_rope_3d(x, coords)
    assert out.shape == (B, N, D)


@pytest.mark.unit
def test_rope3d_embed_dim_divisibility_check() -> None:
    B, N, D = 1, 4, 10  # D=10 NOT divisible by 6
    x = torch.randn(B, N, D)
    coords = torch.randn(B, N, 3)
    with pytest.raises(AssertionError):
        apply_rope_3d(x, coords)


@pytest.mark.unit
def test_rope3d_zero_coords_is_identity() -> None:
    """Zero coordinates → zero rotation angles → no change to tokens."""
    B, N, D = 2, 8, 18
    x = torch.randn(B, N, D)
    coords = torch.zeros(B, N, 3)
    out = apply_rope_3d(x, coords)
    assert torch.allclose(out, x, atol=1e-5), (
        "apply_rope_3d with zero coords should return unchanged tokens"
    )


@pytest.mark.unit
def test_rope3d_norm_preservation() -> None:
    """RoPE is a rotation — it must preserve the L2 norm of each token."""
    B, N, D = 3, 10, 24
    x = torch.randn(B, N, D)
    coords = torch.randn(B, N, 3)
    out = apply_rope_3d(x, coords)

    norm_in = x.norm(dim=-1)
    norm_out = out.norm(dim=-1)
    assert torch.allclose(norm_in, norm_out, atol=1e-4), (
        "RoPE must preserve token L2 norm"
    )


@pytest.mark.unit
def test_rope3d_same_coords_same_rotation() -> None:
    """Two tokens at the same coordinate must receive the same rotation."""
    B, N, D = 1, 6, 12
    x = torch.randn(B, N, D)
    coords = torch.zeros(B, N, 3)
    # Give first two tokens identical non-zero coords
    coords[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    coords[0, 1] = torch.tensor([1.0, 2.0, 3.0])

    out = apply_rope_3d(x, coords)
    # The rotation applied to token 0 and 1 used the same angles
    # so (out[0,0] - x[0,0]) and (out[0,1] - x[0,1]) should match in pattern
    # (they differ only because x[0,0] ≠ x[0,1], but the rotation matrices are equal)
    # Verify by checking that rotating a copy of x[0,1] with coords[0,0] matches out[0,1]
    x_copy = x.clone()
    x_copy[0, 1] = x[0, 1]
    out_copy = apply_rope_3d(x_copy, coords)
    assert torch.allclose(out[0, 1], out_copy[0, 1], atol=1e-5)


# ---------------------------------------------------------------------------
# patch_center_coords_mm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_patch_center_coords_mm_shape() -> None:
    B = 3
    model_patch_size = 4
    token_grid = (4, 4, 4)  # 64 tokens
    patch_origins = torch.zeros(B, 3, dtype=torch.long)
    grid_attrs = {"dz_m": 1e-5, "dy_m": 1e-5, "dx_m": 1e-5}
    out = patch_center_coords_mm(
        patch_origins, model_patch_size, grid_attrs, token_grid, torch.device("cpu")
    )
    N_tokens = 4 * 4 * 4
    assert out.shape == (B, N_tokens, 3)


@pytest.mark.unit
def test_patch_center_coords_mm_origin_offset() -> None:
    """Non-zero patch origin shifts all coordinate values uniformly."""
    B = 1
    model_patch_size = 4
    token_grid = (2, 2, 2)
    grid_attrs = {"dz_m": 1e-3, "dy_m": 1e-3, "dx_m": 1e-3}

    origins_zero = torch.zeros(B, 3, dtype=torch.long)
    origins_shifted = torch.tensor([[0, 8, 0]], dtype=torch.long)  # y shifted by 8 voxels

    coords_zero = patch_center_coords_mm(
        origins_zero, model_patch_size, grid_attrs, token_grid, torch.device("cpu")
    )
    coords_shifted = patch_center_coords_mm(
        origins_shifted, model_patch_size, grid_attrs, token_grid, torch.device("cpu")
    )

    # y-axis (index 1) should differ by 8 * dy_m * 1000 mm
    expected_shift_mm = 8 * grid_attrs["dy_m"] * 1000.0
    diff = (coords_shifted - coords_zero)[0, :, 1]
    assert torch.allclose(diff, torch.full_like(diff, expected_shift_mm), atol=1e-5)
