"""Unit tests for euler_rollout and euler_rollout_rope in fm_stepper."""

from __future__ import annotations

import torch
import pytest

from neural_pbf.integrator.fm_stepper import euler_rollout, euler_rollout_rope
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiT, VelocityDiTRoPE


def _make_batch_dit(B: int, D: int, H: int, W: int, cond_dim: int) -> dict:
    return {
        "T_in": torch.randn(B, 1, 1, D, H, W),
        "mask": torch.randn(B, 1, 1, D, H, W),
        "Q": torch.randn(B, 1, 1, D, H, W),
        "conditioning": torch.randn(B, cond_dim),
    }


def _make_batch_rope(B: int, D: int, H: int, W: int, cond_dim: int) -> dict:
    return {
        **_make_batch_dit(B, D, H, W, cond_dim),
        "patch_origin": torch.zeros(B, 3, dtype=torch.long),
    }


@pytest.mark.unit
def test_euler_rollout_output_shape() -> None:
    B, D, H, W = 2, 32, 32, 32
    cond_dim = 4
    cond_embed_dim = 32
    model = VelocityDiT(
        patch_size=8, in_channels=3, embed_dim=32, depth=1, num_heads=4,
        cond_embed_dim=cond_embed_dim, input_size=32,
    )
    cond_enc = ConditioningEncoder(cond_dim, cond_embed_dim)
    batch = _make_batch_dit(B, D, H, W, cond_dim)
    device = torch.device("cpu")
    out = euler_rollout(model, cond_enc, batch, n_steps=3, device=device)
    assert out.shape == (B, 1, D, H, W)


@pytest.mark.unit
def test_euler_rollout_returns_float32() -> None:
    B, D, H, W = 1, 32, 32, 32
    cond_embed_dim = 32
    model = VelocityDiT(
        patch_size=8, in_channels=3, embed_dim=32, depth=1, num_heads=4,
        cond_embed_dim=cond_embed_dim, input_size=32,
    )
    cond_enc = ConditioningEncoder(4, cond_embed_dim)
    batch = _make_batch_dit(B, D, H, W, 4)
    out = euler_rollout(model, cond_enc, batch, n_steps=2, device=torch.device("cpu"))
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_euler_rollout_rope_output_shape() -> None:
    B, D, H, W = 2, 32, 32, 32
    patch_size = 4
    cond_dim = 4
    cond_embed_dim = 48
    model = VelocityDiTRoPE(
        patch_size=patch_size, in_channels=3, embed_dim=48, depth=1, num_heads=8,
        cond_embed_dim=cond_embed_dim, input_size=32,
    )
    cond_enc = ConditioningEncoder(cond_dim, cond_embed_dim)
    batch = _make_batch_rope(B, D, H, W, cond_dim)
    grid_attrs = {"dz_m": 1e-5, "dy_m": 1e-5, "dx_m": 1e-5}
    out = euler_rollout_rope(
        model, cond_enc, batch, n_steps=3,
        device=torch.device("cpu"), grid_attrs=grid_attrs,
        model_patch_size=patch_size,
    )
    assert out.shape == (B, 1, D, H, W)


@pytest.mark.unit
def test_euler_rollout_rope_no_grad_in_output() -> None:
    """Rollout runs under torch.no_grad(); output must not require grad."""
    B, D, H, W = 1, 32, 32, 32
    patch_size = 4
    cond_embed_dim = 48
    model = VelocityDiTRoPE(
        patch_size=patch_size, in_channels=3, embed_dim=48, depth=1, num_heads=8,
        cond_embed_dim=cond_embed_dim, input_size=32,
    )
    cond_enc = ConditioningEncoder(4, cond_embed_dim)
    batch = _make_batch_rope(B, D, H, W, 4)
    grid_attrs = {"dz_m": 1e-5, "dy_m": 1e-5, "dx_m": 1e-5}
    out = euler_rollout_rope(
        model, cond_enc, batch, n_steps=2,
        device=torch.device("cpu"), grid_attrs=grid_attrs,
        model_patch_size=patch_size,
    )
    assert not out.requires_grad
