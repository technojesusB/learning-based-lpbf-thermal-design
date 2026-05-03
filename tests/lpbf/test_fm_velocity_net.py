"""Tests for FMConfig, ConditioningEncoder, and VelocityNet.

All run on CPU. Tiny grids (8×4×4) for speed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.velocity_net import VelocityNet


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NZ, NY, NX = 8, 4, 4
BATCH = 2
COND_DIM = 12


@pytest.fixture()
def small_cfg() -> FMConfig:
    return FMConfig(
        base_channels=8,
        depth=2,
        cond_dim=COND_DIM,
        cond_embed_dim=32,
        tau_embed_dim=32,
    )


@pytest.fixture()
def encoder(small_cfg: FMConfig) -> ConditioningEncoder:
    return ConditioningEncoder(small_cfg.cond_dim, small_cfg.cond_embed_dim)


@pytest.fixture()
def velocity_net(small_cfg: FMConfig) -> VelocityNet:
    return VelocityNet(small_cfg)


def _make_x_tau(B: int = BATCH) -> torch.Tensor:
    """3-channel input: (T_τ, mask, Q_norm) — shape (B, 3, Nz, Ny, Nx)."""
    return torch.randn(B, 3, NZ, NY, NX)


def _make_tau(B: int = BATCH) -> torch.Tensor:
    return torch.rand(B)


def _make_cond(B: int = BATCH, embed_dim: int = 32) -> torch.Tensor:
    return torch.randn(B, embed_dim)


def _make_scalars(B: int = BATCH) -> torch.Tensor:
    return torch.randn(B, COND_DIM)


# ---------------------------------------------------------------------------
# FMConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fm_config_frozen() -> None:
    cfg = FMConfig()
    with pytest.raises(Exception):
        cfg.base_channels = 999  # type: ignore[misc]


@pytest.mark.unit
def test_fm_config_default_in_channels() -> None:
    cfg = FMConfig()
    assert cfg.in_channels == 3, "Default in_channels must be 3 (T, mask, Q)"


@pytest.mark.unit
def test_fm_config_default_out_channels() -> None:
    cfg = FMConfig()
    assert cfg.out_channels == 1


@pytest.mark.unit
def test_fm_config_default_cond_dim() -> None:
    cfg = FMConfig()
    assert cfg.cond_dim == 12


@pytest.mark.unit
def test_fm_config_embed_dims_equal_by_default() -> None:
    cfg = FMConfig()
    assert cfg.cond_embed_dim == cfg.tau_embed_dim


# ---------------------------------------------------------------------------
# ConditioningEncoder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conditioning_encoder_is_nn_module(encoder: ConditioningEncoder) -> None:
    assert isinstance(encoder, nn.Module)


@pytest.mark.unit
def test_conditioning_encoder_output_shape(encoder: ConditioningEncoder, small_cfg: FMConfig) -> None:
    scalars = _make_scalars()
    out = encoder(scalars)
    assert out.shape == (BATCH, small_cfg.cond_embed_dim), (
        f"Expected ({BATCH}, {small_cfg.cond_embed_dim}), got {tuple(out.shape)}"
    )


@pytest.mark.unit
def test_conditioning_encoder_gradient_flow(encoder: ConditioningEncoder) -> None:
    scalars = _make_scalars()
    out = encoder(scalars)
    loss = out.sum()
    loss.backward()
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"Param {name} has no gradient"


@pytest.mark.unit
def test_conditioning_encoder_deterministic(encoder: ConditioningEncoder) -> None:
    encoder.eval()
    scalars = _make_scalars()
    with torch.no_grad():
        out1 = encoder(scalars)
        out2 = encoder(scalars)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# VelocityNet — forward shape contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_velocity_net_is_nn_module(velocity_net: VelocityNet) -> None:
    assert isinstance(velocity_net, nn.Module)


@pytest.mark.unit
def test_velocity_net_output_shape(velocity_net: VelocityNet, small_cfg: FMConfig) -> None:
    x_tau = _make_x_tau()
    tau = _make_tau()
    cond = _make_cond(embed_dim=small_cfg.cond_embed_dim)
    out = velocity_net(x_tau, tau, cond)
    expected = (BATCH, 1, NZ, NY, NX)
    assert tuple(out.shape) == expected, f"Expected {expected}, got {tuple(out.shape)}"


@pytest.mark.unit
def test_velocity_net_output_shape_single_item(velocity_net: VelocityNet, small_cfg: FMConfig) -> None:
    x_tau = _make_x_tau(B=1)
    tau = _make_tau(B=1)
    cond = _make_cond(B=1, embed_dim=small_cfg.cond_embed_dim)
    out = velocity_net(x_tau, tau, cond)
    assert tuple(out.shape) == (1, 1, NZ, NY, NX)


@pytest.mark.unit
def test_velocity_net_gradient_flow(velocity_net: VelocityNet, small_cfg: FMConfig) -> None:
    x_tau = _make_x_tau()
    tau = _make_tau()
    cond = _make_cond(embed_dim=small_cfg.cond_embed_dim)
    out = velocity_net(x_tau, tau, cond)
    loss = out.sum()
    loss.backward()
    for name, p in velocity_net.named_parameters():
        assert p.grad is not None, f"Param {name} has no gradient"


@pytest.mark.unit
def test_velocity_net_eval_deterministic(velocity_net: VelocityNet, small_cfg: FMConfig) -> None:
    velocity_net.eval()
    x_tau = _make_x_tau()
    tau = _make_tau()
    cond = _make_cond(embed_dim=small_cfg.cond_embed_dim)
    with torch.no_grad():
        out1 = velocity_net(x_tau, tau, cond)
        out2 = velocity_net(x_tau, tau, cond)
    assert torch.allclose(out1, out2)


@pytest.mark.unit
def test_velocity_net_no_nan_in_output(velocity_net: VelocityNet, small_cfg: FMConfig) -> None:
    x_tau = _make_x_tau()
    tau = _make_tau()
    cond = _make_cond(embed_dim=small_cfg.cond_embed_dim)
    with torch.no_grad():
        out = velocity_net(x_tau, tau, cond)
    assert not torch.isnan(out).any(), "VelocityNet output contains NaN"
    assert not torch.isinf(out).any(), "VelocityNet output contains Inf"


@pytest.mark.unit
def test_velocity_net_output_near_zero_at_init(small_cfg: FMConfig) -> None:
    """Head weight init is small so initial outputs are near zero."""
    torch.manual_seed(0)
    model = VelocityNet(small_cfg)
    model.eval()
    x_tau = _make_x_tau()
    tau = _make_tau()
    cond = _make_cond(embed_dim=small_cfg.cond_embed_dim)
    with torch.no_grad():
        out = model(x_tau, tau, cond)
    assert out.abs().max().item() < 1.0, (
        f"Initial output magnitude too large: {out.abs().max().item():.4f}; "
        "check head initialization (expected small init)"
    )


# ---------------------------------------------------------------------------
# Full pipeline: ConditioningEncoder + VelocityNet
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encoder_velocity_net_pipeline(small_cfg: FMConfig) -> None:
    """End-to-end: scalars → conditioning → velocity field."""
    encoder = ConditioningEncoder(small_cfg.cond_dim, small_cfg.cond_embed_dim)
    model = VelocityNet(small_cfg)
    model.eval()
    encoder.eval()

    scalars = _make_scalars()
    x_tau = _make_x_tau()
    tau = _make_tau()

    with torch.no_grad():
        cond = encoder(scalars)
        out = model(x_tau, tau, cond)

    assert tuple(out.shape) == (BATCH, 1, NZ, NY, NX)


@pytest.mark.unit
def test_encoder_velocity_net_gradient_flow(small_cfg: FMConfig) -> None:
    """Gradients flow from velocity loss back through both modules."""
    encoder = ConditioningEncoder(small_cfg.cond_dim, small_cfg.cond_embed_dim)
    model = VelocityNet(small_cfg)

    scalars = _make_scalars()
    x_tau = _make_x_tau()
    tau = _make_tau()

    cond = encoder(scalars)
    out = model(x_tau, tau, cond)
    out.sum().backward()

    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"Encoder param {name} has no gradient"
    for name, p in model.named_parameters():
        assert p.grad is not None, f"VelocityNet param {name} has no gradient"
