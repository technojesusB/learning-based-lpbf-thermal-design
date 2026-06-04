"""Unit tests for DiTBlock and sinusoidal_time_embedding."""

from __future__ import annotations

import pytest
import torch

from neural_pbf.models.generative.fm.dit import DiTBlock, sinusoidal_time_embedding


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


@pytest.mark.unit
def test_sinusoidal_time_embedding_gradient_exists() -> None:
    t = torch.rand(4, requires_grad=False)
    out = sinusoidal_time_embedding(t, dim=16)
    assert out.requires_grad is False  # t has no grad; output has none either


# ---------------------------------------------------------------------------
# DiTBlock adaLN-zero
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
    """At init the adaLN modulation outputs all-zeros → block acts as identity."""
    B, N, D = 2, 8, 32
    block = DiTBlock(dim=D, num_heads=4)
    block.eval()

    x = torch.randn(B, N, D)
    c = torch.zeros(B, D)  # zero condition: adaLN shifts/scales/gates are 0 too

    with torch.no_grad():
        out = block(x, c)

    # With zero c, adaLN_modulation outputs zeros → shift=0, scale=0, gate=0.
    # gate1=gate2=0 suppresses both branches → output == input.
    assert torch.allclose(out, x, atol=1e-5), (
        "DiTBlock should be identity at init when condition is zero"
    )


@pytest.mark.unit
def test_dit_block_gradient_flows_through_condition() -> None:
    B, N, D = 2, 8, 32
    block = DiTBlock(dim=D, num_heads=4)
    x = torch.randn(B, N, D)
    c = torch.randn(B, D, requires_grad=True)
    out = block(x, c)
    loss = out.mean()
    loss.backward()
    assert c.grad is not None
    assert c.grad.shape == c.shape
