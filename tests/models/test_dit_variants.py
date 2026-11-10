"""DiT v1: learned absolute position embedding (journal section 5.1), outside the S4-hashed model package."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_pbf.models.dit_variants import VelocityDiTAPE
from neural_pbf.models.generative.fm.dit import VelocityDiT

KW = {"patch_size": 4, "in_channels": 4, "embed_dim": 24, "depth": 1, "num_heads": 2, "cond_embed_dim": 8,
      "input_size": 16}  # fmt: skip


def _inputs(B: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, 4, 16, 16, 16, generator=g), torch.rand(B, generator=g), torch.randn(B, 8, generator=g)


@pytest.mark.unit
def test_position_embedding_is_learned_with_one_vector_per_token() -> None:
    model = VelocityDiTAPE(**KW)
    assert isinstance(model.pos_embed, nn.Parameter) and model.pos_embed.requires_grad
    assert model.pos_embed.shape == (1, (16 // 4) ** 3, 24)
    assert "pos_embed" in dict(model.named_parameters()) and "pos_embed" not in dict(model.named_buffers())
    assert 0.005 < float(model.pos_embed.std()) < 0.05  # trunc_normal std 0.02


@pytest.mark.unit
def test_forward_shape_and_the_embedding_is_added_unscaled() -> None:
    torch.manual_seed(0)
    model = VelocityDiTAPE(**KW).eval()
    x, t, c = _inputs()
    out = model(x, t, c)
    assert out.shape == (2, 1, 16, 16, 16) and torch.isfinite(out).all()
    with torch.no_grad():
        model.pos_embed.add_(1.0)
    assert not torch.allclose(model(x, t, c), out)
    tokens = model.patch_embed(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(model.embed_tokens(x), tokens + model.pos_embed)


@pytest.mark.unit
def test_the_embedding_trains() -> None:
    torch.manual_seed(0)
    model = VelocityDiTAPE(**KW)
    x, t, c = _inputs()
    model(x, t, c).pow(2).mean().backward()
    assert model.pos_embed.grad is not None and model.pos_embed.grad.abs().sum() > 0


@pytest.mark.unit
def test_state_dict_round_trip_and_v2_is_unchanged() -> None:
    torch.manual_seed(0)
    a = VelocityDiTAPE(**KW)
    b = VelocityDiTAPE(**KW)
    b.load_state_dict(a.state_dict())
    x, t, c = _inputs()
    torch.testing.assert_close(a.eval()(x, t, c), b.eval()(x, t, c))
    assert "pos_embed" in dict(VelocityDiT(**KW).named_buffers())  # v2 keeps its fixed sinusoidal buffer
