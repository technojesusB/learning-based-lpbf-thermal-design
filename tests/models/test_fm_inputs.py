"""One input layout for FM velocity models, shared by training and sampling."""

from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.benchmark.model_loader import load_model_adaptive
from neural_pbf.eval.benchmark.rollout import run_euler_rollout
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.dit import VelocityDiT
from neural_pbf.models.generative.fm.inputs import model_in_channels, net_inputs
from neural_pbf.models.generative.fm.velocity_net import VelocityNet

SHAPE = (2, 1, 4, 8, 8)


def _batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(SHAPE, generator=g) for k in ("T_in", "mask", "Q")} | {"conditioning": torch.zeros(2, 13)}


@pytest.mark.unit
def test_three_channels_keep_the_legacy_layout():
    b, x = _batch(), torch.ones(SHAPE)
    out = net_inputs(x, b, in_channels=3)
    assert torch.equal(out, torch.cat([x, b["mask"], b["Q"]], dim=1))


@pytest.mark.unit
def test_four_channels_append_t_in():
    b, x = _batch(), torch.ones(SHAPE)
    out = net_inputs(x, b, in_channels=4)
    assert out.shape[1] == 4 and torch.equal(out[:, 3:], b["T_in"])


@pytest.mark.unit
def test_unknown_channel_count_fails():
    with pytest.raises(ValueError, match="in_channels"):
        net_inputs(torch.ones(SHAPE), _batch(), in_channels=5)


@pytest.mark.unit
def test_in_channels_is_read_from_unet_and_dit():
    assert model_in_channels(VelocityNet(FMConfig(in_channels=4, base_channels=4, depth=1))) == 4
    dit = VelocityDiT(depth=1, embed_dim=32, num_heads=2, patch_size=4, input_size=8, in_channels=4, cond_embed_dim=16)
    assert model_in_channels(dit) == 4


class _Spy(torch.nn.Module):
    """Records what the rollout feeds the model; returns zero velocity."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.cfg = FMConfig(in_channels=in_channels)
        self.seen: list[torch.Tensor] = []

    def forward(self, x, tau, cond):  # noqa: ANN001
        self.seen.append(x.clone())
        return torch.zeros_like(x[:, :1])


@pytest.mark.unit
@pytest.mark.parametrize("in_channels", [3, 4])
def test_rollout_feeds_the_same_layout_as_training(in_channels):
    b = _batch()
    spy, enc = _Spy(in_channels), ConditioningEncoder(13, 8)
    gen = torch.Generator().manual_seed(1)
    run_euler_rollout(spy, enc, b, "net", torch.device("cpu"), n_steps=1, generator=gen)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))
    assert torch.equal(spy.seen[0], net_inputs(noise, b, in_channels))


@pytest.mark.unit
def test_loader_builds_dit_with_the_checkpoint_channels_and_cond_dim(tmp_path):
    dit = VelocityDiT(depth=6, embed_dim=256, num_heads=8, patch_size=4, input_size=64, in_channels=4, cond_embed_dim=128)
    enc = ConditioningEncoder(13, 128)
    path = tmp_path / "dit.pt"
    torch.save({"model_state": dit.state_dict(), "cond_encoder_state": enc.state_dict()}, path)
    model, cond_enc, _, _ = load_model_adaptive("d", path, "dit", torch.device("cpu"))
    assert model_in_channels(model) == 4
    assert cond_enc.net[0].in_features == 13
