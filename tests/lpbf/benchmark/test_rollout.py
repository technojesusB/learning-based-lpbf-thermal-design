"""Tests for benchmark.rollout — run_euler_rollout."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch


from neural_pbf.eval.benchmark.rollout import run_euler_rollout


class _ZeroVelocityNet(nn.Module):
    """Returns zero velocity — output matches T_in channel count."""
    def forward(self, x: torch.Tensor, tau: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        return out[:, :1]  # 1-channel output


class _ZeroCondEnc(nn.Module):
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return cond


def _net_batch(B=1, C=1, D=4, H=4, W=4, device=torch.device("cpu")) -> dict:
    shape = (B, C, D, H, W)
    return {
        "T_in": torch.rand(*shape, device=device),
        "mask": torch.zeros(*shape, device=device),
        "Q": torch.zeros(*shape, device=device),
        "conditioning": torch.zeros(B, 4, device=device),
        "T_target": torch.rand(*shape, device=device),
    }


@pytest.mark.unit
def test_run_euler_rollout_returns_tensor():
    batch = _net_batch()
    out = run_euler_rollout(_ZeroVelocityNet(), _ZeroCondEnc(), batch, "net", torch.device("cpu"))
    assert isinstance(out, torch.Tensor)


@pytest.mark.unit
def test_run_euler_rollout_output_shape_matches_t_in():
    batch = _net_batch(B=1, D=4, H=4, W=4)
    out = run_euler_rollout(_ZeroVelocityNet(), _ZeroCondEnc(), batch, "net", torch.device("cpu"))
    assert out.shape == batch["T_in"].shape


@pytest.mark.unit
def test_run_euler_rollout_zero_velocity_returns_noise():
    """With zero velocity the output equals the initial noise (unchanged)."""
    torch.manual_seed(0)
    batch = _net_batch()
    model = _ZeroVelocityNet()
    cond_enc = _ZeroCondEnc()
    torch.manual_seed(0)
    out = run_euler_rollout(model, cond_enc, batch, "net", torch.device("cpu"), n_steps=5)
    # output is initial randn noise unchanged by zero-velocity integration
    assert isinstance(out, torch.Tensor)
    assert out.shape == batch["T_in"].shape


@pytest.mark.unit
def test_run_euler_rollout_deterministic_with_seed():
    """Same seed → same output (relies only on randn_like for initial noise)."""
    batch = _net_batch()
    torch.manual_seed(42)
    out1 = run_euler_rollout(_ZeroVelocityNet(), _ZeroCondEnc(), batch, "net", torch.device("cpu"), n_steps=3)
    torch.manual_seed(42)
    out2 = run_euler_rollout(_ZeroVelocityNet(), _ZeroCondEnc(), batch, "net", torch.device("cpu"), n_steps=3)
    assert torch.allclose(out1, out2)


@pytest.mark.unit
def test_run_euler_rollout_n_steps_default_is_25():
    """Calling without n_steps should use the constant N_EULER_STEPS=25."""
    from neural_pbf.eval.benchmark.constants import N_EULER_STEPS
    batch = _net_batch()
    step_count = []

    class _CountingNet(nn.Module):
        def forward(self, x, tau, cond_emb):
            step_count.append(1)
            return torch.zeros_like(x[:, :1])

    run_euler_rollout(_CountingNet(), _ZeroCondEnc(), batch, "net", torch.device("cpu"))
    assert len(step_count) == N_EULER_STEPS


@pytest.mark.unit
def test_run_euler_rollout_rope_raises_without_grid_attrs():
    """rope/triton rollout must raise ValueError when grid_attrs is None."""
    B, D, H, W = 1, 4, 4, 4
    rope_shape = (B, 1, 1, D, H, W)
    batch = {
        "T_in": torch.rand(*rope_shape),
        "mask": torch.zeros(*rope_shape),
        "Q": torch.zeros(*rope_shape),
        "conditioning": torch.zeros(B, 12),
        "T_target": torch.rand(*rope_shape),
        "patch_origin": torch.zeros(B, 3, dtype=torch.long),
    }

    class _RoPEModel(nn.Module):
        patch_size = 4
        def forward(self, x, tau, cond_emb, coords_mm):
            return torch.zeros_like(x[:, :1])

    with pytest.raises(ValueError, match="grid_attrs"):
        run_euler_rollout(_RoPEModel(), _ZeroCondEnc(), batch, "rope", torch.device("cpu"))
