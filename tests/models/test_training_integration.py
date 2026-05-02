"""Integration tests for the surrogate training loop — written BEFORE scripts (TDD RED).

These tests exercise the full data pipeline:
  replay buffer → model forward → loss → backward → parameter update

They use tiny 8×8×8 domains so they complete in seconds on CPU.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim_cfg():
    from neural_pbf.core.config import SimulationConfig

    return SimulationConfig(Lx=1.0, Ly=1.0, Lz=1.0, Nx=8, Ny=8, Nz=8)


@pytest.fixture()
def mat_cfg():
    from neural_pbf.physics.material import MaterialConfig

    return MaterialConfig.ss316l_preset()


@pytest.fixture()
def surrogate_cfg():
    from neural_pbf.models.config import SurrogateConfig

    return SurrogateConfig(
        strategy="direct",
        base_channels=4,
        depth=2,
        patch_size=8,
        lr=1e-3,
        pde_weight=0.01,
        batch_size=2,
        buffer_capacity=32,
    )


@pytest.fixture()
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_volume(size: int = 8, val: float = 300.0) -> torch.Tensor:
    """Uniform temperature volume (1, 1, D, H, W)."""
    return torch.full((1, 1, size, size, size), val, dtype=torch.float32)


def _make_random_volume(
    size: int = 8, lo: float = 300.0, hi: float = 500.0
) -> torch.Tensor:
    return torch.rand(1, 1, size, size, size, dtype=torch.float32) * (hi - lo) + lo


def _fill_buffer(buf, n: int = 10, size: int = 8) -> None:
    """Push ``n`` synthetic experiences into the buffer."""
    for _ in range(n):
        T_in = _make_random_volume(size)
        Q = torch.rand_like(T_in) * 1e8
        T_target = T_in + torch.rand_like(T_in) * 20
        buf.push(T_in, Q, T_target)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_training_loop_reduces_loss(sim_cfg, mat_cfg, surrogate_cfg, device):
    """Running several gradient steps must reduce the surrogate training loss."""
    from neural_pbf.models.loss import PhysicsInformedLoss
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(0)

    model = ThermalSurrogate3D(surrogate_cfg).to(device)
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=surrogate_cfg.pde_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=surrogate_cfg.lr)

    buf = ExperienceReplayBuffer(
        capacity=surrogate_cfg.buffer_capacity,
        patch_size=surrogate_cfg.patch_size,
        device=device,
    )
    _fill_buffer(buf, n=20)

    dt = sim_cfg.dt_base

    # Record initial loss
    batch = buf.sample(surrogate_cfg.batch_size)
    T_in = batch["T_in"]
    Q = batch["Q"]
    T_target = batch["T_target"]

    with torch.no_grad():
        T_pred_init = model(T_in, Q)
        init_metrics = loss_fn(T_pred_init, T_target, T_in, Q, dt)
    initial_loss = init_metrics["loss"].item()

    # Train for several iterations
    n_train_iters = 10
    for _ in range(n_train_iters):
        batch = buf.sample(surrogate_cfg.batch_size)
        T_in = batch["T_in"]
        Q = batch["Q"]
        T_target = batch["T_target"]

        T_pred = model(T_in, Q)
        metrics = loss_fn(T_pred, T_target, T_in, Q, dt)

        optimizer.zero_grad()
        metrics["loss"].backward()
        optimizer.step()

    # Record final loss
    batch = buf.sample(surrogate_cfg.batch_size)
    T_in = batch["T_in"]
    Q = batch["Q"]
    T_target = batch["T_target"]

    with torch.no_grad():
        T_pred_final = model(T_in, Q)
        final_metrics = loss_fn(T_pred_final, T_target, T_in, Q, dt)
    final_loss = final_metrics["loss"].item()

    assert final_loss < initial_loss, (
        f"Training did not reduce loss: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_checkpoint_save_load(surrogate_cfg, device):
    """Saving and loading a checkpoint must reproduce identical forward-pass outputs."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(42)
    model = ThermalSurrogate3D(surrogate_cfg).to(device)
    model.eval()

    T = _make_random_volume()
    Q = _make_random_volume(lo=0, hi=1e8)

    with torch.no_grad():
        out_before = model(T, Q).clone()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": surrogate_cfg.model_dump(),
            },
            ckpt_path,
        )

        # Reload
        ckpt = torch.load(ckpt_path, map_location=device)
        from neural_pbf.models.config import SurrogateConfig

        cfg_loaded = SurrogateConfig(**ckpt["cfg"])
        model_loaded = ThermalSurrogate3D(cfg_loaded).to(device)
        model_loaded.load_state_dict(ckpt["model"])
        model_loaded.eval()

    with torch.no_grad():
        out_after = model_loaded(T, Q)

    assert torch.allclose(
        out_before, out_after, atol=1e-6
    ), "Loaded model must produce identical outputs to the saved model"


def test_buffer_to_model_pipeline(surrogate_cfg, sim_cfg, mat_cfg, device):
    """End-to-end: buffer→model forward→loss must not raise or produce NaN."""
    from neural_pbf.models.loss import PhysicsInformedLoss
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(5)
    model = ThermalSurrogate3D(surrogate_cfg).to(device)
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=surrogate_cfg.pde_weight)

    buf = ExperienceReplayBuffer(
        capacity=surrogate_cfg.buffer_capacity,
        patch_size=surrogate_cfg.patch_size,
        device=device,
    )
    _fill_buffer(buf, n=8)

    batch = buf.sample(surrogate_cfg.batch_size)
    T_in = batch["T_in"]
    Q = batch["Q"]
    T_target = batch["T_target"]

    T_pred = model(T_in, Q)
    metrics = loss_fn(T_pred, T_target, T_in, Q, sim_cfg.dt_base)

    for key, val in metrics.items():
        assert not torch.isnan(val), f"NaN detected in metric '{key}'"
        assert val.ndim == 0, f"Expected scalar for '{key}', got shape {val.shape}"


def test_residual_strategy_training(sim_cfg, mat_cfg, device):
    """Residual strategy surrogate can be trained end-to-end without errors."""
    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.loss import PhysicsInformedLoss
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    cfg = SurrogateConfig(
        strategy="residual",
        base_channels=4,
        depth=2,
        patch_size=8,
        lr=1e-3,
        pde_weight=0.01,
        batch_size=2,
        buffer_capacity=32,
    )
    torch.manual_seed(1)
    model = ThermalSurrogate3D(cfg).to(device)
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=cfg.pde_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    buf = ExperienceReplayBuffer(capacity=32, patch_size=8, device=device)

    # Push experiences WITH T_lf
    for _ in range(10):
        T_in = _make_random_volume()
        Q = torch.rand_like(T_in) * 1e8
        T_target = T_in + torch.rand_like(T_in) * 20
        T_lf = T_in + torch.rand_like(T_in) * 5  # coarse approx
        buf.push(T_in, Q, T_target, T_lf=T_lf)

    batch = buf.sample(cfg.batch_size)
    T_in = batch["T_in"]
    Q = batch["Q"]
    T_target = batch["T_target"]
    T_lf = batch["T_lf"]

    T_pred = model(T_in, Q, T_lf=T_lf)
    metrics = loss_fn(T_pred, T_target, T_in, Q, sim_cfg.dt_base)

    optimizer.zero_grad()
    metrics["loss"].backward()
    optimizer.step()

    assert not torch.isnan(metrics["loss"]), "NaN in residual strategy training loss"


def test_gradient_clip_does_not_break_training(surrogate_cfg, sim_cfg, mat_cfg, device):
    """Gradient clipping (common in practice) must not break the training step."""
    from neural_pbf.models.loss import PhysicsInformedLoss
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(99)
    model = ThermalSurrogate3D(surrogate_cfg).to(device)
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=surrogate_cfg.pde_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=surrogate_cfg.lr)

    buf = ExperienceReplayBuffer(capacity=32, patch_size=8, device=device)
    _fill_buffer(buf, n=8)

    batch = buf.sample(surrogate_cfg.batch_size)
    T_pred = model(batch["T_in"], batch["Q"])
    metrics = loss_fn(
        T_pred, batch["T_target"], batch["T_in"], batch["Q"], sim_cfg.dt_base
    )

    optimizer.zero_grad()
    metrics["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Verify model params updated (no NaN weights)
    for name, param in model.named_parameters():
        assert not torch.isnan(
            param
        ).any(), f"NaN in parameter '{name}' after grad clip"
