"""Tests for PhysicsInformedLoss — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations

import torch


def _make_sim_config(size: int = 8):
    """Create a minimal 3D SimulationConfig for a small cube."""
    from neural_pbf.core.config import SimulationConfig

    # size grid points over a 1mm domain in each direction
    # dx = 1mm / (size-1)
    return SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        Nx=size,
        Ny=size,
        Nz=size,
    )


def _make_mat_config():
    from neural_pbf.physics.material import MaterialConfig

    return MaterialConfig.ss316l_preset()


def _make_volume(B: int = 1, size: int = 8, val: float = 300.0) -> torch.Tensor:
    """Uniform temperature volume (B, 1, D, H, W)."""
    return torch.full((B, 1, size, size, size), val, dtype=torch.float32)


def test_loss_dict_keys():
    """Forward pass must return a dict with keys: 'loss', 'mse', 'pde'."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.1)

    T_pred = _make_volume()
    T_target = _make_volume(val=310.0)
    T_in = _make_volume()
    Q = torch.zeros_like(T_pred)
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    assert "loss" in out
    assert "mse" in out
    assert "pde" in out
    assert "mask_bce" in out


def test_mse_only_when_pde_weight_zero():
    """When pde_weight=0, total loss must equal the normalised MSE (divided by T_ref²)."""
    import torch.nn.functional as F

    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    T_ref = 2000.0  # default
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.0, T_ref=T_ref)

    T_pred = _make_volume(val=350.0)
    T_target = _make_volume(val=300.0)
    T_in = _make_volume()
    Q = torch.zeros_like(T_pred)
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    # MSE is normalised internally by T_ref² so both components are O(1)
    expected_mse = F.mse_loss(T_pred, T_target) / (T_ref**2)
    assert torch.allclose(
        out["loss"], expected_mse, atol=1e-6
    ), f"loss={out['loss'].item():.6f}, expected={expected_mse.item():.6f}"
    assert torch.allclose(out["mse"], expected_mse, atol=1e-6)


def test_uniform_field_pde_residual_near_zero():
    """For a spatially uniform temperature and Q=0, the PDE residual must be ~0."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=1.0)

    T_uniform = _make_volume(val=500.0)
    T_target = _make_volume(val=500.0)
    Q = torch.zeros_like(T_uniform)
    # dT/dt ≈ 0 because T_pred == T_in
    dt = 1e-5

    out = loss_fn(T_uniform, T_target, T_uniform.clone(), Q, dt)

    # pde residual for uniform T, Q=0: div(k grad T) = 0; rho*cp*dT/dt = 0
    pde_val = out["pde"].item()
    assert (
        pde_val < 1.0
    ), f"PDE residual should be near zero for uniform field, got {pde_val:.4f}"


def test_gradients_flow():
    """loss.backward() must complete without error and T_pred.grad must be non-None."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.1)

    T_pred = _make_volume(val=400.0).requires_grad_(True)
    T_target = _make_volume(val=300.0)
    T_in = _make_volume()
    Q = torch.rand_like(T_pred.detach()) * 1e8
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)
    out["loss"].backward()

    assert T_pred.grad is not None, "Gradient must flow through T_pred"
    assert not torch.isnan(T_pred.grad).any(), "Gradient must not contain NaN"


def test_pde_weight_scales_pde_term():
    """Higher pde_weight must produce a strictly higher total loss."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()

    T_pred = _make_volume(val=500.0)
    T_target = _make_volume(val=300.0)
    T_in = _make_volume(val=450.0)  # Different from T_pred → non-zero dT/dt
    Q = torch.rand_like(T_pred) * 1e8  # Strong source → non-zero PDE residual
    dt = 1e-5

    loss_low = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.001)
    loss_high = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=10.0)

    out_low = loss_low(T_pred, T_target, T_in, Q, dt)
    out_high = loss_high(T_pred, T_target, T_in, Q, dt)

    assert (
        out_high["loss"].item() > out_low["loss"].item()
    ), "Higher pde_weight should increase total loss when PDE residual is non-zero"


def test_loss_values_are_scalar():
    """All returned loss values must be 0-dimensional (scalar) tensors."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.1)

    T_pred = _make_volume()
    T_target = _make_volume(val=310.0)
    T_in = _make_volume()
    Q = torch.zeros_like(T_pred)
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    for key, val in out.items():
        assert val.ndim == 0, f"Expected scalar for key '{key}', got shape {val.shape}"


def test_no_nan_in_loss():
    """Loss computation must not produce NaN values."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=0.1)

    # Use realistic temperature range
    T_pred = _make_volume(val=1500.0)
    T_target = _make_volume(val=1200.0)
    T_in = _make_volume(val=1000.0)
    Q = torch.ones_like(T_pred) * 1e9
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    for key, val in out.items():
        assert not torch.isnan(val), f"NaN detected in loss key '{key}'"


# ---------------------------------------------------------------------------
# Normalization tests (RED phase — written before implementation)
# ---------------------------------------------------------------------------


def test_loss_scale_order_of_magnitude():
    """With realistic LPBF inputs (T~1500 K, Q~1e12 W/m³), loss terms must be O(1)."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=1.0)

    T_pred = _make_volume(val=1500.0)
    T_target = _make_volume(val=1400.0)
    T_in = _make_volume(val=1300.0)
    Q = torch.ones((1, 1, 8, 8, 8)) * 1e12
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    mse_val = out["mse"].item()
    pde_val = out["pde"].item()

    # Both terms must be finite and not exploding (raw SI would give ~10^22)
    assert mse_val < 1e6, f"MSE is not O(1)-scaled, got {mse_val:.3e}"
    assert pde_val < 1e6, f"PDE loss is not O(1)-scaled, got {pde_val:.3e}"
    assert mse_val >= 0.0, "MSE must be non-negative"
    assert pde_val >= 0.0, "PDE loss must be non-negative"


def test_loss_accepts_custom_normalization_params():
    """PhysicsInformedLoss must accept and apply custom T_ref and Q_ref."""
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()

    T_ref = 1000.0
    Q_ref = 1e10
    loss_fn = PhysicsInformedLoss(
        sim_cfg, mat_cfg, pde_weight=0.1, T_ref=T_ref, Q_ref=Q_ref
    )

    assert loss_fn.T_ref == T_ref
    assert loss_fn.Q_ref == Q_ref

    T_pred = _make_volume(val=1500.0)
    T_target = _make_volume(val=1400.0)
    T_in = _make_volume(val=1300.0)
    Q = torch.ones_like(T_pred) * 1e10
    dt = 1e-5

    out = loss_fn(T_pred, T_target, T_in, Q, dt)

    for key, val in out.items():
        assert not torch.isnan(val), f"NaN in '{key}' with custom normalization params"
    assert out["loss"].ndim == 0


# ---------------------------------------------------------------------------
# PhysicsInformedLoss.from_config factory (MEDIUM fix #5)
# ---------------------------------------------------------------------------


def test_k_field_is_detached_in_loss():
    """k_field passed to PhysicsInformedLoss.forward must not receive gradients.

    RED  (before fix): k_field.grad is non-None after backward because gradients
         flow through k_field in the div_k_grad computation.
    GREEN (after fix): k_field.grad is None because the code calls .detach().
    """
    from neural_pbf.models.loss import PhysicsInformedLoss

    sim_cfg = _make_sim_config()
    mat_cfg = _make_mat_config()
    loss_fn = PhysicsInformedLoss(sim_cfg, mat_cfg, pde_weight=1.0)

    T_pred = _make_volume(val=400.0).requires_grad_(True)
    T_target = _make_volume(val=300.0)
    T_in = _make_volume(val=350.0)
    Q = torch.zeros_like(T_pred.detach())
    dt = 1e-5

    # k_field has requires_grad=True — after the fix, no gradient should flow into it
    k_field = torch.full_like(T_pred.detach(), 15.0, requires_grad=True)

    out = loss_fn(T_pred, T_target, T_in, Q, dt, k_field=k_field)
    out["loss"].backward()

    assert k_field.grad is None, (
        "Gradient flowed through k_field, but it should be detached. "
        "Apply k_field.detach() in PhysicsInformedLoss.forward before using it."
    )


class TestFromConfigFactory:
    """from_config must wire mask_weight from SurrogateConfig, not use the default 0.0."""

    def test_factory_wires_mask_weight_from_surrogate_cfg(self):
        from neural_pbf.models.config import SurrogateConfig
        from neural_pbf.models.loss import PhysicsInformedLoss

        sim_cfg = _make_sim_config()
        mat_cfg = _make_mat_config()
        surrogate_cfg = SurrogateConfig(
            strategy="residual",
            use_dual_output=True,
            base_channels=4,
            depth=2,
            mask_weight=2.5,
            pde_weight=0.3,
        )
        loss = PhysicsInformedLoss.from_config(surrogate_cfg, sim_cfg, mat_cfg)
        assert (
            loss.mask_weight == 2.5
        ), "from_config must forward mask_weight from SurrogateConfig"
        assert (
            loss.pde_weight == 0.3
        ), "from_config must forward pde_weight from SurrogateConfig"

    def test_factory_produces_functional_loss(self):
        import torch

        from neural_pbf.models.config import SurrogateConfig
        from neural_pbf.models.loss import PhysicsInformedLoss

        sim_cfg = _make_sim_config()
        mat_cfg = _make_mat_config()
        surrogate_cfg = SurrogateConfig(
            use_dual_output=True,
            base_channels=4,
            depth=2,
        )
        loss = PhysicsInformedLoss.from_config(surrogate_cfg, sim_cfg, mat_cfg)
        shape = (1, 1, 8, 8, 8)
        T_pred = torch.full(shape, 1200.0)
        T_tgt = torch.full(shape, 1100.0)
        Q = torch.zeros(shape)
        T_in = torch.full(shape, 1000.0)
        result = loss(T_pred, T_tgt, T_in, Q, dt=5e-5)
        assert result["loss"].item() >= 0.0

    def test_default_mask_weight_differs_from_surrogate_config_default(self):
        """Ensure the known mismatch is documented: PhysicsInformedLoss default
        is 0.0 (disabled), SurrogateConfig default is 1.0. from_config resolves
        the ambiguity by always taking the SurrogateConfig value."""
        from neural_pbf.models.config import SurrogateConfig
        from neural_pbf.models.loss import PhysicsInformedLoss

        assert PhysicsInformedLoss.__init__.__defaults__ is not None
        # Direct construction with no mask_weight uses 0.0 (disabled by default)
        sim_cfg = _make_sim_config()
        mat_cfg = _make_mat_config()
        direct_loss = PhysicsInformedLoss(sim_cfg, mat_cfg)
        assert direct_loss.mask_weight == 0.0

        # SurrogateConfig.mask_weight defaults to 1.0
        assert SurrogateConfig().mask_weight == 1.0

        # from_config always uses SurrogateConfig's value — no silent disagreement
        cfg_with_default = SurrogateConfig()
        factory_loss = PhysicsInformedLoss.from_config(
            cfg_with_default, sim_cfg, mat_cfg
        )
        assert factory_loss.mask_weight == cfg_with_default.mask_weight
