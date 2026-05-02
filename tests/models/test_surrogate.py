"""Tests for ThermalSurrogate3D — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def direct_cfg():
    """Small direct-strategy config for fast tests."""
    from neural_pbf.models.config import SurrogateConfig

    return SurrogateConfig(
        strategy="direct",
        base_channels=4,
        depth=2,
        patch_size=8,
    )


@pytest.fixture()
def residual_cfg():
    """Small residual-strategy config for fast tests."""
    from neural_pbf.models.config import SurrogateConfig

    return SurrogateConfig(
        strategy="residual",
        base_channels=4,
        depth=2,
        patch_size=8,
    )


def _make_volume(B: int = 1, size: int = 8) -> torch.Tensor:
    """Return a float32 volume of shape (B, 1, size, size, size)."""
    return torch.rand(B, 1, size, size, size, dtype=torch.float32) * 500 + 300


def test_direct_strategy_output_shape(direct_cfg):
    """Forward pass under 'direct' strategy returns same spatial shape as input."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    T = _make_volume(B=2, size=8)
    Q = _make_volume(B=2, size=8)

    with torch.no_grad():
        out = model(T, Q)

    assert out.shape == T.shape, f"Expected {T.shape}, got {out.shape}"


def test_residual_strategy_output_shape(residual_cfg):
    """Forward pass under 'residual' strategy returns same spatial shape as T."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    model = ThermalSurrogate3D(residual_cfg)
    model.eval()

    T = _make_volume(B=2, size=8)
    Q = _make_volume(B=2, size=8)
    T_lf = _make_volume(B=2, size=8)

    with torch.no_grad():
        out = model(T, Q, T_lf=T_lf)

    assert out.shape == T.shape, f"Expected {T.shape}, got {out.shape}"


def test_direct_strategy_adds_input_T(direct_cfg):
    """Output should differ from input T (network applies a non-zero increment)."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(42)
    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    T = _make_volume(B=1, size=8)
    Q = _make_volume(B=1, size=8)

    with torch.no_grad():
        out = model(T, Q)

    # The output is T + dT, so it must differ from T (unless dT is exactly zero)
    assert not torch.allclose(out, T, atol=1e-6), "Output should differ from input T"


def test_residual_strategy_output_near_lf(residual_cfg):
    """When the model is freshly initialised (small random weights, small Q),
    the residual delta should be small and the output should be in the same
    ballpark as T_lf (not wildly off)."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(0)
    model = ThermalSurrogate3D(residual_cfg)
    # Zero out all weights so delta = 0 → output == T_lf exactly
    for p in model.parameters():
        p.data.zero_()

    model.eval()
    T = _make_volume(B=1, size=8)
    Q = torch.zeros_like(T)
    T_lf = _make_volume(B=1, size=8)

    with torch.no_grad():
        out = model(T, Q, T_lf=T_lf)

    # With zeroed weights the network output (increment) is 0, so out == T_lf
    assert torch.allclose(
        out, T_lf, atol=1e-5
    ), "With zero weights, output should equal T_lf"


def test_predict_autoregressive_returns_n_steps(direct_cfg):
    """predict_autoregressive must return exactly len(Q_sequence) tensors."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    n_steps = 5
    T_init = _make_volume(B=1, size=8)
    Q_sequence = [_make_volume(B=1, size=8) for _ in range(n_steps)]

    with torch.no_grad():
        predictions = model.predict_autoregressive(T_init, Q_sequence)

    assert len(predictions) == n_steps


def test_no_nan_in_forward(direct_cfg):
    """Forward pass must not produce NaN or Inf values."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(7)
    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    T = _make_volume(B=2, size=8)
    Q = _make_volume(B=2, size=8)

    with torch.no_grad():
        out = model(T, Q)

    assert not torch.isnan(out).any(), "NaN detected in forward pass output"
    assert not torch.isinf(out).any(), "Inf detected in forward pass output"


def test_autoregressive_no_nan(direct_cfg):
    """Autoregressive prediction must not produce NaN across multiple steps."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(3)
    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    T_init = _make_volume(B=1, size=8)
    Q_sequence = [_make_volume(B=1, size=8) for _ in range(4)]

    with torch.no_grad():
        predictions = model.predict_autoregressive(T_init, Q_sequence)

    for i, T_pred in enumerate(predictions):
        assert not torch.isnan(T_pred).any(), f"NaN at autoregressive step {i}"


def test_residual_forward_requires_t_lf(residual_cfg):
    """Calling forward without T_lf under residual strategy must raise ValueError."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    model = ThermalSurrogate3D(residual_cfg)
    model.eval()

    T = _make_volume(B=1, size=8)
    Q = _make_volume(B=1, size=8)

    with pytest.raises((ValueError, AssertionError)):
        model(T, Q)  # T_lf not provided — should fail for residual strategy


def test_autoregressive_with_t_lf_sequence(residual_cfg):
    """predict_autoregressive must accept T_lf_sequence for residual strategy."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    model = ThermalSurrogate3D(residual_cfg)
    model.eval()

    n_steps = 3
    T_init = _make_volume(B=1, size=8)
    Q_sequence = [_make_volume(B=1, size=8) for _ in range(n_steps)]
    T_lf_sequence = [_make_volume(B=1, size=8) for _ in range(n_steps)]

    with torch.no_grad():
        predictions = model.predict_autoregressive(
            T_init, Q_sequence, T_lf_sequence=T_lf_sequence
        )

    assert len(predictions) == n_steps
    for pred in predictions:
        assert not torch.isnan(pred).any()


def test_centre_crop_handles_odd_dims():
    """_centre_crop must work correctly when dimensions differ by 1."""
    from neural_pbf.models.surrogate import _centre_crop

    # 9×9×9 src → crop to 8×8×8
    src = torch.rand(1, 4, 9, 9, 9)
    target_shape = torch.Size([1, 4, 8, 8, 8])
    cropped = _centre_crop(src, target_shape)
    assert cropped.shape == target_shape


def test_parameter_count_scales_with_depth():
    """Deeper models must have more parameters than shallower ones."""
    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    shallow = ThermalSurrogate3D(SurrogateConfig(base_channels=4, depth=1))
    deep = ThermalSurrogate3D(SurrogateConfig(base_channels=4, depth=3))

    shallow_params = sum(p.numel() for p in shallow.parameters())
    deep_params = sum(p.numel() for p in deep.parameters())

    assert (
        deep_params > shallow_params
    ), f"Deeper model should have more params: {deep_params} vs {shallow_params}"


# ---------------------------------------------------------------------------
# Normalization tests (RED phase — written before implementation)
# ---------------------------------------------------------------------------


def test_config_has_normalization_fields():
    """SurrogateConfig must expose T_ref, Q_ref, T_ambient with correct defaults."""
    from neural_pbf.models.config import SurrogateConfig

    cfg = SurrogateConfig()
    assert cfg.T_ref == 2000.0, f"Expected T_ref=2000.0, got {cfg.T_ref}"
    assert cfg.Q_ref == 1e12, f"Expected Q_ref=1e12, got {cfg.Q_ref}"
    assert cfg.T_ambient == 300.0, f"Expected T_ambient=300.0, got {cfg.T_ambient}"


def test_forward_output_in_si_kelvin(direct_cfg):
    """Forward pass must return temperatures in a physically plausible Kelvin range."""
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    torch.manual_seed(42)
    model = ThermalSurrogate3D(direct_cfg)
    model.eval()

    # Realistic LPBF inputs: T ~ 1500 K, Q ~ 1e12 W/m³
    T = torch.full((1, 1, 8, 8, 8), 1500.0)
    Q = torch.full((1, 1, 8, 8, 8), 1e12)

    with torch.no_grad():
        out = model(T, Q)

    # Output must be in SI Kelvin, not in the normalized [0, 1] range
    assert (
        out.min().item() > 50.0
    ), "Output below 50 K — likely still in normalized space"
    assert (
        out.max().item() < 1e6
    ), "Output above 1e6 K — likely un-normalization is broken"


def test_normalization_is_transparent(direct_cfg):
    """Normalization must be internal: the model always accepts and returns SI Kelvin."""
    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    # Model with custom (non-default) normalization refs
    cfg_custom = SurrogateConfig(
        strategy="direct",
        base_channels=4,
        depth=2,
        patch_size=8,
        T_ref=1000.0,
        Q_ref=1e10,
        T_ambient=400.0,
    )

    torch.manual_seed(0)
    model_default = ThermalSurrogate3D(direct_cfg)
    torch.manual_seed(0)
    model_custom = ThermalSurrogate3D(cfg_custom)

    model_default.eval()
    model_custom.eval()

    T = torch.full((1, 1, 8, 8, 8), 1500.0)
    Q = torch.full((1, 1, 8, 8, 8), 1e12)

    with torch.no_grad():
        out_default = model_default(T, Q)
        out_custom = model_custom(T, Q)

    # Both outputs must be in SI Kelvin (same input/output contract regardless of refs)
    for name, out in [("default", out_default), ("custom", out_custom)]:
        assert not torch.isnan(out).any(), f"{name}: NaN in output"
        assert out.min().item() > 50.0, f"{name}: output below 50 K"
        assert out.max().item() < 1e6, f"{name}: output above 1e6 K"


def test_custom_normalization_params_accepted():
    """SurrogateConfig must accept custom T_ref, Q_ref, T_ambient without error."""
    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    cfg = SurrogateConfig(
        base_channels=4,
        depth=2,
        T_ref=1000.0,
        Q_ref=1e10,
        T_ambient=400.0,
    )
    assert cfg.T_ref == 1000.0
    assert cfg.Q_ref == 1e10
    assert cfg.T_ambient == 400.0

    model = ThermalSurrogate3D(cfg)
    model.eval()

    T = _make_volume(B=1, size=8)
    Q = _make_volume(B=1, size=8)

    with torch.no_grad():
        out = model(T, Q)

    assert out.shape == T.shape
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Dual-output return type contract
# ---------------------------------------------------------------------------


@pytest.fixture()
def dual_cfg():
    from neural_pbf.models.config import SurrogateConfig

    return SurrogateConfig(
        strategy="residual",
        use_dual_output=True,
        base_channels=4,
        depth=2,
        patch_size=8,
    )


def _make_const_volume(B: int = 1, size: int = 8) -> torch.Tensor:
    import torch

    return torch.full((B, 1, size, size, size), 1200.0)


class TestForwardReturnTypeContract:
    """forward() must return Tensor when use_dual_output=False, tuple when True."""

    def test_single_output_returns_tensor(self, direct_cfg):
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        model = ThermalSurrogate3D(direct_cfg).eval()
        T = _make_const_volume()
        Q = _make_const_volume()
        with torch.no_grad():
            result = model(T, Q)
        assert isinstance(
            result, torch.Tensor
        ), "use_dual_output=False must return a plain Tensor, not a tuple"

    def test_dual_output_returns_tuple_of_tensors(self, dual_cfg):
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        model = ThermalSurrogate3D(dual_cfg).eval()
        T = _make_const_volume()
        Q = _make_const_volume()
        T_lf = _make_const_volume()
        with torch.no_grad():
            result = model(T, Q, T_lf=T_lf)
        assert (
            isinstance(result, tuple) and len(result) == 2
        ), "use_dual_output=True must return (T_pred, mask_logits) tuple"
        T_pred, mask_logits = result
        assert isinstance(T_pred, torch.Tensor)
        assert isinstance(mask_logits, torch.Tensor)
        assert T_pred.shape == T.shape
        assert mask_logits.shape == T.shape

    def test_return_type_matches_use_dual_output_flag(self, direct_cfg, dual_cfg):
        """isinstance(result, tuple) must equal cfg.use_dual_output."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        T = _make_const_volume()
        Q = _make_const_volume()
        T_lf = _make_const_volume()

        for cfg, expect_tuple in [(direct_cfg, False), (dual_cfg, True)]:
            model = ThermalSurrogate3D(cfg).eval()
            kwargs = {"T_lf": T_lf} if cfg.use_dual_output else {}
            with torch.no_grad():
                result = model(T, Q, **kwargs)
            assert (
                isinstance(result, tuple) == expect_tuple == cfg.use_dual_output
            ), f"cfg.use_dual_output={cfg.use_dual_output} but got tuple={isinstance(result, tuple)}"

    def test_predict_autoregressive_dual_output_returns_tuples(self, dual_cfg):
        """predict_autoregressive with use_dual_output=True must return list of tuples."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        model = ThermalSurrogate3D(dual_cfg).eval()
        T_init = _make_const_volume()
        Q_seq = [_make_const_volume() for _ in range(3)]
        T_lf_seq = [_make_const_volume() for _ in range(3)]
        preds = model.predict_autoregressive(T_init, Q_seq, T_lf_sequence=T_lf_seq)
        assert len(preds) == 3
        for item in preds:
            assert (
                isinstance(item, tuple) and len(item) == 2
            ), "Each step in dual-output autoregressive must be (T_pred, mask_logits)"

    def test_predict_autoregressive_single_output_returns_tensors(self, direct_cfg):
        """predict_autoregressive with use_dual_output=False must return list of Tensors."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        model = ThermalSurrogate3D(direct_cfg).eval()
        T_init = _make_const_volume()
        Q_seq = [_make_const_volume() for _ in range(3)]
        preds = model.predict_autoregressive(T_init, Q_seq)
        assert len(preds) == 3
        for item in preds:
            assert isinstance(
                item, torch.Tensor
            ), "Single-output autoregressive must return Tensors, not tuples"


# ---------------------------------------------------------------------------
# Device-default fix for predict_autoregressive (TDD RED phase)
# ---------------------------------------------------------------------------


class TestPredictAutoregressiveDeviceDefault:
    """predict_autoregressive must inherit device from T_init, not default to CPU."""

    def test_device_defaults_to_t_init_device(self, direct_cfg):
        """When device=None, predictions must stay on the same device as T_init (CPU)."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        model = ThermalSurrogate3D(direct_cfg).eval()
        cpu = torch.device("cpu")
        T_init = _make_const_volume().to(cpu)
        Q_seq = [_make_const_volume().to(cpu) for _ in range(3)]

        preds = model.predict_autoregressive(T_init, Q_seq, device=None)

        for i, pred in enumerate(preds):
            p = pred[0] if isinstance(pred, tuple) else pred
            assert p.device.type == cpu.type, (
                f"Step {i}: prediction device {p.device} does not match "
                f"T_init device {cpu}. "
                "Fix: change `device = torch.device('cpu')` default to "
                "`device = T_init.device if device is None else device`."
            )

    def test_explicit_device_overrides_t_init_device(self, direct_cfg):
        """When device=cpu is passed explicitly, all predictions must be on CPU."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        cpu = torch.device("cpu")
        model = ThermalSurrogate3D(direct_cfg).eval().to(cpu)
        T_init = _make_const_volume().to(cpu)
        Q_seq = [_make_const_volume().to(cpu) for _ in range(2)]

        preds = model.predict_autoregressive(T_init, Q_seq, device=cpu)

        for i, pred in enumerate(preds):
            p = pred[0] if isinstance(pred, tuple) else pred
            assert p.device.type == "cpu", (
                f"Step {i}: expected cpu, got {p.device}"
            )

    def test_model_device_assertion_same_device_no_error(self, direct_cfg):
        """No error when model and inputs are both on CPU."""
        from neural_pbf.models.surrogate import ThermalSurrogate3D

        cpu = torch.device("cpu")
        model = ThermalSurrogate3D(direct_cfg).eval().to(cpu)
        T_init = _make_const_volume().to(cpu)
        Q_seq = [_make_const_volume().to(cpu) for _ in range(2)]

        # Must not raise
        preds = model.predict_autoregressive(T_init, Q_seq, device=cpu)
        assert len(preds) == 2

    def test_implementation_uses_t_init_device_not_hardcoded_cpu(self, direct_cfg):
        """Verify the implementation inherits device from T_init, not hardcoded 'cpu'.

        We inspect the source code to confirm that the device default was changed
        from `torch.device('cpu')` to `T_init.device`.
        """
        import inspect

        from neural_pbf.models.surrogate import ThermalSurrogate3D

        source = inspect.getsource(ThermalSurrogate3D.predict_autoregressive)
        # The fixed implementation must reference T_init.device
        assert "T_init.device" in source, (
            "predict_autoregressive still uses a hardcoded device default. "
            "Change `device = torch.device('cpu')` to "
            "`device = T_init.device if device is None else device`."
        )
        # Must NOT have the hardcoded CPU default
        assert 'torch.device("cpu")' not in source and "torch.device('cpu')" not in source, (
            "predict_autoregressive still contains hardcoded `torch.device('cpu')`. "
            "Remove it and use `T_init.device` instead."
        )
