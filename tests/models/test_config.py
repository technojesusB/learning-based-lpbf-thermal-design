"""Tests for SurrogateConfig — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_default_strategy_is_direct():
    """Default strategy must be 'direct'."""
    from neural_pbf.models.config import SurrogateConfig

    cfg = SurrogateConfig()
    assert cfg.strategy == "direct"


def test_invalid_strategy_raises():
    """Passing an unknown strategy must raise ValidationError."""
    from neural_pbf.models.config import SurrogateConfig

    with pytest.raises(ValidationError):
        SurrogateConfig(strategy="invalid_strategy")  # type: ignore[arg-type]


def test_frozen_model_immutable():
    """Frozen model must raise an error on attribute assignment."""
    from neural_pbf.models.config import SurrogateConfig

    cfg = SurrogateConfig()
    with pytest.raises(ValidationError):
        cfg.lr = 1e-3  # type: ignore[misc]


def test_invalid_arch_raises():
    """Passing an unsupported architecture must raise ValidationError."""
    from neural_pbf.models.config import SurrogateConfig

    with pytest.raises(ValidationError):
        SurrogateConfig(arch="transformer3d")  # type: ignore[arg-type]


def test_negative_lr_raises():
    """A negative learning rate must be rejected."""
    from neural_pbf.models.config import SurrogateConfig

    with pytest.raises(ValidationError):
        SurrogateConfig(lr=-1e-4)


def test_pde_weight_range():
    """pde_weight must be a non-negative float; negative values raise."""
    from neural_pbf.models.config import SurrogateConfig

    # Valid: zero is acceptable
    cfg = SurrogateConfig(pde_weight=0.0)
    assert cfg.pde_weight == 0.0

    # Invalid: negative
    with pytest.raises(ValidationError):
        SurrogateConfig(pde_weight=-0.5)


def test_default_values():
    """All defaults must match specification."""
    from neural_pbf.models.config import SurrogateConfig

    cfg = SurrogateConfig()
    assert cfg.arch == "unet3d"
    assert cfg.in_channels == 2
    assert cfg.out_channels == 1
    assert cfg.base_channels == 32
    assert cfg.depth == 4
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.pde_weight == pytest.approx(0.1)
    assert cfg.batch_size == 4
    assert cfg.buffer_capacity == 2048
    assert cfg.patch_size == 64
    assert cfg.lf_coarsen_factor == 4
    assert cfg.lf_substep_factor == 1


def test_residual_strategy_valid():
    """'residual' strategy must be accepted."""
    from neural_pbf.models.config import SurrogateConfig

    cfg = SurrogateConfig(strategy="residual")
    assert cfg.strategy == "residual"
