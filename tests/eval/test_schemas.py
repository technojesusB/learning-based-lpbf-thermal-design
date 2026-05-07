"""Tests for Pydantic eval schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from neural_pbf.eval.schemas import EvalConfig, ProbeSpec, RolloutConfig


@pytest.mark.unit
def test_rollout_config_defaults():
    cfg = RolloutConfig()
    assert cfg.n_steps == 10
    assert cfg.mode == "one_step"
    assert cfg.divergence_T_max == 5000.0


@pytest.mark.unit
def test_rollout_config_custom():
    cfg = RolloutConfig(n_steps=50, mode="autoregressive")
    assert cfg.n_steps == 50
    assert cfg.mode == "autoregressive"


@pytest.mark.unit
def test_rollout_config_invalid_n_steps():
    with pytest.raises(ValidationError):
        RolloutConfig(n_steps=0)


@pytest.mark.unit
def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.log_mlflow is True
    assert cfg.rollout.n_steps == 10


@pytest.mark.unit
def test_probe_spec_immutable():
    p = ProbeSpec(name="center", ix=4, iy=4)
    with pytest.raises((AttributeError, ValidationError)):
        p.ix = 99  # type: ignore[misc]


@pytest.mark.unit
def test_probe_spec_iz_optional():
    p = ProbeSpec(name="surface", ix=3, iy=3)
    assert p.iz is None


@pytest.mark.unit
def test_eval_config_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        EvalConfig(unknown_field="oops")  # type: ignore[call-arg]
