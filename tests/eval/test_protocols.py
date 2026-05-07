"""Tests for BaseStepper protocol compliance."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.protocols import BaseStepper


@pytest.mark.unit
def test_identity_adapter_satisfies_protocol():
    adapter = IdentityAdapter()
    assert isinstance(adapter, BaseStepper)


@pytest.mark.unit
def test_protocol_requires_step_method():
    class NoStep:
        name = "bad"

    assert not isinstance(NoStep(), BaseStepper)


@pytest.mark.unit
def test_protocol_requires_name():
    # A class with step but without name is not structural subtype — runtime check
    # only validates presence of 'step'; name is checked at type-check time.
    # This test documents that 'name' is a required attribute in practice.
    adapter = IdentityAdapter()
    assert hasattr(adapter, "name")
    assert adapter.name == "identity"


@pytest.mark.unit
def test_triton_adapter_satisfies_protocol(sim_cfg, mat_cfg):
    from neural_pbf.eval.adapters.triton_adapter import TritonAdapter

    adapter = TritonAdapter(sim_cfg, mat_cfg, use_triton=False)
    assert isinstance(adapter, BaseStepper)
    assert adapter.name == "triton"
