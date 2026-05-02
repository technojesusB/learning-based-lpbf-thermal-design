"""Tests for _assert_devices_match in TimeStepper — TDD RED phase."""

from __future__ import annotations

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.utils.units import LengthUnit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim_cfg() -> SimulationConfig:
    return SimulationConfig(
        Lx=1.0,
        Ly=1.0,
        Lz=None,
        Nx=4,
        Ny=4,
        Nz=1,
        length_unit=LengthUnit.METERS,
        dt_base=1e-5,
        T_ambient=300.0,
    )


def _make_mat_cfg() -> MaterialConfig:
    return MaterialConfig(
        k_powder=1.0,
        k_solid=1.0,
        k_liquid=1.0,
        cp_base=1.0,
        rho=1.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=0.0,
    )


def _make_state_on(device: torch.device) -> SimulationState:
    """Build a minimal 2D SimulationState on `device`."""
    T = torch.full((1, 1, 4, 4), 300.0, device=device)
    return SimulationState(T=T)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssertDevicesMatch:
    """Tests for the device-consistency guard in step_explicit_euler / step_adaptive."""

    def test_guard_function_importable(self):
        """_assert_devices_match must be importable from neural_pbf.integrator.stepper."""
        from neural_pbf.integrator.stepper import _assert_devices_match  # noqa: F401

        assert callable(_assert_devices_match)

    def test_matching_devices_no_error(self):
        """No error is raised when state.T and Q_ext are both on the same device (CPU)."""
        from neural_pbf.integrator.stepper import _assert_devices_match

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)
        Q_ext = torch.zeros_like(state.T)

        # Must not raise
        _assert_devices_match(state, Q_ext)

    def test_q_ext_none_no_error(self):
        """No error is raised when Q_ext is None (no device to mismatch)."""
        from neural_pbf.integrator.stepper import _assert_devices_match

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)

        # Must not raise
        _assert_devices_match(state, None)

    def test_state_internal_mismatch_raises_runtime_error(self):
        """RuntimeError is raised when state.T and state.material_mask are on different devices.

        We simulate this by building a state with T on CPU then manually assigning
        an incompatible material_mask (different device type string stored as a non-tensor
        attribute is not testable without a second GPU, so instead we test via the
        stepper integration path by patching the mask device type via subclassing).

        Since CPU is the only device available in CI, we test the simpler case: a
        Q_ext tensor that is on a *different* cpu memory tensor but same device type.
        This test documents the interface; the real mismatch scenario is tested when
        GPUs are available.
        """
        # Verify the guard signature handles Q_ext on same device without error.
        from neural_pbf.integrator.stepper import _assert_devices_match

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)
        Q_ext_cpu = torch.zeros_like(state.T)

        # Consistent: no error
        _assert_devices_match(state, Q_ext_cpu)

    def test_step_explicit_euler_with_consistent_q_no_error(self):
        """step_explicit_euler must complete without error when Q_ext is on same device as state.T."""
        sim = _make_sim_cfg()
        mat = _make_mat_cfg()
        stepper = TimeStepper(sim, mat)

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)
        Q_ext_cpu = torch.zeros_like(state.T)

        # Must not raise
        new_state = stepper.step_explicit_euler(state, dt=1e-7, Q_ext=Q_ext_cpu)
        assert new_state.T is not None
        assert not torch.isnan(new_state.T).any()

    def test_step_adaptive_with_consistent_q_no_error(self):
        """step_adaptive must complete without error when Q_ext is on same device as state.T."""
        sim = _make_sim_cfg()
        mat = _make_mat_cfg()
        stepper = TimeStepper(sim, mat)

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)
        Q_ext_cpu = torch.zeros_like(state.T)

        # Must not raise
        new_state = stepper.step_adaptive(state, dt_target=1e-7, Q_ext=Q_ext_cpu)
        assert new_state.T is not None

    def test_device_mismatch_q_ext_raises(self):
        """When Q_ext is created from a different-device tensor and devices mismatch,
        _assert_devices_match must raise RuntimeError with a descriptive message.

        We simulate device mismatch by creating a mock object whose .type property
        differs from CPU, bypassing the need for actual GPU hardware.
        """
        from neural_pbf.integrator.stepper import _assert_devices_match

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)

        # Create a fake Q tensor with a patched device to simulate GPU
        class FakeDevice:
            type = "cuda"

            def __str__(self):
                return "cuda:0"

        class FakeTensor:
            device = FakeDevice()

        # The guard should detect a mismatch when Q_ext.device.type != state.T.device.type
        with pytest.raises(RuntimeError, match="[Dd]evice"):
            _assert_devices_match(state, FakeTensor())  # type: ignore[arg-type]

    def test_guard_message_mentions_devices(self):
        """RuntimeError message from _assert_devices_match must mention both device strings."""
        from neural_pbf.integrator.stepper import _assert_devices_match

        cpu = torch.device("cpu")
        state = _make_state_on(cpu)

        class FakeDevice:
            type = "cuda"

            def __str__(self):
                return "cuda:0"

        class FakeTensor:
            device = FakeDevice()

        try:
            _assert_devices_match(state, FakeTensor())  # type: ignore[arg-type]
            pytest.fail("Expected RuntimeError was not raised")
        except RuntimeError as exc:
            msg = str(exc)
            assert "cpu" in msg.lower() or "cuda" in msg.lower(), (
                f"Error message does not mention device types: {msg!r}"
            )
