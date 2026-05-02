"""Tests for training_pipeline — TDD RED phase.

All tests run on CPU with tiny grids (4×4×4, 3 time steps) for speed.
"""

from __future__ import annotations

import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.models.config import SurrogateConfig
from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
from neural_pbf.models.surrogate import ThermalSurrogate3D
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.scan.sources import GaussianSourceConfig
from neural_pbf.utils.units import LengthUnit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim_cfg() -> SimulationConfig:
    """Tiny 3D sim config (4×4×4 grid, 1m domain in each axis)."""
    return SimulationConfig(
        Lx=4.0,
        Ly=4.0,
        Lz=4.0,
        Nx=4,
        Ny=4,
        Nz=4,
        length_unit=LengthUnit.METERS,
        dt_base=1e-5,
        T_ambient=300.0,
    )


@pytest.fixture()
def mat_cfg() -> MaterialConfig:
    return MaterialConfig(
        k_powder=1.0,
        k_solid=10.0,
        k_liquid=10.0,
        cp_base=500.0,
        rho=1000.0,
        T_solidus=1000.0,
        T_liquidus=1100.0,
        latent_heat_L=0.0,
    )


@pytest.fixture()
def surrogate_cfg() -> SurrogateConfig:
    return SurrogateConfig(
        strategy="direct",
        base_channels=4,
        depth=1,
        patch_size=4,
        batch_size=2,
        lr=1e-3,
        pde_weight=0.01,
    )


@pytest.fixture()
def beam_cfg() -> GaussianSourceConfig:
    """Gaussian beam with volumetric source (depth provided)."""
    return GaussianSourceConfig(
        power=200.0,
        eta=0.35,
        sigma=0.5,  # 0.5 m sigma (matches the 4m domain)
        depth=0.5,  # volumetric, 0.5 m penetration depth
    )


@pytest.fixture()
def scan_positions() -> list[tuple[float, float, float]]:
    """Three scan positions in the 4m domain."""
    return [(1.0, 1.0, 4.0), (2.0, 1.0, 4.0), (3.0, 1.0, 4.0)]


@pytest.fixture()
def cpu() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_direct_surrogate(cfg: SurrogateConfig) -> ThermalSurrogate3D:
    torch.manual_seed(0)
    return ThermalSurrogate3D(cfg)


def _make_small_buffer(capacity: int = 10, patch_size: int = 4) -> ExperienceReplayBuffer:
    return ExperienceReplayBuffer(capacity=capacity, patch_size=patch_size)


# ---------------------------------------------------------------------------
# Tests: module importability
# ---------------------------------------------------------------------------


class TestModuleImportability:
    def test_module_importable(self):
        from neural_pbf.pipelines import training_pipeline  # noqa: F401

        assert hasattr(training_pipeline, "generate_hf_dataset")
        assert hasattr(training_pipeline, "train_surrogates")
        assert hasattr(training_pipeline, "evaluate_autoregressive")

    def test_generate_hf_dataset_importable(self):
        from neural_pbf.pipelines.training_pipeline import (
            generate_hf_dataset,  # noqa: F401
        )

        assert callable(generate_hf_dataset)

    def test_train_surrogates_importable(self):
        from neural_pbf.pipelines.training_pipeline import (
            train_surrogates,  # noqa: F401
        )

        assert callable(train_surrogates)

    def test_evaluate_autoregressive_importable(self):
        from neural_pbf.pipelines.training_pipeline import (
            evaluate_autoregressive,  # noqa: F401
        )

        assert callable(evaluate_autoregressive)


# ---------------------------------------------------------------------------
# Tests: generate_hf_dataset
# ---------------------------------------------------------------------------


class TestGenerateHfDataset:
    """Tests for generate_hf_dataset function."""

    def test_returns_experience_replay_buffer(
        self, sim_cfg, mat_cfg, beam_cfg, scan_positions, cpu
    ):
        """generate_hf_dataset must return an ExperienceReplayBuffer."""
        from neural_pbf.pipelines.training_pipeline import generate_hf_dataset

        buffer = generate_hf_dataset(
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            scan_positions=scan_positions,
            dt_macro=1e-5,
            buffer_capacity=20,
            patch_size=4,
            device=cpu,
            beam_cfg=beam_cfg,
        )

        assert isinstance(buffer, ExperienceReplayBuffer), (
            f"Expected ExperienceReplayBuffer, got {type(buffer)}"
        )

    def test_buffer_has_correct_number_of_entries(
        self, sim_cfg, mat_cfg, beam_cfg, scan_positions, cpu
    ):
        """Buffer must contain exactly len(scan_positions) entries after generation."""
        from neural_pbf.pipelines.training_pipeline import generate_hf_dataset

        buffer = generate_hf_dataset(
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            scan_positions=scan_positions,
            dt_macro=1e-5,
            buffer_capacity=20,
            patch_size=4,
            device=cpu,
            beam_cfg=beam_cfg,
        )

        assert len(buffer) == len(scan_positions), (
            f"Buffer should have {len(scan_positions)} entries "
            f"(one per scan position), got {len(buffer)}."
        )

    def test_buffer_entries_stored_on_cpu(
        self, sim_cfg, mat_cfg, beam_cfg, scan_positions, cpu
    ):
        """Buffer entries must be stored on CPU (regardless of computation device)."""
        from neural_pbf.pipelines.training_pipeline import generate_hf_dataset

        buffer = generate_hf_dataset(
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            scan_positions=scan_positions,
            dt_macro=1e-5,
            buffer_capacity=20,
            patch_size=4,
            device=cpu,
            beam_cfg=beam_cfg,
        )

        batch = buffer.sample(min(len(buffer), 2))
        for key, tensor in batch.items():
            assert tensor.device.type == "cpu", (
                f"Buffer key '{key}' is on {tensor.device}, expected cpu."
            )

    def test_sampled_batch_has_mandatory_keys(
        self, sim_cfg, mat_cfg, beam_cfg, scan_positions, cpu
    ):
        """Sampled batch must contain T_in, Q, T_target."""
        from neural_pbf.pipelines.training_pipeline import generate_hf_dataset

        buffer = generate_hf_dataset(
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            scan_positions=scan_positions,
            dt_macro=1e-5,
            buffer_capacity=20,
            patch_size=4,
            device=cpu,
            beam_cfg=beam_cfg,
        )

        batch = buffer.sample(min(len(buffer), 2))
        assert "T_in" in batch
        assert "Q" in batch
        assert "T_target" in batch

    def test_state_built_with_zeros_classmethod(
        self, sim_cfg, mat_cfg, beam_cfg, scan_positions, cpu
    ):
        """generate_hf_dataset must use SimulationState.zeros (not manual construction).

        We verify indirectly: the initial temperature at scan start must equal
        T_ambient from the config (the SimulationState.zeros default behavior).
        """
        from neural_pbf.pipelines.training_pipeline import generate_hf_dataset

        # Use T_ambient = 500.0 to distinguish from zero-filled state
        cfg_custom = sim_cfg.model_copy(update={"T_ambient": 500.0})

        buffer = generate_hf_dataset(
            sim_cfg=cfg_custom,
            mat_cfg=mat_cfg,
            scan_positions=[(2.0, 2.0, 4.0)],  # Single position
            dt_macro=1e-5,
            buffer_capacity=10,
            patch_size=4,
            device=cpu,
            beam_cfg=beam_cfg,
        )

        batch = buffer.sample(1)
        # T_in should have values at or near T_ambient (300 K or 500 K depending on config)
        t_in = batch["T_in"]
        assert not torch.all(t_in == 0.0), (
            "T_in is all zeros — SimulationState.zeros was not used correctly. "
            "Initial temperature must be T_ambient."
        )


# ---------------------------------------------------------------------------
# Tests: train_surrogates
# ---------------------------------------------------------------------------


class TestTrainSurrogates:
    """Tests for train_surrogates function."""

    def _make_filled_buffer(
        self, capacity: int = 8, patch_size: int = 4
    ) -> ExperienceReplayBuffer:
        buf = ExperienceReplayBuffer(capacity=capacity, patch_size=patch_size)
        for _ in range(capacity):
            shape = (1, 1, patch_size, patch_size, patch_size)
            buf.push(
                torch.rand(shape) * 500 + 300,
                torch.rand(shape) * 1e9,
                torch.rand(shape) * 500 + 300,
            )
        return buf

    def test_returns_dict_with_loss_history(
        self, sim_cfg, mat_cfg, surrogate_cfg, cpu
    ):
        """train_surrogates must return a dict containing 'direct_loss' key."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        result = train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=2,
            dt=1e-5,
        )

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "direct_loss" in result, (
            "train_surrogates result must contain 'direct_loss' key."
        )

    def test_direct_loss_has_one_entry_per_epoch(
        self, sim_cfg, mat_cfg, surrogate_cfg, cpu
    ):
        """direct_loss list must have exactly num_epochs entries."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        num_epochs = 3
        result = train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=num_epochs,
            dt=1e-5,
        )

        assert len(result["direct_loss"]) == num_epochs, (
            f"Expected {num_epochs} loss entries, got {len(result['direct_loss'])}."
        )

    def test_loss_values_are_finite_floats(
        self, sim_cfg, mat_cfg, surrogate_cfg, cpu
    ):
        """All loss values in the history must be finite Python floats."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        result = train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=2,
            dt=1e-5,
        )

        for i, val in enumerate(result["direct_loss"]):
            assert isinstance(val, float), f"direct_loss[{i}] is not float: {type(val)}"
            assert not (val != val), f"direct_loss[{i}] is NaN"
            assert val < float("inf"), f"direct_loss[{i}] is Inf"

    def test_model_moved_to_device(self, sim_cfg, mat_cfg, surrogate_cfg, cpu):
        """After train_surrogates, model parameters must reside on the requested device."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=1,
            dt=1e-5,
        )

        for p in model.parameters():
            assert p.device.type == "cpu", (
                f"Model parameter is on {p.device} after training, expected cpu."
            )

    def test_residual_loss_key_when_residual_model_is_none(
        self, sim_cfg, mat_cfg, surrogate_cfg, cpu
    ):
        """When surrogate_residual is None, 'residual_loss' must still be present but empty."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        result = train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=2,
            dt=1e-5,
        )

        assert "residual_loss" in result, (
            "'residual_loss' key must be present even when surrogate_residual=None."
        )
        assert result["residual_loss"] == [], (
            "When no residual model, 'residual_loss' must be an empty list."
        )

    def test_with_residual_model_trains_both(self, sim_cfg, mat_cfg, surrogate_cfg, cpu):
        """When surrogate_residual is provided, residual_loss must be populated."""
        from neural_pbf.pipelines.training_pipeline import train_surrogates

        residual_cfg = SurrogateConfig(
            strategy="residual",
            base_channels=4,
            depth=1,
            patch_size=4,
            batch_size=2,
            lr=1e-3,
        )
        direct_model = _make_direct_surrogate(surrogate_cfg)
        torch.manual_seed(1)
        residual_model = ThermalSurrogate3D(residual_cfg)

        buf = ExperienceReplayBuffer(capacity=8, patch_size=4)
        for _ in range(4):
            shape = (1, 1, 4, 4, 4)
            T_in = torch.rand(shape) * 500 + 300
            Q = torch.rand(shape) * 1e9
            T_target = torch.rand(shape) * 500 + 300
            T_lf = torch.rand(shape) * 500 + 300
            buf.push(T_in, Q, T_target, T_lf=T_lf)

        result = train_surrogates(
            buffer=buf,
            surrogate_direct=direct_model,
            surrogate_residual=residual_model,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=2,
            dt=1e-5,
        )

        assert len(result["residual_loss"]) == 2, (
            "residual_loss must have 2 entries when surrogate_residual is provided."
        )
        for val in result["residual_loss"]:
            assert isinstance(val, float)
            assert not (val != val), "residual_loss contains NaN"

    def test_tracker_called_when_provided(self, sim_cfg, mat_cfg, surrogate_cfg, cpu):
        """When tracker is provided with log_metrics, it must be called each epoch."""
        from unittest.mock import MagicMock

        from neural_pbf.pipelines.training_pipeline import train_surrogates

        model = _make_direct_surrogate(surrogate_cfg)
        buffer = self._make_filled_buffer()

        tracker = MagicMock()
        tracker.log_metrics = MagicMock()

        train_surrogates(
            buffer=buffer,
            surrogate_direct=model,
            surrogate_residual=None,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=2,
            dt=1e-5,
            tracker=tracker,
        )

        assert tracker.log_metrics.call_count == 2, (
            f"tracker.log_metrics must be called once per epoch (2), "
            f"got {tracker.log_metrics.call_count}"
        )

    def test_tracker_called_with_residual_model(self, sim_cfg, mat_cfg, surrogate_cfg, cpu):
        """Tracker must receive residual_loss when both models and tracker are provided."""
        from unittest.mock import MagicMock

        from neural_pbf.pipelines.training_pipeline import train_surrogates

        residual_cfg = SurrogateConfig(
            strategy="residual",
            base_channels=4,
            depth=1,
            patch_size=4,
            batch_size=2,
            lr=1e-3,
        )
        direct_model = _make_direct_surrogate(surrogate_cfg)
        torch.manual_seed(2)
        residual_model = ThermalSurrogate3D(residual_cfg)

        buf = ExperienceReplayBuffer(capacity=8, patch_size=4)
        for _ in range(4):
            shape = (1, 1, 4, 4, 4)
            T_in = torch.rand(shape) * 500 + 300
            Q = torch.rand(shape) * 1e9
            T_target = torch.rand(shape) * 500 + 300
            T_lf = torch.rand(shape) * 500 + 300
            buf.push(T_in, Q, T_target, T_lf=T_lf)

        tracker = MagicMock()
        tracker.log_metrics = MagicMock()

        train_surrogates(
            buffer=buf,
            surrogate_direct=direct_model,
            surrogate_residual=residual_model,
            sim_cfg=sim_cfg,
            mat_cfg=mat_cfg,
            surrogate_cfg=surrogate_cfg,
            device=cpu,
            num_epochs=1,
            dt=1e-5,
            tracker=tracker,
        )

        assert tracker.log_metrics.call_count == 1
        call_kwargs = tracker.log_metrics.call_args[0][0]
        assert "residual_loss" in call_kwargs, (
            "Tracker must log residual_loss when both models are provided."
        )

    def test_uses_sample_to_not_sample(self, sim_cfg, mat_cfg, surrogate_cfg, cpu):
        """train_surrogates must use buffer.sample_to to move batches to device.

        We verify this by checking the source code references sample_to.
        """
        import inspect

        from neural_pbf.pipelines import training_pipeline

        source = inspect.getsource(training_pipeline.train_surrogates)
        assert "sample_to" in source, (
            "train_surrogates must call buffer.sample_to(batch_size, device) "
            "instead of buffer.sample(batch_size) to ensure tensors are on the "
            "correct device."
        )


# ---------------------------------------------------------------------------
# Tests: evaluate_autoregressive
# ---------------------------------------------------------------------------


class TestEvaluateAutoregressive:
    """Tests for evaluate_autoregressive function."""

    def _make_volume(self, size: int = 4) -> torch.Tensor:
        return torch.rand(1, 1, size, size, size) * 500 + 300

    def test_returns_dict(self, surrogate_cfg, cpu):
        """evaluate_autoregressive must return a dict."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 3
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_result_has_mae_per_step_key(self, surrogate_cfg, cpu):
        """Result dict must contain 'mae_per_step' (list of MAE per step)."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 3
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        assert "mae_per_step" in result, (
            "evaluate_autoregressive result must contain 'mae_per_step'."
        )

    def test_mae_per_step_has_correct_length(self, surrogate_cfg, cpu):
        """mae_per_step must have exactly n_steps entries."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 4
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        assert len(result["mae_per_step"]) == n_steps, (
            f"Expected {n_steps} MAE values, got {len(result['mae_per_step'])}."
        )

    def test_mae_values_are_python_floats(self, surrogate_cfg, cpu):
        """All MAE values must be Python floats (scalars, not tensors)."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 2
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        for i, val in enumerate(result["mae_per_step"]):
            assert isinstance(val, float), (
                f"mae_per_step[{i}] must be a Python float, got {type(val)}"
            )

    def test_mae_values_are_non_negative(self, surrogate_cfg, cpu):
        """MAE values must be >= 0."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 3
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        for i, val in enumerate(result["mae_per_step"]):
            assert val >= 0.0, f"mae_per_step[{i}] is negative: {val}"

    def test_mae_is_zero_when_prediction_equals_gt(self, surrogate_cfg, cpu):
        """When predictions exactly match ground truth, MAE per step must be 0.0."""
        from unittest.mock import patch

        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 2
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        # Ground truth equals what model would predict — we patch to make model return GT
        gt_values = [self._make_volume() for _ in range(n_steps)]

        # Patch predict_autoregressive to return exact gt values
        with patch.object(
            model, "predict_autoregressive", return_value=gt_values
        ):
            result = evaluate_autoregressive(
                surrogate=model,
                T_init=T_init,
                Q_sequence=Q_seq,
                gt_sequence=gt_values,
                device=cpu,
            )

        for i, val in enumerate(result["mae_per_step"]):
            assert val == pytest.approx(0.0, abs=1e-6), (
                f"mae_per_step[{i}] should be 0.0 when predictions == ground truth, "
                f"got {val}"
            )

    def test_all_inputs_moved_to_device(self, surrogate_cfg, cpu):
        """Inputs on CPU must be correctly processed when device=cpu."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval().to(cpu)
        n_steps = 2
        T_init = self._make_volume().to(cpu)
        Q_seq = [self._make_volume().to(cpu) for _ in range(n_steps)]
        gt_seq = [self._make_volume().to(cpu) for _ in range(n_steps)]

        # Must not raise
        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )
        assert "mae_per_step" in result

    def test_result_has_mean_mae_key(self, surrogate_cfg, cpu):
        """Result dict must also contain 'mean_mae' as a float summary."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        model = _make_direct_surrogate(surrogate_cfg).eval()
        n_steps = 3
        T_init = self._make_volume()
        Q_seq = [self._make_volume() for _ in range(n_steps)]
        gt_seq = [self._make_volume() for _ in range(n_steps)]

        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
        )

        assert "mean_mae" in result, "Result must contain 'mean_mae' scalar."
        assert isinstance(result["mean_mae"], float), (
            f"mean_mae must be a float, got {type(result['mean_mae'])}"
        )

    def test_t_lf_sequence_accepted_when_provided(self, cpu):
        """evaluate_autoregressive must accept T_lf_sequence for residual strategy."""
        from neural_pbf.pipelines.training_pipeline import evaluate_autoregressive

        residual_cfg = SurrogateConfig(
            strategy="residual",
            base_channels=4,
            depth=1,
            patch_size=4,
        )
        model = ThermalSurrogate3D(residual_cfg).eval()
        n_steps = 2
        size = 4

        T_init = torch.rand(1, 1, size, size, size) * 500 + 300
        Q_seq = [torch.rand(1, 1, size, size, size) * 1e9 for _ in range(n_steps)]
        gt_seq = [torch.rand(1, 1, size, size, size) * 500 + 300 for _ in range(n_steps)]
        lf_seq = [torch.rand(1, 1, size, size, size) * 500 + 300 for _ in range(n_steps)]

        # Must not raise
        result = evaluate_autoregressive(
            surrogate=model,
            T_init=T_init,
            Q_sequence=Q_seq,
            gt_sequence=gt_seq,
            device=cpu,
            T_lf_sequence=lf_seq,
        )
        assert "mae_per_step" in result
