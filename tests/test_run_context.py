from unittest.mock import MagicMock

import pytest
import torch

from neural_pbf.schemas.artifacts import ArtifactConfig
from neural_pbf.schemas.diagnostics import DiagnosticsConfig
from neural_pbf.schemas.run_meta import RunMeta
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.run_context import RunContext


@pytest.fixture
def run_meta():
    return RunMeta(
        seed=0,
        device="cpu",
        dtype="float32",
        started_at="now",
        dx=0.1,
        dy=0.1,
        dz=0.1,
        dt=0.01,
        grid_shape=[10, 10],
    )


def test_run_context_lifecycle(tmp_path, run_meta):
    t_cfg = TrackingConfig(enabled=False, run_name="test_run")
    a_cfg = ArtifactConfig(enabled=False)
    d_cfg = DiagnosticsConfig(enabled=False)

    ctx = RunContext(t_cfg, a_cfg, d_cfg, run_meta, tmp_path)

    # Mock tracker to verify calls
    ctx.tracker = MagicMock()
    ctx.tracker.start_run.return_value = MagicMock()

    ctx.start()
    ctx.tracker.start_run.assert_called_once()

    ctx.log_step(0, {"T": torch.tensor(0.0)}, {})
    ctx.end({"T": torch.tensor(0.0)})

    # Check mocked context exit called
    ctx._tracker_ctx.__exit__.assert_called()


def test_run_context_cadence(tmp_path, run_meta):
    t_cfg = TrackingConfig(enabled=True, log_every_n_steps=2)
    a_cfg = ArtifactConfig(enabled=False)
    d_cfg = DiagnosticsConfig(enabled=False)

    ctx = RunContext(t_cfg, a_cfg, d_cfg, run_meta, tmp_path)
    ctx.tracker = MagicMock()
    ctx.tracker.start_run.return_value = MagicMock()
    ctx.start()

    # Step 1 (should NOT log)
    ctx.log_step(1, {"T": torch.tensor(0.0)})
    ctx.tracker.log_metrics.assert_not_called()

    # Step 2 (should log)
    ctx.log_step(2, {"T": torch.tensor(0.0)})
    ctx.tracker.log_metrics.assert_called_once()
