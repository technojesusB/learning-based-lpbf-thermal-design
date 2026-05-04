"""Unit tests for FlightRecorder."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neural_pbf.tracking.instrumentation.flight_recorder import FlightRecorder


@pytest.mark.unit
class TestFlightRecorderBuffer:
    def test_ring_buffer_keeps_last_n(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        for i in range(10):
            fr.record_step({"step_idx": i, "value": i * 2})
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        steps = payload["last_steps"]
        assert len(steps) == 5
        assert steps[0]["step_idx"] == 5
        assert steps[-1]["step_idx"] == 9

    def test_fewer_steps_than_capacity(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        fr.record_step({"step_idx": 0})
        fr.record_step({"step_idx": 1})
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        assert len(payload["last_steps"]) == 2

    def test_empty_buffer_dumps_cleanly(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        assert payload["last_steps"] == []


@pytest.mark.unit
class TestFlightRecorderDump:
    def test_exception_info_captured(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        fr.record_step({"step_idx": 0, "t_sim": 1.23})
        try:
            raise ValueError("simulated CUDA crash")
        except ValueError as exc:
            out = fr.dump(tmp_path / "blackbox.json", exc=exc)
        payload = json.loads(out.read_text())
        assert payload["exception_type"] == "ValueError"
        assert "simulated CUDA crash" in payload["exception_repr"]
        assert "ValueError" in payload["traceback"]

    def test_no_exception_fields_are_none(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        out = fr.dump(tmp_path / "blackbox.json", exc=None)
        payload = json.loads(out.read_text())
        assert payload["exception_type"] is None
        assert payload["traceback"] is None

    def test_run_context_stored(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        fr.set_run_context(
            mat_params={"mat_key": "SS316L", "k_solid": 16.3},
            run_info={"run_idx": 7},
        )
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        assert payload["run_meta"]["material_params"]["mat_key"] == "SS316L"
        assert payload["run_meta"]["run_info"]["run_idx"] == 7

    def test_env_block_present(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        assert "python_version" in payload["env"]
        assert "platform" in payload["env"]

    def test_no_torch_cuda_calls_in_dump(self, tmp_path):
        """dump() must not call any torch.cuda APIs — safe after CUDA error."""
        fr = FlightRecorder(capacity=5)
        fr.record_step({"step_idx": 0})

        cuda_mock = pytest.importorskip("unittest.mock").MagicMock()
        with patch("torch.cuda", cuda_mock):
            try:
                raise RuntimeError("cuda error")
            except RuntimeError as exc:
                fr.dump(tmp_path / "blackbox.json", exc=exc)

        cuda_mock.synchronize.assert_not_called()
        cuda_mock.empty_cache.assert_not_called()

    def test_numpy_arrays_serialised(self, tmp_path):
        import numpy as np

        fr = FlightRecorder(capacity=5)
        fr.record_step({"array": np.array([1.0, 2.0, 3.0])})
        out = fr.dump(tmp_path / "blackbox.json")
        payload = json.loads(out.read_text())
        assert payload["last_steps"][0]["array"] == [1.0, 2.0, 3.0]

    def test_returns_path(self, tmp_path):
        fr = FlightRecorder(capacity=5)
        out = fr.dump(tmp_path / "blackbox.json")
        assert isinstance(out, Path)
        assert out.exists()
