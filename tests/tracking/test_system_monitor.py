"""Unit tests for SystemMonitor — mocked at the _optional_deps boundary."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neural_pbf.tracking.instrumentation.system_monitor import SystemMonitor


def _make_nvml(*, raise_on=(), supported_fan=True):
    """Build a minimal pynvml mock."""
    nvml = MagicMock()

    # Constants
    nvml.NVML_TEMPERATURE_GPU = 0
    nvml.NVML_PCIE_UTIL_TX_BYTES = 1
    nvml.NVML_PCIE_UTIL_RX_BYTES = 2
    nvml.NVML_VOLTAGE_POWER_SUPPLY = 0

    nvml.nvmlInit.return_value = None
    nvml.nvmlDeviceGetHandleByIndex.return_value = object()
    nvml.nvmlShutdown.return_value = None

    rates = SimpleNamespace(gpu=85, memory=60)
    nvml.nvmlDeviceGetUtilizationRates.return_value = rates
    nvml.nvmlDeviceGetTemperature.return_value = 72
    nvml.nvmlDeviceGetPowerUsage.return_value = 150_000  # milliwatts

    if supported_fan:
        nvml.nvmlDeviceGetFanSpeed.return_value = 45
    else:
        nvml.nvmlDeviceGetFanSpeed.side_effect = Exception("NVMLError_NotSupported")

    nvml.nvmlDeviceGetVoltage.side_effect = Exception("NVMLError_NotSupported")
    nvml.nvmlDeviceGetPcieThroughput.side_effect = [1024, 512]  # TX then RX (KB/s)

    for name in raise_on:
        getattr(nvml, name).side_effect = RuntimeError("mock forced error")

    return nvml


def _make_psutil():
    ps = MagicMock()
    ps.cpu_percent.return_value = [10.0, 20.0, 15.0, 5.0]
    vm = SimpleNamespace(used=4 * 1024**3, available=12 * 1024**3)
    ps.virtual_memory.return_value = vm
    ct = SimpleNamespace(iowait=0.3)
    ps.cpu_times.return_value = ct
    dio = SimpleNamespace(read_bytes=100 * 1024**2, write_bytes=50 * 1024**2)
    ps.disk_io_counters.return_value = dio
    return ps


@pytest.mark.unit
class TestSystemMonitorNvml:
    def _monitor_with_mocks(self, nvml, psutil_mod=None):
        with (
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_nvml",
                return_value=(nvml, None),
            ),
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_psutil",
                return_value=(psutil_mod, None) if psutil_mod else (None, "no psutil"),
            ),
        ):
            m = SystemMonitor()
            m.start()
        return m

    def test_gpu_utilization_present(self):
        m = self._monitor_with_mocks(_make_nvml())
        sample = m.sample()
        assert sample["sys/gpu_utilization"] == pytest.approx(85.0)
        assert sample["sys/gpu_memory_utilization"] == pytest.approx(60.0)

    def test_power_converted_to_watts(self):
        m = self._monitor_with_mocks(_make_nvml())
        sample = m.sample()
        assert sample["sys/gpu_power_w"] == pytest.approx(150.0)

    def test_temperature_present(self):
        m = self._monitor_with_mocks(_make_nvml())
        sample = m.sample()
        assert sample["sys/gpu_temperature"] == pytest.approx(72.0)

    def test_unsupported_voltage_returns_none(self):
        m = self._monitor_with_mocks(_make_nvml())
        sample = m.sample()
        assert sample["sys/gpu_voltage"] is None

    def test_unsupported_fan_returns_none(self):
        m = self._monitor_with_mocks(_make_nvml(supported_fan=False))
        sample = m.sample()
        assert sample["sys/gpu_fan_speed"] is None

    def test_pcie_throughput_present(self):
        # nvmlDeviceGetPcieThroughput is called twice: TX then RX
        nvml = _make_nvml()
        nvml.nvmlDeviceGetPcieThroughput.side_effect = None
        nvml.nvmlDeviceGetPcieThroughput.return_value = 1024
        m = self._monitor_with_mocks(nvml)
        sample = m.sample()
        assert sample["sys/pcie_tx_kb"] is not None
        assert sample["sys/pcie_rx_kb"] is not None

    def test_no_nvml_returns_empty_gpu_keys(self):
        with (
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_nvml",
                return_value=(None, "no pynvml"),
            ),
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_psutil",
                return_value=(None, "no psutil"),
            ),
        ):
            m = SystemMonitor()
            m.start()
        sample = m.sample()
        assert sample == {}

    def test_stop_calls_nvml_shutdown(self):
        nvml = _make_nvml()
        m = self._monitor_with_mocks(nvml)
        m.stop()
        nvml.nvmlShutdown.assert_called_once()
        assert m._nvml is None


@pytest.mark.unit
class TestSystemMonitorPsutil:
    def _monitor_psutil_only(self):
        with (
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_nvml",
                return_value=(None, "no pynvml"),
            ),
            patch(
                "neural_pbf.tracking.instrumentation.system_monitor.try_import_psutil",
                return_value=(_make_psutil(), None),
            ),
        ):
            m = SystemMonitor()
            m.start()
        return m

    def test_per_core_cpu_keys(self):
        m = self._monitor_psutil_only()
        sample = m.sample()
        assert "sys/cpu_core_00" in sample
        assert "sys/cpu_core_03" in sample
        assert sample["sys/cpu_core_01"] == pytest.approx(20.0)

    def test_ram_keys_present(self):
        m = self._monitor_psutil_only()
        sample = m.sample()
        assert sample["sys/ram_used_mb"] == pytest.approx(4 * 1024.0)
        assert sample["sys/ram_available_mb"] == pytest.approx(12 * 1024.0)

    def test_iowait_present(self):
        m = self._monitor_psutil_only()
        sample = m.sample()
        assert sample["sys/iowait_pct"] == pytest.approx(0.3)

    def test_disk_io_present(self):
        m = self._monitor_psutil_only()
        sample = m.sample()
        assert sample["sys/disk_read_mb"] == pytest.approx(100.0)
        assert sample["sys/disk_write_mb"] == pytest.approx(50.0)
