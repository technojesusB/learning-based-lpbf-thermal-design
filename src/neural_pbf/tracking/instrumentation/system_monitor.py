"""Hardware metrics collector: NVML (GPU) + psutil (CPU/RAM/Disk)."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from ._optional_deps import try_import_nvml, try_import_psutil

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Samples GPU and system metrics per simulation step.

    Handles missing/unsupported metrics gracefully — each metric that cannot
    be read is returned as None and a one-time warning is emitted.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._nvml: Any | None = None
        self._nvml_handle: Any | None = None
        self._psutil: Any | None = None
        self._unsupported: set[str] = set()
        self._started = False

    def start(self) -> None:
        nvml, nvml_err = try_import_nvml()
        if nvml is not None:
            try:
                nvml.nvmlInit()
                self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(self._device_index)
                self._nvml = nvml
                logger.info("NVML initialised (device %d).", self._device_index)
            except Exception as exc:
                logger.warning("NVML init failed: %s. GPU metrics disabled.", exc)

        psutil_mod, psutil_err = try_import_psutil()
        if psutil_mod is not None:
            self._psutil = psutil_mod
            # Warm up percpu so the first call returns a non-zero reading
            psutil_mod.cpu_percent(percpu=True)

        self._started = True

    def sample(self) -> dict[str, float | None]:
        """Return a flat dict of all available metrics (None = unsupported)."""
        result: dict[str, float | None] = {}
        if self._nvml is not None and self._nvml_handle is not None:
            result.update(self._sample_nvml())
        if self._psutil is not None:
            result.update(self._sample_psutil())
        return result

    def stop(self) -> None:
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        self._nvml = None
        self._nvml_handle = None
        self._started = False

    # ── NVML ─────────────────────────────────────────────────────────────────

    def _nvml_get(self, key: str, fn, *args) -> float | None:
        if key in self._unsupported:
            return None
        try:
            return float(fn(*args))
        except Exception as exc:
            name = type(exc).__name__
            if "NotSupported" in name or "NoPermission" in name:
                if key not in self._unsupported:
                    logger.debug("NVML metric '%s' not supported on this device.", key)
                    self._unsupported.add(key)
            else:
                logger.debug("NVML metric '%s' error: %s", key, exc)
            return None

    def _sample_nvml(self) -> dict[str, float | None]:
        if self._nvml is None or self._nvml_handle is None:
            # Fallback for WSL2 where native NVML often fails
            return self._sample_wsl_fallback()

        nvml = self._nvml
        h = self._nvml_handle

        util = None
        mem_util = None
        try:
            rates = nvml.nvmlDeviceGetUtilizationRates(h)
            util = float(rates.gpu)
            mem_util = float(rates.memory)
        except Exception:
            pass

        power_mw = self._nvml_get("sys/gpu_power_mw", nvml.nvmlDeviceGetPowerUsage, h)
        power_w = power_mw / 1000.0 if power_mw is not None else None

        temp = self._nvml_get(
            "sys/gpu_temperature",
            nvml.nvmlDeviceGetTemperature,
            h,
            nvml.NVML_TEMPERATURE_GPU,
        )
        fan = self._nvml_get("sys/gpu_fan_speed", nvml.nvmlDeviceGetFanSpeed, h)

        voltage = self._nvml_get(
            "sys/gpu_voltage",
            lambda hh: nvml.nvmlDeviceGetVoltage(hh, nvml.NVML_VOLTAGE_POWER_SUPPLY),
            h,
        )

        pcie_tx = self._nvml_get(
            "sys/pcie_tx_kb",
            nvml.nvmlDeviceGetPcieThroughput,
            h,
            nvml.NVML_PCIE_UTIL_TX_BYTES,
        )
        pcie_rx = self._nvml_get(
            "sys/pcie_rx_kb",
            nvml.nvmlDeviceGetPcieThroughput,
            h,
            nvml.NVML_PCIE_UTIL_RX_BYTES,
        )

        return {
            "sys/gpu_utilization": util,
            "sys/gpu_memory_utilization": mem_util,
            "sys/gpu_temperature": temp,
            "sys/gpu_fan_speed": fan,
            "sys/gpu_power_w": power_w,
            "sys/gpu_voltage": voltage,
            "sys/pcie_tx_kb": pcie_tx,
            "sys/pcie_rx_kb": pcie_rx,
        }

    def _sample_wsl_fallback(self) -> dict[str, float | None]:
        """Call Windows nvidia-smi.exe as a last resort (slow but works in WSL)."""
        import subprocess
        from pathlib import Path

        exe = Path("/mnt/c/Windows/System32/nvidia-smi.exe")
        if not exe.exists():
            return {}

        try:
            # Query multiple metrics at once to minimize overhead
            cmd = [
                str(exe),
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,clocks.gr",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, encoding="utf-8", timeout=2.0).strip()
            if not out:
                return {}

            # Take only the first line in case of multiple GPUs
            first_line = out.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]

            def _f(idx: int) -> float | None:
                try:
                    val = parts[idx]
                    return float(val) if val not in ("N/A", "[Not Supported]") else None
                except (IndexError, ValueError):
                    return None

            return {
                "sys/gpu_utilization": _f(0),
                "sys/gpu_memory_utilization": _f(1),
                "sys/gpu_temperature": _f(2),
                "sys/gpu_fan_speed": _f(3),
                "sys/gpu_power_w": _f(4),
                "sys/gpu_clock_mhz": _f(5),
            }
        except Exception:
            return {}

    # ── psutil ────────────────────────────────────────────────────────────────

    def _sample_psutil(self) -> dict[str, float | None]:
        if self._psutil is None:
            return {}
        ps = self._psutil
        result: dict[str, float | None] = {}

        try:
            per_core = ps.cpu_percent(percpu=True)
            for i, val in enumerate(per_core):
                result[f"sys/cpu_core_{i:02d}"] = float(val)
        except Exception:
            pass

        try:
            vm = ps.virtual_memory()
            result["sys/ram_used_mb"] = vm.used / 1024**2
            result["sys/ram_available_mb"] = vm.available / 1024**2
        except Exception:
            pass

        try:
            ct = ps.cpu_times()
            iowait = getattr(ct, "iowait", None)
            if iowait is not None:
                result["sys/iowait_pct"] = float(iowait)
        except Exception:
            pass

        try:
            dio = ps.disk_io_counters()
            if dio is not None:
                result["sys/disk_read_mb"] = dio.read_bytes / 1024**2
                result["sys/disk_write_mb"] = dio.write_bytes / 1024**2
        except Exception:
            pass

        return result
