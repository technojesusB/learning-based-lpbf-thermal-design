import logging
import time
from typing import Any

import torch

from ..schemas.diagnostics import DiagnosticsConfig

logger = logging.getLogger(__name__)


class DiagnosticsRecorder:
    """Records diagnostics metrics and performs stability checks."""

    def __init__(self, cfg: DiagnosticsConfig):
        self.cfg = cfg
        # State tracking
        self.step_start_time = 0.0
        self.last_T = None

        # Energy integration
        self.total_energy_J = 0.0

    def on_step_start(self, state: Any, meta: dict[str, Any]):
        self.step_start_time = time.perf_counter()

    def on_after_update(self, prev_state: Any, new_state: Any, meta: dict[str, Any]):
        # Store for diff calculation
        # assuming state is tensor T or dict with 'T'
        pass

    def on_step_end(
        self,
        step_idx: int,
        state: Any,
        prev_state_T: torch.Tensor | None,
        meta: dict[str, Any],
    ) -> dict[str, float]:
        """Compute metrics for the step.

        Args:
            step_idx: Current step index
            state: Current state (dict or tensor)
            prev_state_T: Tensor of temperature from previous step (for dT calcs)
            meta: Metadata dict (e.g. scan_power)
        """
        metrics: dict[str, float | int | None] = {}

        if not self.cfg.enabled:
            return metrics

        # Extract T
        if isinstance(state, dict):
            T = state.get("T", state.get("temperature"))
        else:
            T = state

        if T is None:
            return metrics

        # Basic stats
        metrics["sim/temperature_min"] = float(T.min())
        metrics["sim/temperature_max"] = float(T.max())
        metrics["sim/temperature_mean"] = float(T.mean())
        metrics["sim/temperature_std"] = float(T.std())

        # L1/L2 "Energy" proxies
        metrics["energy/temperature_l1"] = float(torch.abs(T).sum())
        metrics["energy/temperature_l2"] = float(torch.norm(T, p=2))

        # Real Energy Integration
        # meta should contain 'dt' and current 'power'
        dt = meta.get("dt", 0.0)
        power = meta.get("scan_power", 0.0)
        metrics["energy/power_in_W"] = power
        self.total_energy_J += power * dt
        metrics["energy/energy_in_J"] = self.total_energy_J

        # Stability Checks
        if self.cfg.check_nan_inf:
            nan_count = torch.isnan(T).sum().item()
            inf_count = torch.isinf(T).sum().item()
            metrics["stability/nan_count"] = nan_count
            metrics["stability/inf_count"] = inf_count

            if self.cfg.strict and (nan_count > 0 or inf_count > 0):
                raise RuntimeError(
                    f"Stability check failed: NaN={nan_count}, Inf={inf_count}"
                )

        # Dynamics / Residuals
        if prev_state_T is not None:
            diff = T - prev_state_T
            max_abs_dT = float(torch.abs(diff).max())
            mean_abs_dT = float(torch.abs(diff).mean())

            metrics["stability/max_abs_dT"] = max_abs_dT
            metrics["stability/mean_abs_dT"] = mean_abs_dT

            # Update ratio heuristic
            max_abs_T = float(torch.abs(T).max())
            update_ratio = max_abs_dT / (max_abs_T + 1e-6)
            metrics["stability/update_ratio"] = update_ratio

            # Residual proxy
            norm_T = float(torch.norm(prev_state_T) + 1e-6)
            norm_diff = float(torch.norm(diff))
            metrics["solver/residual_proxy"] = norm_diff / norm_T

        # Performance
        step_end_time = time.perf_counter()
        step_duration = step_end_time - self.step_start_time
        metrics["perf/step_time_ms"] = step_duration * 1000.0
        metrics["perf/steps_per_sec"] = 1.0 / (step_duration + 1e-9)

        if self.cfg.memory_profile and torch.cuda.is_available():
            metrics["perf/gpu_mem_alloc_mb"] = torch.cuda.memory_allocated() / 1e6
            metrics["perf/gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6

        # Threshold Policy
        # e.g. check thresholds from config
        warn_flag = 0
        for name, limit in self.cfg.thresholds.items():
            val = metrics.get(name)
            if val is not None and val > limit:
                warn_flag = 1
                logger.warning(f"Metric {name} exceeded threshold: {val} > {limit}")
                if self.cfg.strict:
                    raise RuntimeError(
                        f"Metric {name} exceeded threshold: {val} > {limit}"
                    )

        metrics["stability/warn_flag"] = warn_flag

        return metrics
