"""RolloutEngine — drives a BaseStepper over a Trajectory and collects metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neural_pbf.core.state import SimulationState
from neural_pbf.eval.data.snapshot import Snapshot
from neural_pbf.eval.data.trajectory import Trajectory
from neural_pbf.eval.metrics.accuracy import mae_global, mae_melt_pool, max_error
from neural_pbf.eval.metrics.geometry import iou_melt_volumes, melt_pool_extent
from neural_pbf.eval.metrics.performance import (
    LatencyTimer,
    reset_vram_peak,
    vram_peak_bytes,
)
from neural_pbf.eval.protocols import BaseStepper

_DIVERGENCE_T_MAX = 5000.0  # [K] — abort threshold for AR rollouts


@dataclass
class RolloutResult:
    """Immutable output of :meth:`RolloutEngine.run`.

    Attributes:
        mode:             ``"one_step"`` or ``"autoregressive"``.
        stepper_name:     :attr:`BaseStepper.name` of the evaluated stepper.
        gt_snapshots:     GT snapshots corresponding to steps 1..N (targets).
        pred_T:           Predicted temperature tensors, one per evaluated step.
        per_step_metrics: Dict of scalar metrics per step.
        latencies_s:      Wall-clock time per step [s].
        vram_peak_bytes:  Peak CUDA memory during the rollout [bytes].
        diverged_at_step: Step index at which the rollout was aborted (or None).
    """

    mode: str
    stepper_name: str
    gt_snapshots: list[Snapshot]
    pred_T: list[torch.Tensor]
    per_step_metrics: list[dict[str, float]]
    latencies_s: list[float]
    vram_peak_bytes: int
    diverged_at_step: int | None = None


class RolloutEngine:
    """Evaluates a :class:`BaseStepper` over a :class:`Trajectory`.

    Supports two modes:

    ``"one_step"``
        Feed the GT snapshot at step *i* into the stepper and compare its
        output to GT[i+1].  Errors are independent across steps.

    ``"autoregressive"``
        Feed the *predicted* state from the previous step into the stepper.
        Errors accumulate — a good test of long-horizon stability.
    """

    def run(
        self,
        stepper: BaseStepper,
        gt_trajectory: Trajectory,
        mode: str = "one_step",
        conditioning_seq: list[dict[str, Any] | None] | None = None,
        T_solidus: float | None = None,
        T_liquidus: float | None = None,
    ) -> RolloutResult:
        """Execute the rollout and return collected metrics.

        Args:
            stepper:          Stepper to evaluate.
            gt_trajectory:    Ground truth (must have ≥ 2 snapshots).
            mode:             ``"one_step"`` or ``"autoregressive"``.
            conditioning_seq: Per-step conditioning dicts (for FM adapter).
                              Defaults to ``[None] * n_steps``.
            T_solidus:        Override solidus from trajectory's mat_cfg.
            T_liquidus:       Override liquidus from trajectory's mat_cfg.

        Returns:
            :class:`RolloutResult` with all collected data.
        """
        if len(gt_trajectory) < 2:
            raise ValueError("Trajectory must contain at least 2 snapshots.")
        if mode not in ("one_step", "autoregressive"):
            raise ValueError(
                f"mode must be 'one_step' or 'autoregressive', got {mode!r}"
            )

        n_steps = len(gt_trajectory) - 1
        cond_seq: list[dict[str, Any] | None] = (
            conditioning_seq if conditioning_seq is not None else [None] * n_steps
        )
        T_sol = T_solidus if T_solidus is not None else gt_trajectory.mat_cfg.T_solidus
        T_liq = (
            T_liquidus if T_liquidus is not None else gt_trajectory.mat_cfg.T_liquidus
        )

        reset_vram_peak()
        pred_T: list[torch.Tensor] = []
        latencies: list[float] = []
        metrics: list[dict[str, float]] = []
        diverged_at: int | None = None

        current_state = _snapshot_to_state(gt_trajectory.snapshots[0])

        for i in range(n_steps):
            snap_in = gt_trajectory.snapshots[i]
            snap_gt = gt_trajectory.snapshots[i + 1]

            state_in = (
                _snapshot_to_state(snap_in) if mode == "one_step" else current_state
            )

            with LatencyTimer() as timer:
                pred_state = stepper.step(
                    state=state_in,
                    Q_ext=snap_in.Q_ext,
                    dt=snap_in.dt,
                    conditioning=cond_seq[i],
                )

            T_pred = pred_state.T
            T_gt = snap_gt.T.to(T_pred.device)

            if (
                not torch.isfinite(T_pred).all()
                or T_pred.max().item() > _DIVERGENCE_T_MAX
            ):
                diverged_at = i
                break

            current_state = pred_state
            pred_T.append(T_pred.detach())
            latencies.append(timer.elapsed_s)

            extent_pred = melt_pool_extent(T_pred, T_liq)
            extent_gt = melt_pool_extent(T_gt, T_liq)

            step_metrics: dict[str, float] = {
                "mae_global": mae_global(T_pred, T_gt),
                "max_error": max_error(T_pred, T_gt),
                "mae_melt_pool": mae_melt_pool(T_pred, T_gt, T_sol),
                "iou_melt": iou_melt_volumes(T_pred, T_gt, T_liq),
                "melt_W_pred": extent_pred["W"],
                "melt_L_pred": extent_pred["L"],
                "melt_D_pred": extent_pred["D"],
                "melt_W_gt": extent_gt["W"],
                "melt_L_gt": extent_gt["L"],
                "melt_D_gt": extent_gt["D"],
            }
            metrics.append(step_metrics)

        return RolloutResult(
            mode=mode,
            stepper_name=stepper.name,
            gt_snapshots=list(gt_trajectory.snapshots[1 : 1 + len(pred_T)]),
            pred_T=pred_T,
            per_step_metrics=metrics,
            latencies_s=latencies,
            vram_peak_bytes=vram_peak_bytes(),
            diverged_at_step=diverged_at,
        )


def _snapshot_to_state(snap: Snapshot) -> SimulationState:
    return SimulationState(
        T=snap.T.clone(),
        t=snap.t,
        step=0,
        material_mask=snap.material_mask.clone()
        if snap.material_mask is not None
        else None,
    )
