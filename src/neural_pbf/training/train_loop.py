"""The step and epoch logic every experiment script shares.

A script builds its model, runs its forward pass and computes its data loss (the FM loss, plus TV regularization for
DiT v1/v2); ``TrainLoop`` does the rest the same way for every script: the physics term and its capped ReLoBRaLo weight,
backward and the optimizer step, per-step metrics every ``log_every`` steps, the per-term epoch averages, the val
physics residual in both forms, and epoch time, peak GPU memory and throughput.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from neural_pbf.training.physics_term import PhysicsTerm, Reduce, ValResidual, _identity


@dataclass
class TrainLoop:
    physics: PhysicsTerm
    tracker: Any | None  # tracking.RunTracker on the main process, None elsewhere
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    parameters: Sequence[Tensor]
    backward: Callable[[Tensor], None]  # loss.backward or accelerator.backward
    val_residual_cfg: tuple[Any, Mapping[str, float]]  # (dataset config, grid attrs)
    log_every: int = 50
    reduce_mean: Reduce = _identity  # mean of a detached scalar across processes
    reduce_sum: Reduce = _identity  # sum of the val residual accumulators across processes
    global_step: int = 0
    on_log: Callable[[int], None] | None = None  # called every log_every steps with the step, e.g. for progress.json
    _sums: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        self.val_residual = ValResidual(*self.val_residual_cfg)
        self.start_epoch()

    def start_epoch(self) -> None:
        self._sums = dict.fromkeys(
            ("train_fm", "train_phys", "train_total", "train_n", "val_fm", "val_phys", "val_n"), 0.0
        )
        self.val_residual = ValResidual(*self.val_residual_cfg)
        self._t0 = time.perf_counter()
        self._t_train: float | None = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def train_step(
        self, loss_fm: Tensor, v_pred: Tensor, x_tau: Tensor, tau: Tensor, batch: Mapping[str, Tensor]
    ) -> float:
        """One optimizer step on ``loss_fm + λ_phys·loss_phys``; returns the (reduced) total loss."""
        loss_phys = self.physics.loss(v_pred, x_tau, tau, batch)
        total, metrics = self.physics.combine(loss_fm, loss_phys, self.reduce_mean)
        self.optimizer.zero_grad()
        self.backward(total)
        log = self.global_step % self.log_every == 0
        if log:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(self.parameters), math.inf))
            lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        s = self._sums
        s["train_fm"] += metrics["loss_fm"]
        s["train_phys"] += metrics["loss_phys"]
        s["train_total"] += metrics["loss_total"]
        s["train_n"] += 1
        if log and self.tracker is not None:
            self.tracker.step({**metrics, "lr": lr, "grad_norm": grad_norm}, self.global_step)
        if log and self.on_log is not None:
            self.on_log(self.global_step)
        self.global_step += 1
        return metrics["loss_total"]

    @torch.no_grad()
    def val_step(
        self, loss_fm: Tensor, v_pred: Tensor, x_tau: Tensor, tau: Tensor, batch: Mapping[str, Tensor]
    ) -> float:
        """Val losses of one batch and its contribution to the val physics residual; returns the val FM loss."""
        if self._t_train is None:
            self._t_train = time.perf_counter() - self._t0
        fm = float(self.reduce_mean(loss_fm.detach()))
        phys = float(self.reduce_mean(self.physics.loss(v_pred, x_tau, tau, batch).detach()))
        s = self._sums
        s["val_fm"] += fm
        s["val_phys"] += phys
        s["val_n"] += 1
        self.val_residual.update(v_pred, x_tau, tau, batch)
        return fm

    def end_epoch(self, epoch: int, n_samples: int) -> dict[str, float]:
        """Epoch metrics (``tracking.REQUIRED_EPOCH``); logged when a tracker is set."""
        s = self._sums
        elapsed = time.perf_counter() - self._t0
        train_time = self._t_train if self._t_train is not None else elapsed
        n_train, n_val = max(s["train_n"], 1.0), max(s["val_n"], 1.0)
        lam = self.physics.weighter.lambda_phys if self.physics.active else 0.0
        val_fm, val_phys = s["val_fm"] / n_val, s["val_phys"] / n_val
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        metrics = {
            "train_loss_fm": s["train_fm"] / n_train, "train_loss_phys": s["train_phys"] / n_train,
            "train_loss_total": s["train_total"] / n_train, "val_loss_fm": val_fm, "val_loss_phys": val_phys,
            "val_loss_total": val_fm + lam * val_phys, **self.val_residual.compute(self.reduce_sum),
            "epoch_time_s": elapsed, "peak_gpu_mem_mb": peak,
            "throughput_samples_s": n_samples / train_time if train_time > 0 else 0.0,
        }  # fmt: skip
        if self.tracker is not None:
            self.tracker.epoch(metrics, epoch)
        return metrics
