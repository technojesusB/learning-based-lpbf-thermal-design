"""TrainLoop: the step and epoch logic every experiment script shares (physics, weighting, metrics)."""

from __future__ import annotations

import math

import pytest
import torch

from neural_pbf.training.train_loop import TrainLoop
from neural_pbf.training.tracking import REQUIRED_EPOCH, REQUIRED_STEP
from tests.training.test_physics_term import DS, GRID, _args, _batch, _model_outputs


class _Tracker:
    def __init__(self) -> None:
        self.steps: list[tuple[int, dict]] = []
        self.epochs: list[tuple[int, dict]] = []

    def step(self, metrics: dict, step: int) -> None:
        assert set(REQUIRED_STEP) <= set(metrics)
        self.steps.append((step, dict(metrics)))

    def epoch(self, metrics: dict, epoch: int) -> None:
        assert set(REQUIRED_EPOCH) <= set(metrics)
        self.epochs.append((epoch, dict(metrics)))


def _loop(
    form: str = "tin", backend: str | None = "reference", log_every: int = 2
) -> tuple[TrainLoop, _Tracker, torch.nn.Parameter]:
    from neural_pbf.training.physics_term import PhysicsTerm

    w = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
    optimizer = torch.optim.SGD([w], lr=0.1)
    tracker = _Tracker()
    loop = TrainLoop(
        physics=PhysicsTerm.from_args(_args(form, backend), DS, GRID), tracker=tracker, optimizer=optimizer,
        scheduler=None, parameters=[w], backward=lambda loss: loss.backward(), val_residual_cfg=(DS, GRID),
        log_every=log_every,
    )  # fmt: skip
    return loop, tracker, w


def _step(loop: TrainLoop, w: torch.nn.Parameter) -> float:
    b = _batch()
    v, x_tau, tau = _model_outputs(b)
    v = v * w  # the "model": the gradient reaches w through both terms
    return loop.train_step((v - x_tau).pow(2).mean(), v, x_tau, tau, b)


@pytest.mark.unit
def test_train_step_updates_the_model_and_logs_every_n_steps() -> None:
    loop, tracker, w = _loop(log_every=2)
    before = float(w)
    loop.start_epoch()
    for _ in range(5):
        _step(loop, w)
    assert float(w) != before
    assert [s for s, _ in tracker.steps] == [0, 2, 4]
    m = tracker.steps[-1][1]
    assert m["loss_phys"] > 0 and m["grad_norm"] > 0 and m["lr"] == 0.1 and m["lambda_phys"] <= 1.0


@pytest.mark.unit
def test_epoch_metrics_average_each_term_and_add_the_val_residual() -> None:
    loop, tracker, w = _loop()
    loop.start_epoch()
    totals = [_step(loop, w) for _ in range(3)]
    b = _batch()
    v, x_tau, tau = _model_outputs(b)
    val_fm = loop.val_step((v - x_tau).pow(2).mean(), v, x_tau, tau, b)
    out = loop.end_epoch(epoch=7, n_samples=6)
    assert tracker.epochs[0][0] == 7
    assert out["train_loss_total"] == pytest.approx(sum(totals) / 3)
    assert out["val_loss_fm"] == pytest.approx(val_fm)
    assert math.isfinite(out["val_phys_tin"]) and math.isfinite(out["val_phys_coupled"])
    assert out["throughput_samples_s"] > 0 and out["epoch_time_s"] > 0


@pytest.mark.unit
def test_form_none_trains_on_the_data_loss_and_still_logs_the_val_residual() -> None:
    loop, tracker, w = _loop("none", None)
    loop.start_epoch()
    total = _step(loop, w)
    assert tracker.steps[0][1]["loss_phys"] == 0.0 and total == tracker.steps[0][1]["loss_fm"]
    b = _batch()
    loop.val_step(torch.tensor(0.1), *_model_outputs(b), b)
    out = loop.end_epoch(epoch=0, n_samples=2)
    assert out["val_loss_phys"] == 0.0 and math.isfinite(out["val_phys_tin"])


@pytest.mark.unit
def test_end_epoch_resets_the_sums() -> None:
    loop, tracker, w = _loop()
    loop.start_epoch()
    _step(loop, w)
    b = _batch()
    loop.val_step(torch.tensor(0.1), *_model_outputs(b), b)
    loop.end_epoch(epoch=0, n_samples=2)
    loop.start_epoch()
    _step(loop, w)
    loop.val_step(torch.tensor(0.3), *_model_outputs(b), b)
    out = loop.end_epoch(epoch=1, n_samples=2)
    assert out["val_loss_fm"] == pytest.approx(0.3)


@pytest.mark.unit
def test_without_a_tracker_nothing_is_logged_but_training_runs() -> None:
    """Non-main Accelerate processes train without a tracker."""
    loop, _, w = _loop()
    loop.tracker = None
    loop.start_epoch()
    _step(loop, w)
    b = _batch()
    loop.val_step(torch.tensor(0.1), *_model_outputs(b), b)
    assert set(REQUIRED_EPOCH) <= set(loop.end_epoch(epoch=0, n_samples=2))
