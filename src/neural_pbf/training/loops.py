"""Generic training loops, test phase, and training-time visualization helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_pbf.eval.metrics.geometry import evaluate_physical_metrics, iou_melt_volumes
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.spatial import val_grid_2x2
from neural_pbf.physics.constants import T_LIQUIDUS_NORM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """Accumulated per-epoch scalars and snapshots for post-training visualisation."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    pred_history: list[Tensor] = field(default_factory=list)
    epoch_labels: list[str] = field(default_factory=list)
    T_gt_fixed: Tensor | None = None


# ---------------------------------------------------------------------------
# Core loop primitives
# ---------------------------------------------------------------------------


def run_train_epoch(
    batch_fn: Callable[[dict], float],
    loader: DataLoader,
    epoch: int,
    *,
    profiler: Any = None,
    tqdm_desc: str | None = None,
    tqdm_disable: bool = False,
) -> float:
    """Iterate one training epoch calling *batch_fn* on every batch.

    Args:
        batch_fn:     Closure that receives a batch dict, runs forward + backward +
                      optimizer step, and returns the scalar loss as a Python float.
        loader:       Training DataLoader.
        epoch:        Current epoch index (used in default tqdm description).
        profiler:     Optional torch.profiler handle; profiler.step() is called after
                      each batch when provided.
        tqdm_desc:    Override for the tqdm progress-bar label.
        tqdm_disable: Pass True to suppress the per-batch tqdm bar (e.g. on
                      non-main Accelerate processes).

    Returns:
        Average loss over the epoch.
    """
    desc = tqdm_desc if tqdm_desc is not None else f"Train {epoch}"
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc=desc, leave=False, disable=tqdm_disable):
        total += batch_fn(batch)
        n += 1
        if profiler is not None:
            profiler.step()
    return total / max(n, 1)


def run_val_epoch(
    step_fn: Callable[[dict], float],
    loader: DataLoader,
) -> float:
    """Iterate one validation epoch under torch.no_grad().

    Args:
        step_fn: Closure that receives a batch dict and returns the scalar
                 validation loss.  The model should already be in eval mode when
                 this is called (set by the caller before run_val_epoch).
        loader:  Validation DataLoader.

    Returns:
        Average validation loss over the epoch.
    """
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            total += step_fn(batch)
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def run_train_loop(
    train_epoch_fn: Callable[[int], float],
    val_epoch_fn: Callable[[int], float],
    checkpoint_fn: Callable[[int, float], None],
    snapshot_fn: Callable[[int], tuple[Tensor, Tensor] | None],
    epochs: int,
    val_every: int = 1,
    *,
    sync_fn: Callable[[], None] | None = None,
    is_main: bool = True,
    tqdm_disable: bool = False,
) -> TrainHistory:
    """Orchestrate the full training loop with val, checkpointing, and snapshots.

    All side-effects (MLflow logging, checkpoint writes, figure generation) are
    handled by the callbacks — this function only manages control flow and
    accumulates :class:`TrainHistory`.

    Args:
        train_epoch_fn:  ``(epoch) -> avg_train_loss``.  Expected to call
                         model.train(), run run_train_epoch, and log train metrics.
        val_epoch_fn:    ``(epoch) -> avg_val_loss``.  Expected to call model.eval()
                         and log the val metric.
        checkpoint_fn:   ``(epoch, val_loss) -> None``.  Called whenever a new best
                         val loss is reached.
        snapshot_fn:     ``(epoch) -> (T_gt, T_pred) | None``.  Called every epoch;
                         return a tensor pair to accumulate in history, or None to skip.
        epochs:          Total number of epochs.
        val_every:       Run validation every N epochs.
        sync_fn:         Optional barrier to call after every val epoch, e.g.
                         ``accelerator.wait_for_everyone``.
        is_main:         When False (non-main Accelerate process) history accumulation
                         and checkpoint/snapshot callbacks are skipped.
        tqdm_disable:    Suppress the outer epoch tqdm bar.

    Returns:
        :class:`TrainHistory` populated with losses and snapshot tensors.
    """
    history = TrainHistory()
    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs", disable=tqdm_disable):
        avg_train = train_epoch_fn(epoch)

        if epoch % val_every != 0:
            if sync_fn is not None:
                sync_fn()
            continue

        avg_val = val_epoch_fn(epoch)

        if is_main:
            history.train_losses.append(avg_train)
            history.val_losses.append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                checkpoint_fn(epoch, avg_val)

            snap = snapshot_fn(epoch)
            if snap is not None:
                T_gt, T_pred = snap
                if history.T_gt_fixed is None:
                    history.T_gt_fixed = T_gt
                history.pred_history.append(T_pred.detach().cpu())
                history.epoch_labels.append(f"Ep {epoch}")

        if sync_fn is not None:
            sync_fn()

    return history


# ---------------------------------------------------------------------------
# Test phase
# ---------------------------------------------------------------------------


def run_test_phase(
    rollout_fn: Callable[[dict], tuple[Tensor, Tensor]],
    test_loader: DataLoader,
    ckpt_dir: Path,
    run: Any,
    seed: int,
    *,
    tracker: Any = None,
) -> list[tuple[Tensor, Tensor]]:
    """Run the standardised test evaluation loop.

    The caller is responsible for loading the best checkpoint into the model and
    setting model.eval() before calling this function.

    Args:
        rollout_fn:   ``(batch) -> (T_pred_cpu, T_tgt_cpu)``.  Closure that handles
                      device placement and inference; returns both predictions and
                      ground-truth as CPU tensors shaped ``(B, 1, D, H, W)``.
        test_loader:  Test DataLoader (expected batch_size=1).
        ckpt_dir:     Directory for test artefacts (mapping file, metrics JSON).
        run:          Active MLflow run object used to log figure artefacts.
        seed:         RNG seed for reproducibility across evaluation runs.
        tracker:      When provided, log the mapping file and metrics JSON as
                      MLflow artefacts via tracker.log_artifact.

    Returns:
        Up to 5 ``(T_gt, T_pred)`` CPU tensor pairs for gallery visualisation.
    """
    torch.manual_seed(seed)
    mse_list: list[float] = []
    iou_list: list[float] = []
    phys_metrics: dict[str, list[float]] = {}
    gallery_samples: list[tuple[Tensor, Tensor]] = []

    mapping_path = ckpt_dir / "test_sample_mapping.txt"
    with mapping_path.open("w") as f_map:
        f_map.write("local_idx,h5_path,sample_key\n")
        for local_idx, batch in enumerate(tqdm(test_loader, desc="Test")):
            T_pred, T_tgt = rollout_fn(batch)

            if "h5_path" in batch and "sample_key" in batch:
                h5_raw = batch["h5_path"]
                key_raw = batch["sample_key"]
                h5_str = h5_raw[0] if isinstance(h5_raw, list) else str(h5_raw)
                key_str = key_raw[0] if isinstance(key_raw, list) else str(key_raw)
                f_map.write(f"{local_idx},{h5_str},{key_str}\n")

            mse_list.append(F.mse_loss(T_pred, T_tgt).item())
            iou_list.append(float(iou_melt_volumes(T_pred, T_tgt, T_LIQUIDUS_NORM)))
            for k, val in evaluate_physical_metrics(T_pred, T_tgt, T_LIQUIDUS_NORM).items():
                phys_metrics.setdefault(k, []).append(val)
            if local_idx < 4:
                log_figure(
                    run,
                    val_grid_2x2(T_tgt[:1], T_pred[:1], epoch=1000 + local_idx),
                    f"test_sample_{local_idx:03d}.png",
                    dpi=120,
                )
            if len(gallery_samples) < 5:
                gallery_samples.append((T_tgt[:1], T_pred[:1]))

    if tracker is not None:
        tracker.log_artifact(str(mapping_path))

    if mse_list:
        avg_mse = sum(mse_list) / len(mse_list)
        avg_iou = sum(iou_list) / len(iou_list)
        mlflow.log_metric("test_mse_rollout", avg_mse)
        mlflow.log_metric("test_iou_rollout", avg_iou)
        if phys_metrics:
            mlflow.log_metrics({k: sum(v) / len(v) for k, v in phys_metrics.items()})
        metrics_detail: dict = {
            "samples": [
                {
                    "idx": i,
                    "mse": mse_list[i],
                    "iou": iou_list[i],
                    **{k: phys_metrics[k][i] for k in phys_metrics},
                }
                for i in range(len(mse_list))
            ],
            "summary": {
                "n_samples": len(mse_list),
                "mse_mean": avg_mse,
                "iou_mean": avg_iou,
                **{k: sum(v) / len(v) for k, v in phys_metrics.items()},
            },
        }
        json_path = str(ckpt_dir / "test_metrics_detailed.json")
        with open(json_path, "w") as _jf:
            json.dump(metrics_detail, _jf, indent=2)
        run.log_artifact(json_path, artifact_path="eval")

    return gallery_samples


# ---------------------------------------------------------------------------
# Training-time visualisation helper
# ---------------------------------------------------------------------------


def log_val_image_rope(
    model: Any,
    cond_encoder: Any,
    rollout_fn: Callable[[dict], Tensor],
    val_batch: dict,
    epoch: int,
    run: Any,
    *,
    filename_prefix: str = "val_rope_epoch",
) -> tuple[Tensor, Tensor]:
    """Run one-sample inference and log a 2×2 validation figure to MLflow.

    Sets model/cond_encoder to eval before inference and restores train mode
    in a finally block.

    Args:
        model:            The velocity model.
        cond_encoder:     Conditioning encoder.
        rollout_fn:       ``(batch) -> T_pred`` where T_pred is on whatever device
                          the closure uses.  Should *not* call no_grad itself —
                          that is handled here.
        val_batch:        A single validation batch dict.
        epoch:            Current epoch number (used for figure filename
                          and axis label).
        run:              Active MLflow run for log_figure.
        filename_prefix:  Prefix for the logged artefact filename.

    Returns:
        ``(T_gt_cpu, T_pred_cpu)`` as squeezed ``(1, 1, D, H, W)`` CPU tensors.
    """
    model.eval()
    cond_encoder.eval()
    try:
        with torch.no_grad():
            T_pred = rollout_fn(val_batch)
        T_gt = val_batch["T_target"].squeeze(1)[:1].cpu()
        log_figure(
            run,
            val_grid_2x2(T_gt, T_pred[:1].cpu(), epoch=epoch),
            f"{filename_prefix}_{epoch:03d}.png",
            dpi=120,
        )
        return T_gt, T_pred[:1].cpu()
    finally:
        model.train()
        cond_encoder.train()
