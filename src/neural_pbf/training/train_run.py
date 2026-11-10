"""The training run every experiment script shares.

A script supplies only what differs between models (``ModelSpec``): how the model is built, how it is called, its data
loss, its schedule, and whether it runs under Accelerate. ``run_training`` does the rest identically for every script:
the manifest split check, the kernel pre-flight, the material per sample, ``TrainLoop`` for every step and epoch,
``RunTracker`` logging, and checkpoints with provenance. Only train and val are read; cal and test belong to the UQ
pipeline.

A script run on its own is a standalone run (``lpbf_experiments``, pinned=false). A layer above the scripts (the
ablation queue) passes ``RunHooks`` that validate the arguments and pin the run; nothing here knows that layer.

Pause and resume (``training/resume.py``): ``latest.pt`` is written after every epoch; a stop file in the checkpoint
directory, or SIGINT/SIGTERM, pauses the run after the current epoch with exit code 75, and the next start with the
same arguments continues from ``latest.pt`` into the same MLflow run. A finished run is not trained again.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, RandomSampler

from neural_pbf.data.material_dataset import MaterialDataset
from neural_pbf.data.patch_dataset import read_grid_attrs
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.losses import loss_panel
from neural_pbf.eval.viz.spatial import gallery_evolution
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.models.generative.fm.inputs import net_inputs
from neural_pbf.training import build_patch_data, log_val_image_rope, run_train_epoch, run_train_loop, run_val_epoch
from neural_pbf.training.checkpointing import atomic_torch_save, save_checkpoint
from neural_pbf.training.loops import TrainHistory
from neural_pbf.training.physics_term import PhysicsTerm, preflight
from neural_pbf.training.resume import (
    PauseRequest,
    epoch_seed,
    ignore_stop_signals,
    remove_stale_tmp,
    rng_state,
    seed_everything,
    set_rng_state,
    write_json_atomic,
)
from neural_pbf.training.script_args import resolve_data
from neural_pbf.training.tracking import RunHooks, RunTracker, split_runs
from neural_pbf.training.train_loop import TrainLoop
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.stages import EXIT_PAUSED

logger = logging.getLogger(__name__)
SCHEDULES = ("onecycle_pct0.1", "constant")


def _default_call(
    model: nn.Module, inputs: Tensor, tau: Tensor, cond_emb: Tensor, batch: Mapping[str, Any],
    grid_attrs: Mapping[str, float],
) -> Tensor:  # fmt: skip
    return model(inputs, tau, cond_emb)


def _default_loss(v_pred: Tensor, noise: Tensor, T_target: Tensor) -> Tensor:
    return fm_loss(v_pred, noise, T_target)


def tv_loss(v: Tensor) -> Tensor:
    """Mean squared difference of neighbouring voxels along each axis, averaged over the three axes."""
    d = (v[:, :, 1:] - v[:, :, :-1]).pow(2).mean()
    h = (v[:, :, :, 1:] - v[:, :, :, :-1]).pow(2).mean()
    w = (v[:, :, :, :, 1:] - v[:, :, :, :, :-1]).pow(2).mean()
    return (d + h + w) / 3.0


def with_tv(weight: float) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """The FM loss plus ``weight``·TV of the velocity field (train_fm_dit.py and train_fm_dit_rope.py)."""

    def loss(v_pred: Tensor, noise: Tensor, T_target: Tensor) -> Tensor:
        return fm_loss(v_pred, noise, T_target) + weight * tv_loss(v_pred)

    return loss


@dataclass
class ModelSpec:
    """What a script supplies. ``build(args, ds_cfg, grid_attrs) -> (model, cond_encoder, model_cfg)``."""

    build: Callable[[argparse.Namespace, Any, Mapping[str, float]], tuple[nn.Module, nn.Module, dict[str, Any]]]
    schedule: str
    use_accelerate: bool
    # call(model, inputs, tau, cond_emb, batch, grid_attrs) -> v_pred; batch holds patch_origin for RoPE models
    call: Callable[[nn.Module, Tensor, Tensor, Tensor, Mapping[str, Any], Mapping[str, float]], Tensor] = _default_call
    data_loss: Callable[[Tensor, Tensor, Tensor], Tensor] = _default_loss
    fixed: Mapping[str, Any] = field(default_factory=dict)  # further hard-coded recipe values (e.g. tv_weight)

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")

    def recipe_fixed(self) -> dict[str, Any]:
        fixed = {"schedule": self.schedule, **self.fixed}
        if not self.use_accelerate:
            fixed["mixed_precision"] = "no"  # no Accelerate, no autocast: fp32
        return fixed


def add_common_args(parser: argparse.ArgumentParser, use_accelerate: bool) -> None:
    """Recipe, output and infrastructure arguments every experiment script takes."""
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Default: checkpoints/<run name>.")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--snapshot_every", type=int, default=10, help="Log a val rollout figure every N epochs.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes; infrastructure, logged but never pinned.")  # fmt: skip
    parser.add_argument("--uq_root", type=str, default=None,
                        help="UQ root with the S1 dataset marker. Default: the UQ configuration's.")  # fmt: skip
    if use_accelerate:
        parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    else:
        parser.add_argument("--device", type=str, default="cuda")


def data_loaders(split: Any, batch_size: int, num_workers: int, seed: int) -> tuple[DataLoader, DataLoader]:
    """Train and val loaders with material, read by ``num_workers`` processes.

    ``build_patch_data`` reads in the main process (S4-hashed, not edited here). The worker count never changes the
    batches: the sampler runs in the main process, and the train order comes from the sampler's own generator
    (``train.sampler.generator``, re-seeded per epoch by ``run_training``). Each loader has a further generator for its
    worker seeds, so creating an iterator never draws from the global RNG or the shuffle: with persistent workers a
    resumed run creates its iterators at another epoch than an uninterrupted one. Each worker opens the HDF5 files
    per item, so no handle is shared.
    """
    t_boil = UQConfig().exploratory.T_boil_K
    kw: dict[str, Any] = {"batch_size": batch_size, "collate_fn": split.train_loader.collate_fn,
                          "num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
                          "persistent_workers": num_workers > 0,
                          "worker_init_fn": partial(ignore_stop_signals, parent=os.getpid())}  # fmt: skip
    train_ds, val_ds = MaterialDataset(split.train_ds, t_boil=t_boil), MaterialDataset(split.val_ds, t_boil=t_boil)
    shuffle = RandomSampler(train_ds, generator=torch.Generator().manual_seed(seed))
    train = DataLoader(train_ds, sampler=shuffle, generator=torch.Generator().manual_seed(seed + 1), **kw)
    val = DataLoader(val_ds, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **kw)
    return train, val


def run_training(args: argparse.Namespace, script: str | Path, spec: ModelSpec, hooks: RunHooks | None = None) -> None:
    """Train one model end to end; ``hooks`` (default: a standalone run) come from a layer above the scripts."""
    hooks = hooks or RunHooks()
    if hooks.validate is not None:
        try:
            hooks.validate(args, spec.recipe_fixed())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    name = hooks.run_name(script)
    h5_paths, manifest_path = resolve_data(args)
    if manifest_path is None:
        raise SystemExit("a training run uses the manifest's train and val splits; no --allow-random-split")

    logging.basicConfig(level=logging.INFO)
    accelerator: Any = None
    if spec.use_accelerate:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision=args.mixed_precision)
        device = accelerator.device
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_main = accelerator is None or accelerator.is_main_process
    seed_everything(args.seed)
    grid_attrs = cast(dict[str, float], dict(read_grid_attrs(h5_paths[0])))  # numeric grid attributes

    split = build_patch_data(
        h5_paths, manifest_path=manifest_path, allow_random_split=False, patch_size=64,
        batch_size=args.batch_size, seed=args.seed, use_origins=True,
    )  # fmt: skip
    ds_cfg = split.ds_cfg
    runs = split_runs(manifest_path, split.train_ds, split.val_ds)
    physics = PhysicsTerm.from_args(args, ds_cfg, grid_attrs)
    checked = preflight(args.physics_backend, device)
    train_loader, val_loader = data_loaders(split, args.batch_size, args.num_workers, args.seed)
    shuffle = cast(RandomSampler, train_loader.sampler).generator  # the same object after accelerator.prepare
    assert shuffle is not None

    model, cond_encoder, model_cfg = spec.build(args, ds_cfg, grid_attrs)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr)
    scheduler: Any = None
    if spec.schedule == "onecycle_pct0.1":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.1
        )
    if accelerator is not None:
        prepared = accelerator.prepare(model, cond_encoder, optimizer, train_loader, val_loader,
                                       *([scheduler] if scheduler is not None else []))  # fmt: skip
        model, cond_encoder, optimizer, train_loader, val_loader = prepared[:5]
        scheduler = prepared[5] if scheduler is not None else None
    else:
        model, cond_encoder = model.to(device), cond_encoder.to(device)

    def backward(loss: Tensor) -> None:
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

    def reduce_mean(t: Tensor) -> Tensor:
        return accelerator.reduce(t, "mean") if accelerator is not None else t

    def reduce_sum(t: Tensor) -> Tensor:
        return accelerator.reduce(t.to(device), "sum").cpu() if accelerator is not None else t

    ckpt_dir = Path(args.checkpoint_dir or f"checkpoints/{name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    if is_main:
        for name_tmp in remove_stale_tmp(ckpt_dir):
            logger.warning("removed %s, left by a run killed during a save", name_tmp)
    # weights_only=False: our own file, holding the numpy and python RNG states and the history
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if saved is not None and saved["state"] == "finished":
        raise SystemExit(f"{name} already finished ({latest}); remove {ckpt_dir} to retrain it")
    configs = {
        "fm_cfg": model_cfg, "dataset_cfg": ds_cfg.model_dump(), "in_channels": args.in_channels,
        "conditioning_keys": list(ds_cfg.conditioning_keys), "grid": dict(grid_attrs),
        "data": {"n_train": len(split.train_ds), "n_val": len(split.val_ds)},
        "runtime": {"accelerate": spec.use_accelerate, "device": str(device),
                    "num_processes": accelerator.num_processes if accelerator is not None else 1},
    }  # fmt: skip
    history = TrainHistory(**saved["history"]) if saved is not None else TrainHistory()
    tracker: RunTracker | None = None
    if is_main and saved is not None:
        tracker = RunTracker.resume(args, hooks, provenance=saved["provenance"], epoch=history.next_epoch,
                                    script=Path(script).name, uq_root=args.uq_root,
                                    configs=configs, preflight=checked)  # fmt: skip
    elif is_main:
        tracker = RunTracker.start(args, hooks, script=Path(script).name, split=runs, configs=configs,
                                   uq_root=args.uq_root,
                                   preflight=checked)  # fmt: skip
    loop = TrainLoop(
        physics=physics, tracker=tracker, optimizer=optimizer, scheduler=scheduler,
        parameters=list(model.parameters()) + list(cond_encoder.parameters()), backward=backward,
        val_residual_cfg=(ds_cfg, grid_attrs), log_every=args.log_every, reduce_mean=reduce_mean,
        reduce_sum=reduce_sum,
    )  # fmt: skip

    def _fields(batch: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {k: batch[k].to(device).squeeze(1) for k in ("T_in", "T_target", "Q", "mask")}
        out.update({k: batch[k].to(device) for k in ("material", "T_boil", "conditioning")})
        if "patch_origin" in batch:
            out["patch_origin"] = batch["patch_origin"].to(device)
        return out

    def _inputs(x: Tensor, f: Mapping[str, Tensor]) -> Tensor:
        return net_inputs(x, {"mask": f["mask"], "Q": f["Q"], "T_in": f["T_in"]}, args.in_channels)

    def _forward(batch: Mapping[str, Any], train: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        """Training uses the script's data loss; validation the plain FM loss, the same quantity for every script."""
        f = _fields(batch)
        cond_emb = cond_encoder(f["conditioning"])
        noise = sample_noise(f["T_target"])
        tau = torch.rand(noise.shape[0], device=device)
        x_tau = interpolate(noise, f["T_target"], tau)
        v_pred = spec.call(model, _inputs(x_tau, f), tau, cond_emb, f, grid_attrs)
        loss = (spec.data_loss if train else _default_loss)(v_pred, noise, f["T_target"])
        return loss, v_pred, x_tau, tau, f

    def _val_batch(batch: Mapping[str, Any]) -> float:
        with torch.no_grad():
            return loop.val_step(*_forward(batch, train=False))

    def _train_epoch(epoch: int) -> float:
        model.train()
        cond_encoder.train()
        shuffle.manual_seed(epoch_seed(args.seed, epoch))
        loop.start_epoch()
        return run_train_epoch(lambda b: loop.train_step(*_forward(b, train=True)), train_loader, epoch,
                               tqdm_disable=not is_main)  # fmt: skip

    def _val_epoch(epoch: int) -> float:
        model.eval()
        cond_encoder.eval()
        val_fm = run_val_epoch(_val_batch, val_loader)
        loop.end_epoch(epoch, n_samples=len(split.train_ds))
        return val_fm  # checkpoint selection on the val FM loss, as before

    def _checkpoint(epoch: int, val_loss: float) -> None:
        save_checkpoint(model, cond_encoder, ckpt_dir, epoch, val_loss, args, accelerator=accelerator,
                        dataset_cfg=ds_cfg.model_dump(), fm_cfg=model_cfg,
                        provenance=tracker.provenance if tracker else None)  # fmt: skip
        mlflow.log_metric("best_val_loss", val_loss, step=epoch)  # run_train_loop calls this on main only

    def _rollout(val_batch: Mapping[str, Any]) -> Tensor:
        f = _fields({k: v[:1] for k, v in val_batch.items()})
        cond_emb = cond_encoder(f["conditioning"])
        x = sample_noise(f["T_in"])
        with torch.no_grad():
            for i in range(25):
                tau = torch.full((1,), i / 25, device=device)
                x = x + spec.call(model, _inputs(x, f), tau, cond_emb, f, grid_attrs) / 25
        return x

    def _snapshot(epoch: int) -> tuple[Tensor, Tensor] | None:
        if epoch % args.snapshot_every != 0:
            return None
        return log_val_image_rope(model, cond_encoder, _rollout, next(iter(val_loader)), epoch, mlflow,
                                  filename_prefix="val_epoch")  # fmt: skip

    def _unwrap(m: nn.Module) -> nn.Module:
        return accelerator.unwrap_model(m) if accelerator is not None else m

    scaler = getattr(accelerator, "scaler", None)

    def _save_latest(h: TrainHistory, state: str) -> None:
        """Everything an uninterrupted run still holds after this epoch; main process only."""
        if not is_main:
            return
        atomic_torch_save({
            "model_state": _unwrap(model).state_dict(), "cond_encoder_state": _unwrap(cond_encoder).state_dict(),
            "ds_cfg": ds_cfg.model_dump(), "fm_cfg": model_cfg, "grid_attrs": dict(grid_attrs),
            "provenance": tracker.provenance if tracker is not None else None, "state": state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "weighter_state": dict(vars(physics.weighter)), "rng": rng_state(), "global_step": loop.global_step,
            "history": asdict(h),
        }, latest)  # fmt: skip

    if saved is not None:
        _unwrap(model).load_state_dict(saved["model_state"])
        _unwrap(cond_encoder).load_state_dict(saved["cond_encoder_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler_state"])
        if scaler is not None and saved["scaler_state"] is not None:
            scaler.load_state_dict(saved["scaler_state"])
        physics.weighter.__dict__.update(saved["weighter_state"])
        loop.global_step = int(saved["global_step"])
        set_rng_state(saved["rng"])  # last: nothing may draw between here and the first resumed epoch
        logger.info("resuming %s at epoch %d (step %d)", name, history.next_epoch, loop.global_step)

    pause = PauseRequest(ckpt_dir)
    epoch_times: list[float] = []
    clock = {"t": time.monotonic()}

    def _progress(state: str, next_epoch: int) -> None:
        """progress.json for the queue's --status and stall watchdog; main process only."""
        if not is_main:
            return
        per_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        remaining = 0 if state == "finished" else args.epochs - next_epoch
        write_json_atomic(ckpt_dir / "progress.json", {
            "name": name, "state": state, "next_epoch": next_epoch, "epochs": args.epochs,
            "global_step": loop.global_step, "sec_per_epoch": per_epoch, "eta_s": per_epoch * remaining,
            "updated": time.time(),
        })  # fmt: skip

    def _epoch_end(_epoch: int, h: TrainHistory) -> None:
        now = time.monotonic()
        epoch_times.append(now - clock["t"])
        clock["t"] = now
        _save_latest(h, "running")
        _progress("running", h.next_epoch)

    loop.on_log = lambda _step: _progress("running", history.next_epoch)

    def _should_pause() -> bool:
        flag = torch.tensor(float(pause.requested()), device=device)
        return bool((accelerator.reduce(flag, "max") if accelerator is not None else flag).item() > 0)

    # The handlers stay installed until the end: a second signal after the loop must still kill, not raise.
    with pause.watch_signals():
        with tracker if tracker is not None else contextlib.nullcontext():
            logger.info("training %s from epoch %d of %d", name, history.next_epoch, args.epochs)
            history = run_train_loop(
                _train_epoch, _val_epoch, _checkpoint, _snapshot, epochs=args.epochs, val_every=args.val_every,
                sync_fn=accelerator.wait_for_everyone if accelerator is not None else None, is_main=is_main,
                tqdm_disable=not is_main, history=history, epoch_end_fn=_epoch_end,
                should_pause=_should_pause,
            )  # fmt: skip
            if history.paused:
                _save_latest(history, "paused")
                _progress("paused", history.next_epoch)
                if tracker is not None:
                    pause.clear()
                    tracker.pause()
                    logger.info("paused %s after epoch %d; start it again to continue", name,
                                history.next_epoch - 1)  # fmt: skip
            elif tracker is not None:
                if history.train_losses and history.val_losses:
                    log_figure(mlflow, loss_panel(history.train_losses, history.val_losses,
                                                  title=f"{name} — Loss Panel"), "loss_panel.png")  # fmt: skip
                if history.T_gt_fixed is not None and history.pred_history:
                    log_figure(mlflow, gallery_evolution(
                        history.T_gt_fixed, history.pred_history, epoch_labels=history.epoch_labels,
                        title=f"{name} — Training Evolution", dpi=150,
                    ), "gallery_evolution.png")  # fmt: skip
                _save_latest(history, "finished")
                _progress("finished", history.next_epoch)
                pause.clear()
                tracker.finish(best=ckpt_dir / "best.pt", final=latest)
        if accelerator is not None:
            accelerator.wait_for_everyone()
    if history.paused:
        raise SystemExit(EXIT_PAUSED)
