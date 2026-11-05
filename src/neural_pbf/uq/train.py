"""Baseline training on a run-level manifest: same model and optimiser as train_fm_patches.py.

Deviations from the legacy script, all documented in docs/uq.md: mask_in instead of the
post-step mask, manifest split, early stopping, seeded validation noise so epochs compare fairly.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import torch

from neural_pbf.data.patch_dataset import read_grid_attrs
from neural_pbf.data.splits import load_manifest
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.models.generative.fm.inputs import model_in_channels, net_inputs
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.training.factories import build_patch_data, setup_run
from neural_pbf.uq.config import UQConfig

logger = logging.getLogger(__name__)

_VAL_NOISE_SEED = 123456  # fixed: the same noise and tau for every validation pass


class TrainingError(RuntimeError):
    pass


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _loss(batch: dict[str, Any], model: Any, encoder: Any, device: torch.device, gen: torch.Generator | None) -> Any:
    target = batch["T_target"].to(device).squeeze(1)
    mask = batch["mask"].to(device).squeeze(1)
    q = batch["Q"].to(device).squeeze(1)
    cond = encoder(batch["conditioning"].to(device))
    noise = sample_noise(target, gen)
    tau = torch.rand(target.shape[0], device=device, generator=gen)
    t_in = batch["T_in"].to(device).squeeze(1)
    x = net_inputs(interpolate(noise, target, tau), {"mask": mask, "Q": q, "T_in": t_in}, model_in_channels(model))
    v = model(x, tau, cond)
    return fm_loss(v, noise, target)


def _validate(loader: Any, model: Any, encoder: Any, device: torch.device) -> float:
    gen = torch.Generator(device=device).manual_seed(_VAL_NOISE_SEED)
    model.eval()
    encoder.eval()
    with torch.no_grad():
        losses = [_loss(b, model, encoder, device, gen).item() for b in loader]
    if not losses:
        raise TrainingError("validation set is empty")
    return sum(losses) / len(losses)


def _load_resume(path: Path, manifest_hash: str, device: torch.device) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    if state["manifest_hash"] != manifest_hash:
        raise TrainingError(f"{path} was trained under another manifest; refuse to resume")
    return state


@dataclass
class _History:
    """Everything that decides early stopping; mutated once per epoch and saved in latest.pt."""

    val: list[float] = field(default_factory=list)
    train: list[float] = field(default_factory=list)
    best: float = math.inf
    since_best: int = 0
    stopped: bool = False


@dataclass
class _Session:
    cfg: UQConfig
    out: Path
    device: torch.device
    manifest: dict[str, Any]
    manifest_path: str
    h5_paths: list[str]
    git_commit: str
    split: Any
    fm_cfg: FMConfig
    model: Any
    encoder: Any
    optimizer: Any


def _build_session(cfg: UQConfig, manifest_path: str | Path, out: Path, git_commit: str) -> _Session:
    t = cfg.train
    manifest = load_manifest(manifest_path)
    h5_paths = sorted({p for runs in manifest["sets"].values() for r in runs for p, _ in r["sample_keys"]})
    device = torch.device(cfg.run.device)
    torch.manual_seed(t.seed)
    split = build_patch_data(
        h5_paths, patch_size=t.patch_size, batch_size=t.batch_size, seed=t.seed,
        manifest_path=str(manifest_path), mask_source=t.mask_source, conditioning_keys=t.conditioning_keys,
    )  # fmt: skip
    if not len(split.train_ds) or not len(split.val_ds):
        raise TrainingError("train or val set is empty")
    fm_cfg = FMConfig(
        in_channels=t.in_channels, base_channels=t.base_channels, depth=t.depth,
        cond_dim=len(split.ds_cfg.conditioning_keys), cond_embed_dim=t.cond_embed_dim,
        tau_embed_dim=t.cond_embed_dim,
    )  # fmt: skip
    model = VelocityNet(fm_cfg).to(device)
    encoder = ConditioningEncoder(fm_cfg.cond_dim, t.cond_embed_dim).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=t.lr)
    return _Session(
        cfg, out, device, manifest, str(manifest_path), h5_paths, git_commit, split, fm_cfg, model, encoder, optimizer
    )


def _restore(s: _Session) -> _History:
    """Load latest.pt when it exists; model, optimizer, and RNG continue exactly where they stopped."""
    state = _load_resume(s.out / "latest.pt", s.manifest["hash"], s.device)
    if state is None:
        return _History()
    s.model.load_state_dict(state["model_state"])
    s.encoder.load_state_dict(state["cond_encoder_state"])
    s.optimizer.load_state_dict(state["optimizer_state"])
    torch.set_rng_state(state["rng_cpu"].cpu())
    if s.device.type == "cuda" and state["rng_cuda"] is not None:
        torch.cuda.set_rng_state(state["rng_cuda"].cpu(), s.device)
    history = _History(
        list(state["val_history"]), list(state["train_history"]), state["best_val"], state["since_best"],
        state["stopped_early"],
    )  # fmt: skip
    logger.info("Resuming at epoch %d (best val %.6f)", len(history.val), history.best)
    return history


def _checkpoint(s: _Session, epoch: int, val_loss: float) -> dict[str, Any]:
    return {
        "model_state": s.model.state_dict(),
        "cond_encoder_state": s.encoder.state_dict(),
        "fm_cfg": s.fm_cfg.model_dump(),
        "dataset_cfg": s.split.ds_cfg.model_dump(),
        "grid_attrs": dict(read_grid_attrs(s.h5_paths[0])),
        "epoch": epoch,
        "val_loss": val_loss,
        "manifest_hash": s.manifest["hash"],
        "manifest_path": s.manifest_path,
        "git_commit": s.git_commit,
    }


def _train_epoch(s: _Session, epoch: int) -> float:
    s.model.train()
    s.encoder.train()
    total = 0.0
    for batch in s.split.train_loader:
        loss = _loss(batch, s.model, s.encoder, s.device, None)
        if not torch.isfinite(loss):
            raise TrainingError(f"non-finite training loss at epoch {epoch}")
        s.optimizer.zero_grad()
        loss.backward()
        s.optimizer.step()
        total += loss.item()
    return total / len(s.split.train_loader)


def seed_epoch_shuffle(loader: Any, base_seed: int, epoch: int) -> None:
    """The epoch order depends only on seed and epoch, so a resumed run shuffles like an uninterrupted one."""
    seed = int(np.random.SeedSequence([base_seed, epoch]).generate_state(1)[0])
    loader.sampler.generator = torch.Generator().manual_seed(seed)


def _run_epoch(s: _Session, h: _History, epoch: int) -> None:
    """Train, validate, select on val only, and save a resumable state."""
    seed_epoch_shuffle(s.split.train_loader, s.cfg.train.seed, epoch)
    train_loss = _train_epoch(s, epoch)
    val_loss = _validate(s.split.val_loader, s.model, s.encoder, s.device)
    if not math.isfinite(val_loss):
        raise TrainingError(f"non-finite validation loss at epoch {epoch}")
    h.train.append(train_loss)
    h.val.append(val_loss)
    mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
    if val_loss < h.best:
        h.best, h.since_best = val_loss, 0
        _atomic_torch_save(_checkpoint(s, epoch, val_loss), s.out / "best.pt")
        mlflow.log_metric("best_val_loss", val_loss, step=epoch)
    else:
        h.since_best += 1
    h.stopped = h.since_best >= s.cfg.train.patience
    state = {
        **_checkpoint(s, epoch, val_loss),
        "optimizer_state": s.optimizer.state_dict(),
        "val_history": h.val,
        "train_history": h.train,
        "best_val": h.best,
        "since_best": h.since_best,
        "stopped_early": h.stopped,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(s.device) if s.device.type == "cuda" else None,
    }
    _atomic_torch_save(state, s.out / "latest.pt")
    logger.info("epoch %d train=%.6f val=%.6f best=%.6f", epoch, train_loss, val_loss, h.best)


def train_baseline(
    cfg: UQConfig,
    manifest_path: str | Path,
    out_dir: str | Path,
    progress: Callable[[int, int], None] | None = None,
    git_commit: str = "unknown",
    should_pause: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train or resume; ``best.pt`` is selected on the manifest's val runs only.

    ``should_pause`` is asked before each epoch once one is saved, so finished training never pauses.
    """
    t = cfg.train
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = _build_session(cfg, manifest_path, out, git_commit)
    history = _restore(session)
    start = len(history.val)
    paused = False
    args = SimpleNamespace(
        mlflow_uri=cfg.run.mlflow_uri, mlflow_experiment=cfg.run.mlflow_experiment, checkpoint_dir=str(out)
    )
    tracker, _ = setup_run(args, timestamped=False)
    with tracker.start_run(
        run_name=f"uq_baseline_{cfg.mode}",
        config={**t.model_dump(mode="json"), "manifest_hash": session.manifest["hash"],
                "manifest_path": str(manifest_path), "resumed_at_epoch": start, "git_commit": git_commit},
        tags={"type": "fm_training", "mode": "uq_run_level", "mask_source": t.mask_source},
    ):  # fmt: skip
        for epoch in range(start, t.max_epochs):
            if history.stopped:
                break
            if should_pause is not None and history.val and should_pause():
                paused = True
                break
            _run_epoch(session, history, epoch)
            if progress is not None:
                progress(epoch + 1, t.max_epochs)
    if not (out / "best.pt").exists():
        raise TrainingError("training ended without a best checkpoint")
    return {
        "val_history": history.val,
        "train_history": history.train,
        "best_val": history.best,
        "best_epoch": history.val.index(min(history.val)),
        "epochs_run": len(history.val),
        "epochs_run_this_call": len(history.val) - start,
        "stopped_early": history.stopped,
        "paused": paused,
    }
