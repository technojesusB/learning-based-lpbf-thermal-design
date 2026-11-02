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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import torch

from neural_pbf.data.patch_dataset import read_grid_attrs
from neural_pbf.data.splits import load_manifest
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
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
    v = model(torch.cat([interpolate(noise, target, tau), mask, q], dim=1), tau, cond)
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
    state = torch.load(path, map_location=device, weights_only=False)
    if state["manifest_hash"] != manifest_hash:
        raise TrainingError(f"{path} was trained under another manifest; refuse to resume")
    return state


def train_baseline(
    cfg: UQConfig,
    manifest_path: str | Path,
    out_dir: str | Path,
    progress: Callable[[int, int], None] | None = None,
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Train or resume; ``best.pt`` is selected on the manifest's val runs only."""
    t = cfg.train
    device = torch.device(cfg.run.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    h5_paths = sorted({p for runs in manifest["sets"].values() for r in runs for p, _ in r["sample_keys"]})

    torch.manual_seed(t.seed)
    split = build_patch_data(
        h5_paths, patch_size=t.patch_size, batch_size=t.batch_size, seed=t.seed,
        manifest_path=str(manifest_path), mask_source=t.mask_source,
    )  # fmt: skip
    if not len(split.train_ds) or not len(split.val_ds):
        raise TrainingError("train or val set is empty")

    fm_cfg = FMConfig(
        base_channels=t.base_channels, depth=t.depth,
        cond_dim=len(split.ds_cfg.conditioning_keys), cond_embed_dim=t.cond_embed_dim,
        tau_embed_dim=t.cond_embed_dim,
    )  # fmt: skip
    model = VelocityNet(fm_cfg).to(device)
    encoder = ConditioningEncoder(fm_cfg.cond_dim, t.cond_embed_dim).to(device)
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=t.lr)

    val_history: list[float] = []
    train_history: list[float] = []
    best, since_best, start, stopped = math.inf, 0, 0, False
    resume = _load_resume(out / "latest.pt", manifest["hash"], device)
    if resume is not None:
        model.load_state_dict(resume["model_state"])
        encoder.load_state_dict(resume["cond_encoder_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        val_history, train_history = list(resume["val_history"]), list(resume["train_history"])
        best, since_best, stopped = resume["best_val"], resume["since_best"], resume["stopped_early"]
        start = len(val_history)
        torch.set_rng_state(resume["rng_cpu"].cpu())
        if device.type == "cuda" and resume["rng_cuda"] is not None:
            torch.cuda.set_rng_state(resume["rng_cuda"].cpu(), device)
        logger.info("Resuming at epoch %d (best val %.6f)", start, best)

    def payload(epoch: int, val_loss: float) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "cond_encoder_state": encoder.state_dict(),
            "fm_cfg": fm_cfg.model_dump(),
            "dataset_cfg": split.ds_cfg.model_dump(),
            "grid_attrs": dict(read_grid_attrs(h5_paths[0])),
            "epoch": epoch,
            "val_loss": val_loss,
            "manifest_hash": manifest["hash"],
            "manifest_path": str(manifest_path),
            "git_commit": git_commit,
        }

    ran = 0
    args = SimpleNamespace(
        mlflow_uri=cfg.run.mlflow_uri, mlflow_experiment=cfg.run.mlflow_experiment, checkpoint_dir=str(out)
    )
    tracker, _ = setup_run(args, timestamped=False)
    with tracker.start_run(
        run_name=f"uq_baseline_{cfg.mode}",
        config={**t.model_dump(mode="json"), "manifest_hash": manifest["hash"], "manifest_path": str(manifest_path),
                "resumed_at_epoch": start, "git_commit": git_commit},
        tags={"type": "fm_training", "mode": "uq_run_level", "mask_source": t.mask_source},
    ):  # fmt: skip
        for epoch in range(start, t.max_epochs):
            if stopped:
                break
            model.train()
            encoder.train()
            total = 0.0
            for batch in split.train_loader:
                loss = _loss(batch, model, encoder, device, None)
                if not torch.isfinite(loss):
                    raise TrainingError(f"non-finite training loss at epoch {epoch}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            train_loss = total / len(split.train_loader)
            val_loss = _validate(split.val_loader, model, encoder, device)
            if not math.isfinite(val_loss):
                raise TrainingError(f"non-finite validation loss at epoch {epoch}")
            train_history.append(train_loss)
            val_history.append(val_loss)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            ran += 1

            if val_loss < best:
                best, since_best = val_loss, 0
                _atomic_torch_save(payload(epoch, val_loss), out / "best.pt")
                mlflow.log_metric("best_val_loss", val_loss, step=epoch)
            else:
                since_best += 1
            stopped = since_best >= t.patience

            state = {
                **payload(epoch, val_loss),
                "optimizer_state": optimizer.state_dict(),
                "val_history": val_history,
                "train_history": train_history,
                "best_val": best,
                "since_best": since_best,
                "stopped_early": stopped,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            _atomic_torch_save(state, out / "latest.pt")
            logger.info("epoch %d train=%.6f val=%.6f best=%.6f", epoch, train_loss, val_loss, best)
            if progress is not None:
                progress(epoch + 1, t.max_epochs)

    if not (out / "best.pt").exists():
        raise TrainingError("training ended without a best checkpoint")
    return {
        "val_history": val_history,
        "train_history": train_history,
        "best_val": best,
        "best_epoch": val_history.index(min(val_history)),
        "epochs_run": len(val_history),
        "epochs_run_this_call": ran,
        "stopped_early": stopped,
    }
