"""train_fm_patches.py — Training for FM thermal surrogate using 64x64x64 patches.

Patches are centered around the laser spot (hot-spot) to ensure the model focuses
on the most dynamic regions while staying within GPU memory limits.
"""

import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.eval.metrics.geometry import evaluate_physical_metrics
from neural_pbf.eval.reporting.hardware import (
    epoch0_profiler,
    log_gpu_telemetry,
    log_step_timing,
)
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.losses import loss_panel
from neural_pbf.eval.viz.spatial import (  # noqa: F401
    gallery_evolution,
    gallery_test,
    val_grid_2x2,
)
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.factory import build_tracker

logger = logging.getLogger(__name__)

_T_LIQUIDUS_NORM: float = 0.6
_RUN_NAME = "v1_baseline_unet"

# ---------------------------------------------------------------------------
# Patch Dataset Wrapper
# ---------------------------------------------------------------------------

class PatchFMThermalDataset(Dataset):
    """Wraps FMThermalDataset to return 64x64x64 patches centered on laser."""

    def __init__(self, base_ds: FMThermalDataset, patch_size: int = 64):
        self.base_ds = base_ds
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        path, sample_key = self.base_ds._keys[idx]
        with h5py.File(path, "r") as f:
            Lx, Ly = f.attrs["Lx_m"], f.attrs["Ly_m"]
            Nx, Ny = f.attrs["Nx"], f.attrs["Ny"]
            dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
            grp = f["samples"][sample_key]
            x0, y0 = grp.attrs["x"], grp.attrs["y"]

        ix, iy = int(round(x0 / dx)), int(round(y0 / dy))
        half = self.patch_size // 2
        x_start = max(0, min(Nx - self.patch_size, ix - half))
        y_start = max(0, min(Ny - self.patch_size, iy - half))

        ps = self.patch_size
        patched = {
            key: sample[key][:, :, :, y_start : y_start + ps, x_start : x_start + ps]
            for key in ["T_in", "T_target", "Q", "mask"]
        }
        return {**{k: v for k, v in sample.items() if k not in patched}, **patched}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _infer_pred_patches(
    model, cond_encoder, batch: dict, device: torch.device, n_steps: int = 25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler rollout for one batch; returns (T_tgt, T_pred) tensors on CPU.

    Caller is responsible for restoring model.train() after this call.
    """
    model.eval()
    cond_encoder.eval()
    with torch.no_grad():
        T_tgt = batch["T_target"][:1].to(device).squeeze(1)
        mask = batch["mask"][:1].to(device).squeeze(1)
        Q = batch["Q"][:1].to(device).squeeze(1)
        cond_emb = cond_encoder(batch["conditioning"][:1].to(device))
        x = sample_noise(T_tgt)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((1,), i * dt, device=device)
            x = x + model(torch.cat([x, mask, Q], dim=1), tau, cond_emb) * dt
    return T_tgt.cpu(), x.cpu()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_train_epoch_patches(
    model: VelocityNet,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epoch: int,
    device: torch.device,
    profiler: Any = None,
) -> float:
    model.train()
    cond_encoder.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"Batches (Epoch {epoch})", leave=False):
        T_tgt = batch["T_target"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond = batch["conditioning"].to(device)
        cond_emb = cond_encoder(cond)
        noise = sample_noise(T_tgt)
        tau = torch.rand(T_tgt.shape[0], device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
        loss = fm_loss(v_pred, noise, T_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        if profiler is not None:
            profiler.step()
    return total / max(len(loader), 1)


def _run_val_epoch_patches(
    model: VelocityNet,
    cond_encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    cond_encoder.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            T_tgt = batch["T_target"].to(device).squeeze(1)
            mask = batch["mask"].to(device).squeeze(1)
            Q = batch["Q"].to(device).squeeze(1)
            cond = batch["conditioning"].to(device)
            cond_emb = cond_encoder(cond)
            noise = sample_noise(T_tgt)
            tau = torch.rand(T_tgt.shape[0], device=device)
            x_tau = interpolate(noise, T_tgt, tau)
            v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
            total += fm_loss(v_pred, noise, T_tgt).item()
    return total / max(len(loader), 1)


def _run_test_phase_patches(
    model: VelocityNet,
    cond_encoder: nn.Module,
    test_ds: Any,
    args: argparse.Namespace,
    ckpt_dir: Path,
    fm_cfg: FMConfig,
    ds_cfg: FMDatasetConfig,
    run: Any,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    logger.info("Starting Final Test Evaluation on BEST model...")
    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        logger.warning("No best checkpoint at %s; skipping test phase.", best_ckpt)
        return []
    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])
    model.eval()
    cond_encoder.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    mse_list: list[float] = []
    phys_metrics: dict[str, list[float]] = {}
    gallery_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            T_tgt = batch["T_target"].to(device).squeeze(1)
            mask = batch["mask"].to(device).squeeze(1)
            Q = batch["Q"].to(device).squeeze(1)
            cond_emb = cond_encoder(batch["conditioning"].to(device))

            x = sample_noise(T_tgt)
            dt = 1.0 / args.test_n_steps
            for step_i in range(args.test_n_steps):
                tau_s = torch.full((T_tgt.shape[0],), step_i * dt, device=device)
                x = x + model(torch.cat([x, mask, Q], dim=1), tau_s, cond_emb) * dt
            mse_list.append(F.mse_loss(x, T_tgt).item())
            sample_phys = evaluate_physical_metrics(x, T_tgt, _T_LIQUIDUS_NORM)
            for k, val in sample_phys.items():
                phys_metrics.setdefault(k, []).append(val)

            if i < 4:
                log_figure(run, val_grid_2x2(T_tgt[:1].cpu(), x[:1].cpu(), epoch=990 + i),
                         f"test_sample_{i:03d}.png", dpi=120)
            if len(gallery_samples) < 5:
                gallery_samples.append((T_tgt[:1].cpu(), x[:1].cpu()))

    if mse_list:
        avg_mse = sum(mse_list) / len(mse_list)
        mlflow.log_metric("test_mse_rollout", avg_mse)
        if phys_metrics:
            mlflow.log_metrics({k: sum(v) / len(v) for k, v in phys_metrics.items()})
        logger.info("Test — MSE (rollout): %.6f", avg_mse)
        metrics_detail = {
            "samples": [
                {"idx": idx, "mse": mse_list[idx],
                 **{k: phys_metrics[k][idx] for k in phys_metrics}}
                for idx in range(len(mse_list))
            ],
            "summary": {
                "n_samples": len(mse_list),
                "mse_mean": avg_mse,
                **{k: sum(v) / len(v) for k, v in phys_metrics.items()},
            },
        }
        json_path = str(ckpt_dir / "test_metrics_detailed.json")
        with open(json_path, "w") as _jf:
            json.dump(metrics_detail, _jf, indent=2)
        run.log_artifact(json_path, artifact_path="eval")
    else:
        logger.warning("No test samples evaluated — skipping test metrics.")
    return gallery_samples


def _run_train_loop_patches(
    model: VelocityNet,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    ckpt_dir: Path,
    fm_cfg: FMConfig,
    ds_cfg: FMDatasetConfig,
    run: Any,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[torch.Tensor], list[str], list[float], list[float]]:
    best_val_loss = float("inf")
    val_batch_fixed = next(iter(val_loader))
    pred_history: list[torch.Tensor] = []
    epoch_labels: list[str] = []
    T_gt_fixed: torch.Tensor | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        t0 = time.perf_counter()
        if epoch == 0:
            with epoch0_profiler(len(train_loader), _RUN_NAME) as prof:
                avg_train = _run_train_epoch_patches(
                    model, cond_encoder, optimizer, train_loader, epoch, device, profiler=prof
                )
        else:
            avg_train = _run_train_epoch_patches(
                model, cond_encoder, optimizer, train_loader, epoch, device
            )
        epoch_secs = time.perf_counter() - t0

        mlflow.log_metric("train_loss", avg_train, step=epoch)
        log_step_timing(epoch, epoch_secs / max(len(train_loader), 1))
        log_gpu_telemetry(epoch)

        avg_val = _run_val_epoch_patches(model, cond_encoder, val_loader, device)
        mlflow.log_metric("val_loss", avg_val, step=epoch)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cond_encoder_state": cond_encoder.state_dict(),
                    "fm_cfg": fm_cfg.model_dump(),
                    "dataset_cfg": ds_cfg.model_dump(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                ckpt_dir / "best.pt",
            )
            mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            logger.info("New best model at epoch %d: %.6f", epoch, best_val_loss)

        if epoch % 10 == 0:
            try:
                T_gt_fixed, T_pred_snap = _infer_pred_patches(model, cond_encoder, val_batch_fixed, device)
                pred_history.append(T_pred_snap)
                epoch_labels.append(f"Ep {epoch}")
                log_figure(run, val_grid_2x2(T_gt_fixed, T_pred_snap, epoch=epoch),
                         f"val_epoch_{epoch:03d}.png", dpi=120)
            finally:
                model.train()
                cond_encoder.train()

    return T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="lpbf_surrogate_benchmark")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline_unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_n_steps", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_cfg = FMDatasetConfig(h5_paths=[args.h5], Q_ref=1.35e15)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds = PatchFMThermalDataset(full_ds, patch_size=64)

    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        patch_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    logger.info("Dataset Split: Train=%d, Val=%d, Test=%d", n_train, n_val, n_test)

    fm_cfg = FMConfig(
        base_channels=32, depth=3,
        cond_dim=len(ds_cfg.conditioning_keys), cond_embed_dim=128,
    )
    model = VelocityNet(fm_cfg).to(device)
    cond_encoder = ConditioningEncoder(fm_cfg.cond_dim, 128).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(args.checkpoint_dir) / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(TrackingConfig(
        enabled=True, backend="mlflow",
        experiment_name=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    ))

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "n_train": n_train},
        tags={"type": "fm_training", "mode": "patches"},
    ) as run:
        T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses = _run_train_loop_patches(
            model, cond_encoder, optimizer, train_loader, val_loader,
            args, ckpt_dir, fm_cfg, ds_cfg, run, device,
        )
        if train_losses and val_losses:
            log_figure(run, loss_panel(train_losses, val_losses, title=f"{_RUN_NAME} — Loss Panel"),
                     "loss_panel.png")

        if T_gt_fixed is not None and pred_history:
            log_figure(run, gallery_evolution(T_gt_fixed, pred_history, epoch_labels=epoch_labels,
                                            title=f"{_RUN_NAME} — Training Evolution", dpi=150),
                     "gallery_evolution.png")

        gallery_samples = _run_test_phase_patches(
            model, cond_encoder, test_ds, args, ckpt_dir, fm_cfg, ds_cfg, run, device,
        )
        if gallery_samples:
            log_figure(run, gallery_test(gallery_samples[:5], title=f"{_RUN_NAME} — Test Samples", dpi=150),
                     "gallery_test.png")

        torch.save(
            {
                "model_state": model.state_dict(),
                "cond_encoder_state": cond_encoder.state_dict(),
                "fm_cfg": fm_cfg.model_dump(),
                "dataset_cfg": ds_cfg.model_dump(),
            },
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
