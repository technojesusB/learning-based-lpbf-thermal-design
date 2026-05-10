"""train_fm_patches.py — Training for FM thermal surrogate using 64x64x64 patches.

Patches are centered around the laser spot (hot-spot) to ensure the model focuses
on the most dynamic regions while staying within GPU memory limits.
"""

import argparse
import logging
import math
import random
import datetime
import os
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import mlflow

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import fm_loss, sample_noise, interpolate
from tqdm import tqdm

# Tracking Convention
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.factory import build_tracker

logger = logging.getLogger(__name__)

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
# Training Logic
# ---------------------------------------------------------------------------

def _log_validation_image(model, tracker, cond_encoder, batch, epoch, device):
    import matplotlib.pyplot as plt
    model.eval()
    cond_encoder.eval()
    with torch.no_grad():
        # Get data: (B, 1, Nz, Ny, Nx)
        T_in = batch["T_in"][:1].to(device).squeeze(1)
        T_tgt = batch["T_target"][:1].to(device).squeeze(1)
        mask = batch["mask"][:1].to(device).squeeze(1)
        Q = batch["Q"][:1].to(device).squeeze(1)
        cond_raw = batch["conditioning"][:1].to(device)
        cond_emb = cond_encoder(cond_raw)
        
        # Inference
        x = sample_noise(T_in)
        n_steps = 25
        dt = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((1,), i * dt, device=device)
            v = model(torch.cat([x, mask, Q], dim=1), tau, cond_emb)
            x = x + v * dt
            
        # Slices
        # Since it's a 64^3 patch, laser is roughly at (32, 32) in XY
        # Shape is (B, C, Nz, Ny, Nx) -> Ny is index 3
        mid_y = T_tgt.shape[3] // 2
        
        # GT
        gt_xy = T_tgt[0, 0, -1].cpu().numpy()
        gt_xz = T_tgt[0, 0, :, mid_y, :].cpu().numpy()
        
        # Pred
        pred_xy = x[0, 0, -1].cpu().numpy()
        pred_xz = x[0, 0, :, mid_y, :].cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Surface
        axes[0, 0].imshow(gt_xy, vmin=0, vmax=1, cmap="magma")
        axes[0, 0].set_title("GT Surface (XY)")
        axes[0, 1].imshow(pred_xy, vmin=0, vmax=1, cmap="magma")
        axes[0, 1].set_title(f"Pred Surface (Ep {epoch})")
        
        # Depth
        axes[1, 0].imshow(gt_xz, vmin=0, vmax=1, cmap="magma", aspect="equal", origin="lower")
        axes[1, 0].set_title("GT Depth (XZ)")
        axes[1, 1].imshow(pred_xz, vmin=0, vmax=1, cmap="magma", aspect="equal", origin="lower")
        axes[1, 1].set_title(f"Pred Depth (Ep {epoch})")
        
        for ax in axes.flatten():
            ax.axis("off")
            
        plt.tight_layout()
        path = f"val_epoch_{epoch:03d}.png"
        plt.savefig(path, dpi=120)
        tracker.log_artifact(path, artifact_path="plots")
        plt.close()
        if os.path.exists(path): os.remove(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="fm_patches")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    ds_cfg = FMDatasetConfig(h5_paths=[args.h5], Q_ref=1.35e15)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds = PatchFMThermalDataset(full_ds, patch_size=64)
    
    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(patch_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    logger.info(f"Dataset Split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    fm_cfg = FMConfig(base_channels=32, depth=3, cond_dim=len(ds_cfg.conditioning_keys), cond_embed_dim=128)
    model = VelocityNet(fm_cfg).to(device)
    cond_encoder = ConditioningEncoder(fm_cfg.cond_dim, 128).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr)
    
    # 3. TRACKING CONVENTION
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    
    tracking_cfg = TrackingConfig(
        enabled=True,
        backend="mlflow",
        experiment_name=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    )
    tracker = build_tracker(tracking_cfg)
    
    with tracker.start_run(
        run_name=f"patch_run_{datetime.datetime.now().strftime('%H%M%S')}",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "n_train": n_train},
        tags={"type": "fm_training", "mode": "patches"}
    ) as run:
        
        best_val_loss = float('inf')
        os.makedirs("checkpoints/fm", exist_ok=True)

        pbar_epochs = tqdm(range(args.epochs), desc="Epochs")
        for epoch in pbar_epochs:
            model.train()
            cond_encoder.train()
            train_loss = 0
            
            pbar_batches = tqdm(train_loader, desc=f"Batches (Epoch {epoch})", leave=False)
            for batch in pbar_batches:
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
                
                train_loss += loss.item()
                pbar_batches.set_postfix({"loss": loss.item()})
            
            avg_train_loss = float(train_loss / len(train_loader))
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            # Validation Phase
            model.eval()
            cond_encoder.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
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
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            
            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "cond_encoder_state": cond_encoder.state_dict(),
                    "fm_cfg": fm_cfg.model_dump(),
                    "dataset_cfg": ds_cfg.model_dump(),
                    "epoch": epoch,
                    "val_loss": best_val_loss
                }, "checkpoints/fm/best.pt")
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                logger.info(f"New best model at epoch {epoch}: {best_val_loss:.6f}")

            pbar_epochs.set_postfix({"train": f"{avg_train_loss:.4f}", "val": f"{avg_val_loss:.4f}"})
            
            if epoch % 10 == 0:
                val_batch = next(iter(val_loader))
                _log_validation_image(model, run, cond_encoder, val_batch, epoch, device)

        # 5. FINAL TEST EVALUATION (Reload Best Model)
        logger.info("Starting Final Test Evaluation on BEST model...")
        # weights_only=False required: checkpoint includes fm_cfg/dataset_cfg dicts
        checkpoint = torch.load("checkpoints/fm/best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])
        
        model.eval()
        cond_encoder.eval()
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
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
                test_loss += loss.item()
                
                # Log first 4 samples as visual indicators
                if i < 4:
                    _log_validation_image(model, run, cond_encoder, batch, 990 + i, device)
        
        avg_test_loss = test_loss / len(test_loader)
        mlflow.log_metric("test_loss_final", avg_test_loss)
        logger.info(f"Final Best Model Test Loss: {avg_test_loss:.6f}")
        
        # Save latest as well
        torch.save({
            "model_state": model.state_dict(),
            "cond_encoder_state": cond_encoder.state_dict(),
            "fm_cfg": fm_cfg.model_dump(),
            "dataset_cfg": ds_cfg.model_dump()
        }, "checkpoints/fm/latest.pt")

if __name__ == "__main__":
    main()
