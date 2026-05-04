"""train_fm.py — Offline training for the Flow Matching thermal surrogate.

Usage::

    uv run python scripts/train_fm.py \\
        --h5 data/offline_dataset.h5 \\
        --epochs 100 \\
        --batch_size 4 \\
        --output_dir checkpoints/fm \\
        --device cuda

Tracking is done via MLflow (project convention).  Set MLFLOW_TRACKING_URI or
rely on the default ./mlruns directory.  View runs with: uv run mlflow ui.

Scheduled sampling starts disabled (epsilon_sched_max=0.0 — pure teacher
forcing).  Enable once 1-step prediction is stable:
    --epsilon_sched_max 0.3 --epsilon_sched_warmup 20
"""
from __future__ import annotations

import argparse
import datetime
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train config (not Pydantic — scripts are excluded from typecheck)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    h5_paths: list[str] = field(default_factory=list)
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    epsilon_sched_warmup: int = 20
    epsilon_sched_max: float = 0.0        # 0 = pure teacher-forcing
    n_steps_train_rollout: int = 8        # steps used for self-rollout samples
    pde_weight: float = 0.0               # physics residual weight (0 = disabled)
    seed: int = 42
    output_dir: str = "checkpoints/fm"
    mlflow_experiment: str = "fm_surrogate"
    mlflow_run_name: str | None = None
    num_workers: int = 0
    # FM model architecture
    base_channels: int = 32
    depth: int = 3
    cond_embed_dim: int = 128
    tau_embed_dim: int = 128
    n_inference_steps: int = 25
    # Dataset
    q_ref: float = 1e12
    t_ref: float = 2000.0
    t_ambient: float = 300.0
    mlflow_tracking_uri: str | None = "sqlite:///mlflow.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epsilon(epoch: int, cfg: TrainConfig) -> float:
    if cfg.epsilon_sched_max <= 0.0 or cfg.epsilon_sched_warmup <= 0:
        return 0.0
    return min(cfg.epsilon_sched_max, (epoch / cfg.epsilon_sched_warmup) * cfg.epsilon_sched_max)


@torch.no_grad()
def _self_rollout(
    model: nn.Module,
    cond_encoder: nn.Module,
    T_in_norm: torch.Tensor,     # (B, 1, Nz, Ny, Nx)
    mask: torch.Tensor,          # (B, 1, Nz, Ny, Nx)
    Q_norm: torch.Tensor,        # (B, 1, Nz, Ny, Nx)
    cond: torch.Tensor,          # (B, cond_embed_dim)
    n_steps: int,
) -> torch.Tensor:
    """One-shot rollout from noise → predicted T (normalised). Returns detached tensor."""
    from neural_pbf.models.generative.fm.flow import interpolate, sample_noise

    noise = sample_noise(T_in_norm)
    x_T = noise.clone()
    dτ = 1.0 / n_steps
    for i in range(n_steps):
        tau_i = torch.full((T_in_norm.shape[0],), i * dτ, device=T_in_norm.device)
        x_full = torch.cat([x_T, mask, Q_norm], dim=1)
        v = model(x_full, tau_i, cond)
        x_T = x_T + v * dτ
    return x_T.detach()


def _log_params_flat(prefix: str, d: dict) -> None:
    """Log a dict of params to mlflow with a prefix, flattening nested dicts."""
    for k, v in d.items():
        if isinstance(v, dict):
            _log_params_flat(f"{prefix}.{k}", v)
        else:
            mlflow.log_param(f"{prefix}.{k}", v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train FM thermal surrogate")
    parser.add_argument("--h5", nargs="+", required=True, help="HDF5 dataset path(s)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--epsilon_sched_max", type=float, default=0.0)
    parser.add_argument("--epsilon_sched_warmup", type=int, default=20)
    parser.add_argument("--pde_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints/fm")
    parser.add_argument("--mlflow_experiment", type=str, default="lpbf")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cond_embed_dim", type=int, default=128)
    parser.add_argument("--n_inference_steps", type=int, default=25)
    args = parser.parse_args()

    cfg = TrainConfig(
        h5_paths=args.h5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        epsilon_sched_max=args.epsilon_sched_max,
        epsilon_sched_warmup=args.epsilon_sched_warmup,
        pde_weight=args.pde_weight,
        seed=args.seed,
        output_dir=args.output_dir,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        depth=args.depth,
        cond_embed_dim=args.cond_embed_dim,
        tau_embed_dim=args.cond_embed_dim,
        n_inference_steps=args.n_inference_steps,
        mlflow_tracking_uri=args.mlflow_uri,
    )

    # Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ---------------------------------------------------------------
    from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset

    ds_cfg = FMDatasetConfig(
        h5_paths=cfg.h5_paths,
        T_ref=cfg.t_ref,
        T_ambient=cfg.t_ambient,
        Q_ref=cfg.q_ref,
    )
    full_ds = FMThermalDataset(ds_cfg)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * cfg.val_fraction))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    logger.info("Dataset: %d train, %d val samples", n_train, n_val)

    # ---- Model -----------------------------------------------------------------
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
    from neural_pbf.models.generative.fm.config import FMConfig
    from neural_pbf.models.generative.fm.flow import (
        compute_physics_residuum,
        fm_loss,
        interpolate,
        sample_noise,
    )
    from neural_pbf.models.generative.fm.velocity_net import VelocityNet

    fm_cfg = FMConfig(
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        cond_dim=len(ds_cfg.conditioning_keys),
        cond_embed_dim=cfg.cond_embed_dim,
        tau_embed_dim=cfg.tau_embed_dim,
        n_inference_steps=cfg.n_inference_steps,
    )

    model = VelocityNet(fm_cfg).to(device)
    cond_encoder = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in cond_encoder.parameters())
    logger.info("FM model: %s params (VelocityNet + ConditioningEncoder)", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- MLflow run (via project's TrackingConfig pattern) ---------------------
    from neural_pbf.schemas.tracking import TrackingConfig
    from neural_pbf.tracking.factory import build_tracker

    tracking_cfg = TrackingConfig(
        enabled=True,
        backend="mlflow",
        experiment_name=cfg.mlflow_experiment,
        run_name=cfg.mlflow_run_name,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
    )
    tracker = build_tracker(tracking_cfg)

    run_name = cfg.mlflow_run_name or f"fm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with tracker.start_run(run_name=run_name, config={}, tags={"type": "fm_training"}):
        # Log all configs as params
        mlflow.log_params({f"fm.{k}": v for k, v in fm_cfg.model_dump().items()})
        mlflow.log_params({
            "train.epochs": cfg.epochs,
            "train.batch_size": cfg.batch_size,
            "train.lr": cfg.lr,
            "train.weight_decay": cfg.weight_decay,
            "train.epsilon_sched_max": cfg.epsilon_sched_max,
            "train.pde_weight": cfg.pde_weight,
            "train.seed": cfg.seed,
            "train.n_train": n_train,
            "train.n_val": n_val,
            "train.device": str(device),
        })

        best_val_loss = math.inf

        for epoch in range(cfg.epochs):
            epsilon = _epsilon(epoch, cfg)

            # --- Training ---
            model.train()
            cond_encoder.train()
            train_losses: list[float] = []
            grad_norms: list[float] = []

            for batch in train_loader:
                T_in_norm = batch["T_in"].to(device)       # (B, 1, 1, Nz, Ny, Nx)
                T_tgt_norm = batch["T_target"].to(device)
                Q_norm = batch["Q"].to(device)
                mask = batch["mask"].to(device)
                conditioning = batch["conditioning"].to(device)

                # Squeeze the extra batch-dim that DataLoader adds when samples are (1,1,Nz,Ny,Nx)
                # After DataLoader batching: (B, 1, 1, Nz, Ny, Nx) → squeeze dim 1
                if T_in_norm.ndim == 6:
                    T_in_norm = T_in_norm.squeeze(1)
                    T_tgt_norm = T_tgt_norm.squeeze(1)
                    Q_norm = Q_norm.squeeze(1)
                    mask = mask.squeeze(1)

                cond_emb = cond_encoder(conditioning)

                # Optional scheduled sampling: use model rollout as x_prev
                if epsilon > 0.0 and random.random() < epsilon:
                    model.eval()
                    cond_encoder.eval()
                    T_in_norm = _self_rollout(
                        model, cond_encoder, T_in_norm, mask, Q_norm,
                        cond_emb, cfg.n_steps_train_rollout,
                    )
                    model.train()
                    cond_encoder.train()
                    cond_emb = cond_encoder(conditioning)

                # FM training step
                noise = sample_noise(T_tgt_norm)
                tau = torch.rand(T_tgt_norm.shape[0], device=device)
                x_tau_T = interpolate(noise, T_tgt_norm, tau)

                x_tau_full = torch.cat([x_tau_T, mask, Q_norm], dim=1)
                v_pred = model(x_tau_full, tau, cond_emb)

                loss = fm_loss(v_pred, noise, T_tgt_norm)

                if cfg.pde_weight > 0.0:
                    # physics residuum is currently a zero placeholder
                    pass  # loss += pde_weight * compute_physics_residuum(...)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cond_encoder.parameters()), max_norm=1.0
                ).item()
                optimizer.step()

                train_losses.append(loss.item())
                grad_norms.append(grad_norm)

            # --- Validation ---
            model.eval()
            cond_encoder.eval()
            val_losses: list[float] = []

            with torch.no_grad():
                for batch in val_loader:
                    T_in_norm = batch["T_in"].to(device)
                    T_tgt_norm = batch["T_target"].to(device)
                    Q_norm = batch["Q"].to(device)
                    mask = batch["mask"].to(device)
                    conditioning = batch["conditioning"].to(device)

                    if T_tgt_norm.ndim == 6:
                        T_in_norm = T_in_norm.squeeze(1)
                        T_tgt_norm = T_tgt_norm.squeeze(1)
                        Q_norm = Q_norm.squeeze(1)
                        mask = mask.squeeze(1)

                    cond_emb = cond_encoder(conditioning)
                    noise = sample_noise(T_tgt_norm)
                    tau = torch.rand(T_tgt_norm.shape[0], device=device)
                    x_tau_T = interpolate(noise, T_tgt_norm, tau)
                    x_tau_full = torch.cat([x_tau_T, mask, Q_norm], dim=1)
                    v_pred = model(x_tau_full, tau, cond_emb)
                    val_losses.append(fm_loss(v_pred, noise, T_tgt_norm).item())

            train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")
            val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
            avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epsilon": epsilon,
                "grad_norm": avg_grad,
            }, step=epoch)

            logger.info(
                "Epoch %d/%d  train=%.6f  val=%.6f  ε=%.3f  ‖g‖=%.3f",
                epoch + 1, cfg.epochs, train_loss, val_loss, epsilon, avg_grad,
            )

            # Checkpoint latest
            latest_path = output_dir / "latest.pt"
            torch.save({
                "model_state": model.state_dict(),
                "cond_encoder_state": cond_encoder.state_dict(),
                "fm_cfg": fm_cfg.model_dump(),
                "dataset_cfg": ds_cfg.model_dump(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, latest_path)

            # Checkpoint best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "cond_encoder_state": cond_encoder.state_dict(),
                    "fm_cfg": fm_cfg.model_dump(),
                    "dataset_cfg": ds_cfg.model_dump(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, best_path)
                logger.info("  ↳ new best val_loss=%.6f  saved %s", best_val_loss, best_path)
                mlflow.log_artifact(str(best_path))

        mlflow.log_artifact(str(output_dir / "latest.pt"))
        logger.info("Training complete. Best val_loss=%.6f", best_val_loss)


if __name__ == "__main__":
    main()
