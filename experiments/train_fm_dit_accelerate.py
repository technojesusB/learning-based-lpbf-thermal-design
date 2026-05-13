"""train_fm_dit_accelerate.py — Physics-informed 3D-DiT with BF16 and ReLoBRaLo.

Extends train_fm_dit_physics.py with:
  - HuggingFace Accelerate: BF16 mixed precision, multi-GPU-ready device placement.
  - ReLoBRaLo: adaptive lambda for physics/FM loss balancing (handles e+26 magnitudes).
  - Deadlock prevention: all list appends use .detach().cpu().item().
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from experiments.train_fm_dit import (
    _T_LIQUIDUS_NORM,
)
from experiments.train_fm_dit_rope import (
    PatchFMThermalDatasetWithOrigin,
    VelocityDiTRoPE,
    _collate_with_strings,
    denorm_cond_batch,
    patch_center_coords_mm,
    physics_heat_residual,
    read_grid_attrs,
)
from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.eval.metrics.geometry import evaluate_physical_metrics, iou_melt_volumes
from neural_pbf.eval.reporting.hardware import (
    epoch0_profiler,
    log_gpu_telemetry,
    log_step_timing,
)
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.losses import loss_panel
from neural_pbf.eval.viz.spatial import gallery_evolution, gallery_test, val_grid_2x2
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.factory import build_tracker
from neural_pbf.training.relobralo import ReLoBRaLoWeighter

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_DIM = 288
_RUN_NAME = "v4_pinn_patches_4"


# ---------------------------------------------------------------------------
# Euler rollout (device-agnostic — no manual .to(device))
# ---------------------------------------------------------------------------


def _euler_rollout_rope_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    batch: dict[str, Any],
    n_steps: int,
    device: torch.device,
    grid_attrs: dict[str, float],
    model_patch_size: int,
) -> Tensor:
    """n-step Euler integration. Tensors assumed already on the correct device."""
    T_in = batch["T_in"].squeeze(1)
    mask = batch["mask"].squeeze(1)
    Q = batch["Q"].squeeze(1)
    cond = batch["conditioning"]
    patch_origins = batch["patch_origin"]

    B, _, D, H, W = T_in.shape
    Nz_t = D // model_patch_size
    Ny_t = H // model_patch_size
    Nx_t = W // model_patch_size

    coords_mm = patch_center_coords_mm(
        patch_origins, model_patch_size, grid_attrs, (Nz_t, Ny_t, Nx_t), device
    )

    cond_emb = cond_encoder(cond)
    x = sample_noise(T_in)
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=device)
            v = model(torch.cat([x, mask, Q], dim=1), tau, cond_emb, coords_mm)
            x = x + v * dt

    return x


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _parse_args_accel() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="lpbf_surrogate_benchmark")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v4_pinn_patches_4")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--test_n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--relobralo_alpha", type=float, default=0.999,
                        help="EMA decay for ReLoBRaLo.")
    parser.add_argument("--relobralo_beta", type=float, default=0.9,
                        help="Random lookback probability for ReLoBRaLo.")
    parser.add_argument("--relobralo_epsilon", type=float, default=1e-8,
                        help="Numerical stability term for ReLoBRaLo.")
    return parser.parse_args()


def _build_data_accel(
    args: argparse.Namespace,
) -> tuple[
    FMDatasetConfig, FMThermalDataset, Any,
    DataLoader, DataLoader, DataLoader, int, int, int,
]:
    ds_cfg = FMDatasetConfig(h5_paths=[args.h5], Q_ref=1.35e15)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds = PatchFMThermalDatasetWithOrigin(full_ds, patch_size=64)
    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        patch_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate_with_strings,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=_collate_with_strings,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=_collate_with_strings,
    )
    logger.info(
        "Dataset split — train: %d  val: %d  test: %d", n_train, n_val, n_test
    )
    return ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, n_val, n_test


def _build_models_accel(
    args: argparse.Namespace,
    ds_cfg: FMDatasetConfig,
) -> tuple[VelocityDiTRoPE, ConditioningEncoder, torch.optim.Optimizer, int]:
    """Build models and optimizer WITHOUT .to(device) — Accelerate handles placement."""
    cond_dim = len(ds_cfg.conditioning_keys)
    cond_embed_dim = 128
    model_patch_size = 4
    model = VelocityDiTRoPE(
        patch_size=model_patch_size,
        in_channels=3,
        embed_dim=_DEFAULT_EMBED_DIM,
        depth=6,
        num_heads=8,
        cond_embed_dim=cond_embed_dim,
    )
    cond_encoder = ConditioningEncoder(cond_dim, cond_embed_dim)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    return model, cond_encoder, optimizer, model_patch_size


def _setup_tracker_accel(args: argparse.Namespace) -> tuple[Path, Any]:
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(args.checkpoint_dir) / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, build_tracker(TrackingConfig(
        enabled=True, backend="mlflow",
        experiment_name=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    ))


def _train_batch_accel(
    batch: dict[str, Any],
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
    weighter: ReLoBRaLoWeighter,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> tuple[float, float, float, float]:
    """Single training step.

    Returns:
        (total_loss, fm_loss, phys_loss, lambda_phys) — all Python floats.
    """
    # Accelerate-prepared loaders deliver tensors on the correct device already
    T_tgt = batch["T_target"].squeeze(1)
    T_in = batch["T_in"].squeeze(1)
    mask = batch["mask"].squeeze(1)
    Q = batch["Q"].squeeze(1)
    cond = batch["conditioning"]
    patch_origins = batch["patch_origin"]

    B, _, D, H, W = T_tgt.shape
    coords_mm = patch_center_coords_mm(
        patch_origins, model_patch_size, grid_attrs,
        (D // model_patch_size, H // model_patch_size, W // model_patch_size),
        accelerator.device,
    )

    cond_emb = cond_encoder(cond)
    noise = sample_noise(T_tgt)
    tau = torch.rand(B, device=accelerator.device)
    x_tau = interpolate(noise, T_tgt, tau)
    v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_mm)

    loss_fm = fm_loss(v_pred, noise, T_tgt)

    rho = denorm_cond_batch(cond, ds_cfg, "rho")
    cp_val = denorm_cond_batch(cond, ds_cfg, "cp")
    k_val = denorm_cond_batch(cond, ds_cfg, "k_s")
    dx_m, dy_m, dz_m = grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"]

    phys = physics_heat_residual(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp_val, k_val,
        dx_m, dy_m, dz_m, ds_cfg,
    )

    loss_fm_val = loss_fm.detach().cpu().item()
    phys_val = phys.detach().cpu().item()
    lambda_phys = weighter.step(loss_fm_val, phys_val)

    loss = loss_fm + lambda_phys * phys

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    return loss.detach().cpu().item(), loss_fm_val, phys_val, lambda_phys


def _run_train_epoch_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epoch: int,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    accelerator: Accelerator,
    weighter: ReLoBRaLoWeighter,
    profiler: Any = None,
) -> tuple[float, float, float]:
    model.train()
    cond_encoder.train()
    loss_total = loss_fm_total = phys_total = 0.0
    n = 0
    t0 = time.perf_counter()
    for batch in tqdm(loader, desc=f"Train {epoch}", leave=False,
                      disable=not accelerator.is_main_process):
        lt, lf, ph, lam = _train_batch_accel(
            batch, model, cond_encoder, optimizer, accelerator,
            args, grid_attrs, model_patch_size, ds_cfg, weighter, scheduler,
        )
        loss_total += lt
        loss_fm_total += lf
        phys_total += ph
        n += 1
        if profiler is not None:
            profiler.step()

    avg = loss_total / max(n, 1)
    avg_fm = loss_fm_total / max(n, 1)
    avg_phys = phys_total / max(n, 1)
    if accelerator.is_main_process and n > 0:
        epoch_secs = time.perf_counter() - t0
        mlflow.log_metric("train_loss", avg, step=epoch)
        mlflow.log_metric("train_loss_fm", avg_fm, step=epoch)
        mlflow.log_metric("physics_residual", avg_phys, step=epoch)
        mlflow.log_metric("lambda_phys", weighter.lambda_phys, step=epoch)
        log_step_timing(epoch, epoch_secs / n)
        log_gpu_telemetry(epoch)
    return avg, avg_fm, avg_phys


def _run_val_epoch_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    loader: DataLoader,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    accelerator: Accelerator,
) -> float:
    model.eval()
    cond_encoder.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            T_tgt = batch["T_target"].squeeze(1)
            mask = batch["mask"].squeeze(1)
            Q = batch["Q"].squeeze(1)
            cond = batch["conditioning"]
            patch_origins = batch["patch_origin"]
            B, _, D, H, W = T_tgt.shape
            coords_mm = patch_center_coords_mm(
                patch_origins, model_patch_size, grid_attrs,
                (D // model_patch_size, H // model_patch_size, W // model_patch_size),
                accelerator.device,
            )
            cond_emb = cond_encoder(cond)
            noise = sample_noise(T_tgt)
            tau = torch.rand(B, device=accelerator.device)
            x_tau = interpolate(noise, T_tgt, tau)
            v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_mm)
            total += fm_loss(v_pred, noise, T_tgt).detach().cpu().item()
            n += 1
    return total / max(n, 1)


def _save_best_checkpoint_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> None:
    if not accelerator.is_main_process:
        return
    torch.save(
        {
            "model_state": accelerator.unwrap_model(model).state_dict(),
            "cond_encoder_state": accelerator.unwrap_model(cond_encoder).state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        ckpt_dir / "best.pt",
    )


def _run_test_phase_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    test_loader: DataLoader,
    test_ds: Any,
    full_ds: FMThermalDataset,
    ckpt_dir: Path,
    args: argparse.Namespace,
    tracker: Any,
    run: Any,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    accelerator: Accelerator,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not accelerator.is_main_process:
        return []

    logger.info("Starting test evaluation with best checkpoint …")
    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        logger.warning("No best checkpoint at %s; skipping test phase.", best_ckpt)
        return []
    ckpt = torch.load(best_ckpt, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
    accelerator.unwrap_model(cond_encoder).load_state_dict(ckpt["cond_encoder_state"])
    model.eval()
    cond_encoder.eval()

    torch.manual_seed(args.seed)
    test_subset_indices: list[int] = list(test_ds.indices)  # type: ignore[attr-defined]
    mse_list: list[float] = []
    iou_list: list[float] = []
    phys_metrics: dict[str, list[float]] = {}
    gallery_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    mapping_path = ckpt_dir / "test_sample_mapping.txt"

    with mapping_path.open("w") as f_map:
        f_map.write("local_idx,h5_path,sample_key\n")
        for local_idx, batch in enumerate(tqdm(test_loader, desc="Test")):
            original_idx = test_subset_indices[local_idx]
            h5_path_str, sample_key_str = full_ds._keys[original_idx]
            f_map.write(f"{local_idx},{h5_path_str},{sample_key_str}\n")

            # Move test batch manually (test_loader is not prepared)
            device = accelerator.device
            moved: dict[str, Any] = {}
            for k, v in batch.items():
                moved[k] = v.to(device) if isinstance(v, Tensor) else v

            T_tgt = moved["T_target"].squeeze(1)
            T_pred = _euler_rollout_rope_accel(
                model, cond_encoder, moved, args.test_n_steps,
                device, grid_attrs, model_patch_size,
            )
            mse_list.append(F.mse_loss(T_pred, T_tgt).detach().cpu().item())
            iou_val = iou_melt_volumes(T_pred, T_tgt, _T_LIQUIDUS_NORM)
            iou_list.append(float(iou_val))
            for k, val in evaluate_physical_metrics(T_pred, T_tgt, _T_LIQUIDUS_NORM).items():
                phys_metrics.setdefault(k, []).append(val)

            if local_idx < 4:
                log_figure(run, val_grid_2x2(T_tgt[:1].cpu(), T_pred[:1].cpu(), epoch=1000 + local_idx),
                         f"test_sample_{local_idx:03d}.png", dpi=120)
            if len(gallery_samples) < 5:
                gallery_samples.append((T_tgt[:1].cpu(), T_pred[:1].cpu()))

    tracker.log_artifact(str(mapping_path))

    if mse_list:
        avg_mse = sum(mse_list) / len(mse_list)
        avg_iou = sum(iou_list) / len(iou_list)
        mlflow.log_metric("test_mse_rollout", avg_mse)
        mlflow.log_metric("test_iou_rollout", avg_iou)
        if phys_metrics:
            mlflow.log_metrics({k: sum(v) / len(v) for k, v in phys_metrics.items()})
        logger.info("Test — MSE (rollout): %.6f  IoU: %.4f", avg_mse, avg_iou)
        metrics_detail = {
            "samples": [
                {"idx": idx, "mse": mse_list[idx], "iou": iou_list[idx],
                 **{k: phys_metrics[k][idx] for k in phys_metrics}}
                for idx in range(len(mse_list))
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
    else:
        logger.warning("No test samples evaluated — skipping test metrics.")
    return gallery_samples


def _train_loop_accel(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    ckpt_dir: Path,
    run: Any,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
    accelerator: Accelerator,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> tuple[torch.Tensor | None, list[torch.Tensor], list[str], list[float], list[float], list[float], list[float], list[float]]:
    weighter = ReLoBRaLoWeighter(
        alpha=args.relobralo_alpha,
        beta=args.relobralo_beta,
        eps=args.relobralo_epsilon,
    )
    best_val_loss = float("inf")
    val_batch_fixed = next(iter(val_loader))
    pred_history: list[torch.Tensor] = []
    epoch_labels: list[str] = []
    T_gt_fixed: torch.Tensor | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []
    fm_losses: list[float] = []
    pde_losses: list[float] = []
    lambda_hist: list[float] = []

    for epoch in tqdm(range(args.epochs), desc="Epochs",
                      disable=not accelerator.is_main_process):
        if epoch == 0 and accelerator.is_main_process:
            with epoch0_profiler(len(train_loader), _RUN_NAME) as prof:
                avg_train, avg_fm, avg_phys = _run_train_epoch_accel(
                    model, cond_encoder, optimizer, train_loader, epoch,
                    args, grid_attrs, model_patch_size, ds_cfg, scheduler,
                    accelerator, weighter, profiler=prof,
                )
        else:
            avg_train, avg_fm, avg_phys = _run_train_epoch_accel(
                model, cond_encoder, optimizer, train_loader, epoch,
                args, grid_attrs, model_patch_size, ds_cfg, scheduler,
                accelerator, weighter,
            )

        if epoch % args.val_every != 0:
            continue

        avg_val = _run_val_epoch_accel(
            model, cond_encoder, val_loader, grid_attrs, model_patch_size, accelerator
        )

        if accelerator.is_main_process:
            mlflow.log_metric("val_loss", avg_val, step=epoch)
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            fm_losses.append(avg_fm)
            pde_losses.append(avg_phys)
            lambda_hist.append(weighter.lambda_phys)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                _save_best_checkpoint_accel(
                    model, cond_encoder, ckpt_dir, epoch, avg_val, args, accelerator
                )
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                logger.info("New best model at epoch %d: %.6f", epoch, best_val_loss)

            if epoch % 10 == 0:
                model.eval()
                cond_encoder.eval()
                try:
                    device = accelerator.device
                    single: dict[str, Any] = {
                        k: v[:1].to(device) if isinstance(v, Tensor) else v
                        for k, v in val_batch_fixed.items()
                    }
                    with torch.no_grad():
                        T_pred_snap = _euler_rollout_rope_accel(
                            model, cond_encoder, single, 25, device, grid_attrs, model_patch_size
                        )
                    if T_gt_fixed is None:
                        T_gt_fixed = single["T_target"].squeeze(1)[:1].cpu()
                    pred_history.append(T_pred_snap[:1].cpu())
                    epoch_labels.append(f"Ep {epoch}")
                    log_figure(run, val_grid_2x2(T_gt_fixed, T_pred_snap[:1].cpu(), epoch=epoch),
                             f"val_rope_epoch_{epoch:03d}.png", dpi=120)
                finally:
                    model.train()
                    cond_encoder.train()

        accelerator.wait_for_everyone()

    return T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses, fm_losses, pde_losses, lambda_hist


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args_accel()
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    torch.manual_seed(args.seed)
    if accelerator.is_main_process:
        logger.info(
            "Accelerate: %d process(es), mixed_precision=%s",
            accelerator.num_processes, args.mixed_precision,
        )

    grid_attrs = read_grid_attrs(args.h5)
    if accelerator.is_main_process:
        logger.info(
            "Grid: dx=%.3e m  dy=%.3e m  dz=%.3e m",
            grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"],
        )

    ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, _, _ = (
        _build_data_accel(args)
    )
    model, cond_encoder, optimizer, model_patch_size = _build_models_accel(args, ds_cfg)

    # OneCycleLR must be built before accelerator.prepare() but after loader is known
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    model, cond_encoder, optimizer, train_loader, val_loader, scheduler = (
        accelerator.prepare(model, cond_encoder, optimizer, train_loader, val_loader, scheduler)
    )

    if not accelerator.is_main_process:
        # Non-main processes participate in training; skip MLflow setup
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _train_loop_accel(
            model, cond_encoder, optimizer, train_loader, val_loader,
            args, ckpt_dir, None, grid_attrs, model_patch_size, ds_cfg,
            accelerator, scheduler,
        )  # return values intentionally ignored on non-main process
        return

    ckpt_dir, tracker = _setup_tracker_accel(args)

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={
            **vars(args),
            "model": "VelocityDiTRoPE",
            "embed_dim": _DEFAULT_EMBED_DIM,
        },
        tags={"architecture": "transformer_rope", "mode": "accelerate_relobralo"},
    ) as run:
        T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses, fm_losses, pde_losses, lambda_hist = (
            _train_loop_accel(
                model, cond_encoder, optimizer, train_loader, val_loader,
                args, ckpt_dir, run, grid_attrs, model_patch_size, ds_cfg,
                accelerator, scheduler,
            )
        )
        if train_losses and val_losses:
            fig = loss_panel(
                train_losses, val_losses,
                fm_losses=fm_losses if fm_losses else None,
                pde_losses=pde_losses if pde_losses else None,
                lambda_hist=lambda_hist if lambda_hist else None,
                title=f"{_RUN_NAME} — Loss Panel",
            )
            log_figure(run, fig, "loss_panel.png")

        if T_gt_fixed is not None and pred_history:
            log_figure(run, gallery_evolution(T_gt_fixed, pred_history, epoch_labels=epoch_labels,
                                            title=f"{_RUN_NAME} — Training Evolution", dpi=150),
                     "gallery_evolution.png")

        gallery_samples = _run_test_phase_accel(
            model, cond_encoder, test_loader, test_ds, full_ds, ckpt_dir,
            args, tracker, run, grid_attrs, model_patch_size, accelerator,
        )
        if gallery_samples:
            log_figure(run, gallery_test(gallery_samples[:5], title=f"{_RUN_NAME} — Test Samples", dpi=150),
                     "gallery_test.png")

        accelerator.wait_for_everyone()
        torch.save(
            {
                "model_state": accelerator.unwrap_model(model).state_dict(),
                "cond_encoder_state": accelerator.unwrap_model(cond_encoder).state_dict(),
                "ds_cfg": ds_cfg.model_dump(),
                "grid_attrs": grid_attrs,
            },
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
