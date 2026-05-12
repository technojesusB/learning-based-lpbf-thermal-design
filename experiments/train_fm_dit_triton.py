"""train_fm_dit_triton.py — Hero Experiment: Physics-Informed FM + Custom Triton PDE Loss.

Extends train_fm_dit_accelerate.py with:
  - Custom Triton forward/backward kernels for the heat-PDE residual (PDEResidualLoss).
  - torch.profiler integration: Chrome trace logged to MLflow artifacts after epoch 0.

Run:
    uv run accelerate launch experiments/train_fm_dit_triton.py --h5 path/to/data.h5
    uv run python experiments/train_fm_dit_triton.py --test-kernel  # kernel unit-test only
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.physics.triton_pde_loss import PDEResidualLoss, _pde_residual_pytorch
from neural_pbf.tracking.factory import build_tracker
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.eval.metrics.geometry import iou_melt_volumes

from experiments.train_fm_dit import _T_LIQUIDUS_NORM
from experiments.train_fm_dit_rope import (
    PatchFMThermalDatasetWithOrigin,
    read_grid_attrs,
    patch_center_coords_mm,
    VelocityDiTRoPE,
    denorm_cond_batch,
    _collate_with_strings,
)
from experiments.train_fm_dit_accelerate import (
    _euler_rollout_rope_accel,
    _log_val_image_rope_accel,
    _run_val_epoch_accel,
    _save_best_checkpoint_accel,
    _run_test_phase_accel,
)
from neural_pbf.training.relobralo import ReLoBRaLoWeighter

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_DIM = 288
_MODEL_DEPTH = 6
_MODEL_HEADS = 8
_MODEL_PATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hero Experiment: Triton PDE-FM training")
    parser.add_argument("--h5", type=str, nargs="+", required=False, default=[],
                        help="Path(s) to HDF5 dataset(s) (required unless --test-kernel).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="fm_dit_triton")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dit_triton")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--test_n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dt_s", type=float, default=5e-6,
                        help="Physical time step for the PDE residual [s].")
    parser.add_argument("--physics_loss_weight", type=float, default=1.0,
                        help="Base weight for physics residual (ReLoBRaLo adapts it dynamically).")
    parser.add_argument("--relobralo_alpha", type=float, default=0.999)
    parser.add_argument("--relobralo_beta", type=float, default=0.9)
    parser.add_argument("--relobralo_epsilon", type=float, default=1e-8)
    parser.add_argument("--test-kernel", action="store_true",
                        help="Run Triton kernel unit test (compare against PyTorch float64) and exit.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Kernel verification (--test-kernel)
# ---------------------------------------------------------------------------


def _run_kernel_test() -> None:
    """Unit-test: Triton PDEResidualLoss vs PyTorch float64 reference.

    Checks both forward value agreement (< 1% relative error) and backward
    gradient agreement (< 2% relative error via finite difference).
    """
    print("=== Triton kernel unit test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available — testing CPU/PyTorch path only.")

    B, D, H, W = 1, 4, 4, 4
    torch.manual_seed(42)
    phys = dict(dx=1e-3, dy=1e-3, dz=1e-3, T_ref=1.0, T_ambient=0.0, Q_ref=1.0, dt_s=1.0)

    v64 = (torch.randn(B, 1, D, H, W, dtype=torch.float64) * 0.1).to(device)
    x64 = (torch.randn(B, 1, D, H, W, dtype=torch.float64) * 0.1).to(device)
    tau64 = (torch.rand(B, dtype=torch.float64) * 0.5).to(device)
    T_in64 = (torch.rand(B, 1, D, H, W, dtype=torch.float64) * 0.5).to(device)
    Q64 = torch.zeros(B, 1, D, H, W, dtype=torch.float64, device=device)
    rho64 = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64, device=device)
    cp64 = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64, device=device)
    k64 = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64, device=device)

    # --- forward value test ---
    ref_val = _pde_residual_pytorch(v64, x64.detach(), tau64, T_in64, Q64, rho64, cp64, k64, **phys).item()
    got_val = PDEResidualLoss.apply(
        v64.float(), x64.float().detach(), tau64.float(), T_in64.float(), Q64.float(),
        rho64.float(), cp64.float(), k64.float(),
        phys["dx"], phys["dy"], phys["dz"],
        phys["T_ref"], phys["T_ambient"], phys["Q_ref"], phys["dt_s"],
    ).item()
    fwd_rel = abs(got_val - ref_val) / (abs(ref_val) + 1e-30)
    fwd_ok = fwd_rel < 0.01 or abs(ref_val) < 1e-12
    print(f"Forward  — ref={ref_val:.6e}  got={got_val:.6e}  rel_err={fwd_rel:.4f}  {'PASS' if fwd_ok else 'FAIL'}")

    # --- gradient test: PDEResidualLoss.apply backward vs finite difference of reference ---
    # Uses PDEResidualLoss.apply (not _pde_residual_pytorch directly) so the custom
    # backward (including the Triton _pde_bwd_kernel on GPU) is actually exercised.
    eps = 1e-4
    v32 = v64.float()
    v_apply = v32.clone().requires_grad_(True)
    PDEResidualLoss.apply(
        v_apply, x64.float().detach(), tau64.float(), T_in64.float(), Q64.float(),
        rho64.float(), cp64.float(), k64.float(),
        phys["dx"], phys["dy"], phys["dz"],
        phys["T_ref"], phys["T_ambient"], phys["Q_ref"], phys["dt_s"],
    ).backward()
    grad_apply = v_apply.grad.clone()

    torch.manual_seed(7)
    direction = torch.randn_like(v64)
    direction = direction / direction.norm()
    L_plus = _pde_residual_pytorch(v64 + eps * direction, x64.detach(), tau64, T_in64, Q64, rho64, cp64, k64, **phys).item()
    L_minus = _pde_residual_pytorch(v64 - eps * direction, x64.detach(), tau64, T_in64, Q64, rho64, cp64, k64, **phys).item()
    fd_proj = (L_plus - L_minus) / (2 * eps)
    ad_proj = (grad_apply.double() * direction).sum().item()
    bwd_rel = abs(ad_proj - fd_proj) / (abs(fd_proj) + 1e-30)
    bwd_ok = bwd_rel < 0.02 or abs(fd_proj) < 1e-12
    print(f"Backward — fd={fd_proj:.6e}  apply_bwd={ad_proj:.6e}  rel_err={bwd_rel:.4f}  {'PASS' if bwd_ok else 'FAIL'}")

    if fwd_ok and bwd_ok:
        print("=== ALL PASS ===")
        sys.exit(0)
    else:
        print("=== FAIL ===")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _build_data(args: argparse.Namespace) -> tuple[
    FMDatasetConfig, FMThermalDataset, Any,
    DataLoader, DataLoader, DataLoader, int, int, int,
]:
    ds_cfg = FMDatasetConfig(h5_paths=args.h5, Q_ref=1.35e15)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds = PatchFMThermalDatasetWithOrigin(full_ds, patch_size=64)
    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        patch_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate_with_strings)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=_collate_with_strings)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=_collate_with_strings)
    logger.info("Dataset split — train: %d  val: %d  test: %d", n_train, n_val, n_test)
    return ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, n_val, n_test


# ---------------------------------------------------------------------------
# Model & optimizer
# ---------------------------------------------------------------------------


def _build_models(
    args: argparse.Namespace, ds_cfg: FMDatasetConfig
) -> tuple[VelocityDiTRoPE, ConditioningEncoder, torch.optim.Optimizer]:
    cond_dim = len(ds_cfg.conditioning_keys)
    cond_embed_dim = 128
    model = VelocityDiTRoPE(
        patch_size=_MODEL_PATCH_SIZE,
        in_channels=3,
        embed_dim=_DEFAULT_EMBED_DIM,
        depth=_MODEL_DEPTH,
        num_heads=_MODEL_HEADS,
        cond_embed_dim=cond_embed_dim,
    )
    cond_encoder = ConditioningEncoder(cond_dim, cond_embed_dim)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    return model, cond_encoder, optimizer


# ---------------------------------------------------------------------------
# MLflow setup + full traceability
# ---------------------------------------------------------------------------


def _setup_tracking(args: argparse.Namespace) -> tuple[Path, Any]:
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(TrackingConfig(
        enabled=True, backend="mlflow",
        experiment_name=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    ))
    return ckpt_dir, tracker


def _log_all_params(
    args: argparse.Namespace,
    n_train: int, n_val: int, n_test: int,
) -> None:
    mlflow.log_params(vars(args))
    mlflow.log_params({
        "data_n_total": n_train + n_val + n_test,
        "data_n_train": n_train,
        "data_n_val": n_val,
        "data_n_test": n_test,
        "data_seed": args.seed,
        "model_embed_dim": _DEFAULT_EMBED_DIM,
        "model_depth": _MODEL_DEPTH,
        "model_num_heads": _MODEL_HEADS,
        "model_patch_size": _MODEL_PATCH_SIZE,
    })


# ---------------------------------------------------------------------------
# Training batch
# ---------------------------------------------------------------------------


def _train_batch_triton(
    batch: dict[str, Any],
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    ds_cfg: FMDatasetConfig,
    weighter: ReLoBRaLoWeighter,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> tuple[float, float, float, float]:
    T_tgt = batch["T_target"].squeeze(1)
    T_in = batch["T_in"].squeeze(1)
    mask = batch["mask"].squeeze(1)
    Q = batch["Q"].squeeze(1)
    cond = batch["conditioning"]
    patch_origins = batch["patch_origin"]

    B, _, D, H, W = T_tgt.shape
    coords_mm = patch_center_coords_mm(
        patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
        (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE),
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

    phys = PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q,
        rho, cp_val, k_val,
        grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"],
        ds_cfg.T_ref, ds_cfg.T_ambient, ds_cfg.Q_ref, args.dt_s,
    )

    # All-reduce before extracting scalars so ReLoBRaLo sees the same values on every process.
    loss_fm_reduced = accelerator.gather(loss_fm.detach()).mean()
    phys_reduced = accelerator.gather(phys.detach()).mean()
    loss_fm_val = loss_fm_reduced.item()
    phys_val = phys_reduced.item()
    lambda_phys = weighter.step(loss_fm_val, phys_val)

    loss = loss_fm + lambda_phys * phys

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    return loss.detach().cpu().item(), loss_fm_val, phys_val, lambda_phys


# ---------------------------------------------------------------------------
# Training epoch (with optional profiler)
# ---------------------------------------------------------------------------


def _run_train_epoch(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epoch: int,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    ds_cfg: FMDatasetConfig,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    accelerator: Accelerator,
    weighter: ReLoBRaLoWeighter,
    profiler: Any = None,
) -> None:
    model.train()
    cond_encoder.train()
    loss_total = loss_fm_total = phys_total = 0.0
    n = 0
    for batch in tqdm(loader, desc=f"Train {epoch}", leave=False,
                      disable=not accelerator.is_main_process):
        lt, lf, ph, lam = _train_batch_triton(
            batch, model, cond_encoder, optimizer, accelerator,
            args, grid_attrs, ds_cfg, weighter, scheduler,
        )
        loss_total += lt
        loss_fm_total += lf
        phys_total += ph
        n += 1
        if profiler is not None:
            profiler.step()

    if accelerator.is_main_process and n > 0:
        mlflow.log_metric("train_loss", loss_total / n, step=epoch)
        mlflow.log_metric("train_loss_fm", loss_fm_total / n, step=epoch)
        mlflow.log_metric("physics_residual", phys_total / n, step=epoch)
        mlflow.log_metric("lambda_phys", weighter.lambda_phys, step=epoch)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _train_loop(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    ckpt_dir: Path,
    run: Any,
    grid_attrs: dict[str, float],
    ds_cfg: FMDatasetConfig,
    accelerator: Accelerator,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> None:
    weighter = ReLoBRaLoWeighter(
        alpha=args.relobralo_alpha,
        beta=args.relobralo_beta,
        eps=args.relobralo_epsilon,
    )
    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs), desc="Epochs",
                      disable=not accelerator.is_main_process):

        # Profile the first epoch (main process only).
        # The schedule (wait=1, warmup=1, active=3) requires >=5 batches to record anything.
        if epoch == 0 and accelerator.is_main_process:
            if len(train_loader) < 5:
                logger.warning(
                    "Profiler schedule requires >=5 batches per epoch but epoch 0 has %d. "
                    "The exported trace will be empty.",
                    len(train_loader),
                )
            profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                with_stack=False,
            )
            profiler.start()
            _run_train_epoch(
                model, cond_encoder, optimizer, train_loader, epoch,
                args, grid_attrs, ds_cfg, scheduler, accelerator, weighter,
                profiler=profiler,
            )
            profiler.stop()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                trace_path = tmp.name
            try:
                profiler.export_chrome_trace(trace_path)
                mlflow.log_artifact(trace_path, artifact_path="profiler")
            finally:
                os.remove(trace_path)
            logger.info("Profiler trace logged to MLflow (epoch 0).")
        else:
            _run_train_epoch(
                model, cond_encoder, optimizer, train_loader, epoch,
                args, grid_attrs, ds_cfg, scheduler, accelerator, weighter,
            )

        if epoch % args.val_every != 0:
            continue

        avg_val = _run_val_epoch_accel(
            model, cond_encoder, val_loader, grid_attrs, _MODEL_PATCH_SIZE, accelerator
        )

        if accelerator.is_main_process:
            mlflow.log_metric("val_loss", avg_val, step=epoch)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                _save_best_checkpoint_accel(
                    model, cond_encoder, ckpt_dir, epoch, avg_val, args, accelerator
                )
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                logger.info("New best model at epoch %d: %.6f", epoch, best_val_loss)

            if epoch % 10 == 0:
                _log_val_image_rope_accel(
                    model, run, cond_encoder, next(iter(val_loader)),
                    epoch, accelerator.device, grid_attrs, _MODEL_PATCH_SIZE, accelerator,
                )

        accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if args.test_kernel:
        _run_kernel_test()
        return

    if not args.h5:
        print("error: --h5 is required unless --test-kernel is specified", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)

    if accelerator.is_main_process:
        logger.info(
            "Accelerate: %d process(es), mixed_precision=%s",
            accelerator.num_processes, args.mixed_precision,
        )

    grid_attrs = read_grid_attrs(args.h5[0])
    if accelerator.is_main_process:
        logger.info(
            "Grid: dx=%.3e m  dy=%.3e m  dz=%.3e m",
            grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"],
        )

    ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, n_val, n_test = (
        _build_data(args)
    )
    model, cond_encoder, optimizer = _build_models(args, ds_cfg)

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
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _train_loop(
            model, cond_encoder, optimizer, train_loader, val_loader,
            args, ckpt_dir, None, grid_attrs, ds_cfg, accelerator, scheduler,
        )
        return

    ckpt_dir, tracker = _setup_tracking(args)

    with tracker.start_run(
        run_name=f"dit_triton_{datetime.datetime.now().strftime('%H%M%S')}",
        config={},
        tags={"architecture": "transformer_rope", "mode": "triton_pde_loss"},
    ) as run:
        _log_all_params(args, n_train, n_val, n_test)

        _train_loop(
            model, cond_encoder, optimizer, train_loader, val_loader,
            args, ckpt_dir, run, grid_attrs, ds_cfg, accelerator, scheduler,
        )
        _run_test_phase_accel(
            model, cond_encoder, test_loader, test_ds, full_ds, ckpt_dir,
            args, tracker, run, grid_attrs, _MODEL_PATCH_SIZE, accelerator,
        )
        # No barrier here: non-main processes have already returned from main().
        # The barrier inside _train_loop (accelerator.wait_for_everyone) is sufficient.
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
