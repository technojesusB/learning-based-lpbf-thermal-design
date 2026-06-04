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
from typing import Any

import mlflow
import torch
from accelerate import Accelerator
from torch.profiler import ProfilerActivity
from torch import Tensor
from torch.utils.data import DataLoader

from neural_pbf.data.patch_dataset import read_grid_attrs
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE, patch_center_coords_idx
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.physics.fm_physics import denorm_cond_batch
from neural_pbf.physics.triton_pde_loss import PDEResidualLoss, _pde_residual_pytorch
from neural_pbf.training import (
    TrainHistory,
    build_patch_data,
    load_checkpoint,
    log_val_image_rope,
    run_test_phase,
    run_train_epoch,
    run_train_loop,
    run_val_epoch,
    save_checkpoint,
    setup_run,
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
    parser.add_argument("--physics_loss_weight", type=float, default=1.0)
    parser.add_argument("--relobralo_alpha", type=float, default=0.999)
    parser.add_argument("--relobralo_beta", type=float, default=0.9)
    parser.add_argument("--relobralo_epsilon", type=float, default=1e-8)
    parser.add_argument("--test-kernel", action="store_true",
                        help="Run Triton kernel unit test and exit.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Kernel verification (--test-kernel)
# ---------------------------------------------------------------------------


def _run_kernel_test() -> None:
    """Unit-test: Triton PDEResidualLoss vs PyTorch float64 reference."""
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
# Training batch (Triton PDE loss variant — script-specific)
# ---------------------------------------------------------------------------


def _make_train_batch_fn(
    model: Any, cond_encoder: Any, optimizer: Any,
    accelerator: Accelerator, args: argparse.Namespace,
    grid_attrs: dict[str, float], ds_cfg: Any,
    weighter: ReLoBRaLoWeighter, scheduler: Any,
) -> Any:
    """Return the Triton-specific _train_batch closure."""

    def _train_batch_triton(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].squeeze(1)
        T_in = batch["T_in"].squeeze(1)
        mask = batch["mask"].squeeze(1)
        Q = batch["Q"].squeeze(1)
        cond = batch["conditioning"]
        patch_origins = batch["patch_origin"]
        B, _, D, H, W = T_tgt.shape
        coords_idx = patch_center_coords_idx(
            patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
            (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE),
            accelerator.device,
        )
        cond_emb = cond_encoder(cond)
        noise = sample_noise(T_tgt)
        tau = torch.rand(B, device=accelerator.device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_idx)
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
        loss_fm_reduced = accelerator.gather(loss_fm.detach()).mean()
        phys_reduced = accelerator.gather(phys.detach()).mean()
        lambda_phys = weighter.step(loss_fm_reduced.item(), phys_reduced.item())
        loss = loss_fm + lambda_phys * phys
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        return loss.detach().cpu().item()

    return _train_batch_triton


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
        logger.info("Accelerate: %d process(es), mixed_precision=%s",
                    accelerator.num_processes, args.mixed_precision)

    grid_attrs = read_grid_attrs(args.h5[0])
    if accelerator.is_main_process:
        logger.info("Grid: dx=%.3e m  dy=%.3e m  dz=%.3e m",
                    grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"])

    split = build_patch_data(
        args.h5, patch_size=64, batch_size=args.batch_size,
        seed=args.seed, use_origins=True,
    )
    ds_cfg = split.ds_cfg

    model = VelocityDiTRoPE(
        patch_size=_MODEL_PATCH_SIZE, in_channels=3, embed_dim=_DEFAULT_EMBED_DIM,
        depth=_MODEL_DEPTH, num_heads=_MODEL_HEADS, cond_embed_dim=128,
    )
    cond_encoder = ConditioningEncoder(len(ds_cfg.conditioning_keys), 128)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(split.train_loader), pct_start=0.1,
    )
    model, cond_encoder, optimizer, split.train_loader, split.val_loader, scheduler = (
        accelerator.prepare(model, cond_encoder, optimizer,
                            split.train_loader, split.val_loader, scheduler)
    )

    weighter = ReLoBRaLoWeighter(
        alpha=args.relobralo_alpha, beta=args.relobralo_beta, eps=args.relobralo_epsilon
    )
    _train_batch_triton = _make_train_batch_fn(
        model, cond_encoder, optimizer, accelerator, args, grid_attrs, ds_cfg, weighter, scheduler
    )

    # --- val step (shared with accelerate variant) ---

    def _val_step(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].squeeze(1)
        mask = batch["mask"].squeeze(1)
        Q = batch["Q"].squeeze(1)
        cond = batch["conditioning"]
        patch_origins = batch["patch_origin"]
        B, _, D, H, W = T_tgt.shape
        coords_idx = patch_center_coords_idx(
            patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
            (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE),
            accelerator.device,
        )
        cond_emb = cond_encoder(cond)
        noise = sample_noise(T_tgt)
        tau = torch.rand(B, device=accelerator.device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_idx)
        return fm_loss(v_pred, noise, T_tgt).detach().cpu().item()

    # --- per-epoch callbacks with epoch-0 profiler ---

    def _train_epoch_fn(epoch: int) -> float:
        model.train()
        cond_encoder.train()
        is_main = accelerator.is_main_process
        if epoch == 0 and is_main:
            if len(split.train_loader) < 5:
                logger.warning("Profiler schedule requires >=5 batches per epoch but has %d.",
                               len(split.train_loader))
            profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                with_stack=False,
            )
            profiler.start()
            avg = run_train_epoch(_train_batch_triton, split.train_loader, epoch,
                                  profiler=profiler, tqdm_disable=not is_main)
            profiler.stop()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                trace_path = tmp.name
            try:
                profiler.export_chrome_trace(trace_path)
                mlflow.log_artifact(trace_path, artifact_path="profiler")
            finally:
                os.remove(trace_path)
        else:
            avg = run_train_epoch(_train_batch_triton, split.train_loader, epoch,
                                  tqdm_disable=not is_main)
        if is_main:
            mlflow.log_metric("train_loss", avg, step=epoch)
            mlflow.log_metric("lambda_phys", weighter.lambda_phys, step=epoch)
        return avg

    def _val_epoch_fn(epoch: int) -> float:
        model.eval()
        cond_encoder.eval()
        avg_val = run_val_epoch(_val_step, split.val_loader)
        if accelerator.is_main_process:
            mlflow.log_metric("val_loss", avg_val, step=epoch)
        return avg_val

    val_loader_for_snap = split.val_loader

    def _inference_rollout(val_batch: dict[str, Any]) -> Tensor:
        single: dict[str, Any] = {
            k: v[:1] if isinstance(v, Tensor) else (v[:1] if isinstance(v, list) else v)
            for k, v in val_batch.items()
        }
        T_in = single["T_in"].squeeze(1)
        mask = single["mask"].squeeze(1)
        Q = single["Q"].squeeze(1)
        cond = single["conditioning"]
        patch_origins = single["patch_origin"]
        B, _, D, H, W = T_in.shape
        coords_idx = patch_center_coords_idx(
            patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
            (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE),
            accelerator.device,
        )
        cond_emb = cond_encoder(cond)
        x = sample_noise(T_in)
        dt = 1.0 / 25
        with torch.no_grad():
            for i in range(25):
                tau = torch.full((B,), i * dt, device=accelerator.device)
                x = x + model(torch.cat([x, mask, Q], dim=1), tau, cond_emb, coords_idx) * dt
        return x

    # Worker processes: run training loop without tracker
    if not accelerator.is_main_process:
        from pathlib import Path
        ckpt_dir_worker = Path(args.checkpoint_dir)
        ckpt_dir_worker.mkdir(parents=True, exist_ok=True)

        def _noop_checkpoint(epoch: int, val_loss: float) -> None:
            save_checkpoint(model, cond_encoder, ckpt_dir_worker, epoch, val_loss, args,
                            accelerator=accelerator)

        run_train_loop(
            _train_epoch_fn, _val_epoch_fn, _noop_checkpoint,
            snapshot_fn=lambda epoch: None,
            epochs=args.epochs, val_every=args.val_every,
            sync_fn=accelerator.wait_for_everyone,
            is_main=False,
            tqdm_disable=True,
        )
        return

    tracker, ckpt_dir = setup_run(args, timestamped=False)

    with tracker.start_run(
        run_name=f"dit_triton_{datetime.datetime.now().strftime('%H%M%S')}",
        config={},
        tags={"architecture": "transformer_rope", "mode": "triton_pde_loss"},
    ) as run:
        mlflow.log_params(vars(args))
        mlflow.log_params({
            "data_n_total": len(split.train_ds) + len(split.val_ds) + len(split.test_ds),
            "data_n_train": len(split.train_ds),
            "data_n_val": len(split.val_ds),
            "data_n_test": len(split.test_ds),
            "model_embed_dim": _DEFAULT_EMBED_DIM,
            "model_depth": _MODEL_DEPTH,
            "model_num_heads": _MODEL_HEADS,
            "model_patch_size": _MODEL_PATCH_SIZE,
        })

        def _checkpoint_fn(epoch: int, val_loss: float) -> None:
            save_checkpoint(model, cond_encoder, ckpt_dir, epoch, val_loss, args,
                            accelerator=accelerator)
            mlflow.log_metric("best_val_loss", val_loss, step=epoch)

        def _snapshot_fn(epoch: int) -> tuple[Tensor, Tensor] | None:
            if epoch % 10 != 0:
                return None
            return log_val_image_rope(
                model, cond_encoder, _inference_rollout,
                next(iter(val_loader_for_snap)), epoch, run,
                filename_prefix="val_rope_epoch",
            )

        run_train_loop(
            _train_epoch_fn, _val_epoch_fn, _checkpoint_fn, _snapshot_fn,
            epochs=args.epochs, val_every=args.val_every,
            sync_fn=accelerator.wait_for_everyone,
            is_main=True,
            tqdm_disable=False,
        )

        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            load_checkpoint(model, cond_encoder, best_ckpt, accelerator.device,
                            accelerator=accelerator)
            model.eval()
            cond_encoder.eval()

            def _test_rollout(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
                moved = {k: v.to(accelerator.device) if isinstance(v, Tensor) else v
                         for k, v in batch.items()}
                T_tgt = moved["T_target"].squeeze(1)
                T_in = moved["T_in"].squeeze(1)
                mask = moved["mask"].squeeze(1)
                Q = moved["Q"].squeeze(1)
                cond = moved["conditioning"]
                patch_origins = moved["patch_origin"]
                B, _, D, H, W = T_tgt.shape
                coords_idx = patch_center_coords_idx(
                    patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
                    (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE),
                    accelerator.device,
                )
                cond_emb = cond_encoder(cond)
                x = sample_noise(T_in)
                dt = 1.0 / args.test_n_steps
                with torch.no_grad():
                    for i in range(args.test_n_steps):
                        tau = torch.full((B,), i * dt, device=accelerator.device)
                        x = x + model(torch.cat([x, mask, Q], dim=1), tau, cond_emb, coords_idx) * dt
                return x.cpu(), T_tgt.cpu()

            run_test_phase(
                _test_rollout, split.test_loader, ckpt_dir, run,
                args.seed, tracker=tracker,
            )

        torch.save(
            {"model_state": accelerator.unwrap_model(model).state_dict(),
             "cond_encoder_state": accelerator.unwrap_model(cond_encoder).state_dict(),
             "ds_cfg": ds_cfg.model_dump(), "grid_attrs": grid_attrs},
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
