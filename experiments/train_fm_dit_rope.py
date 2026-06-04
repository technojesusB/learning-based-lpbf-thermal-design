"""train_fm_dit_physics.py — Physics-informed 3D-DiT with Physical 3D-RoPE.

Extends the base DiT with:
  - Physical 3D-RoPE: patch-center coordinates as positional encodings.
  - Physics-informed FM loss: heat PDE residual in SI units.
  - Test traceability: test_sample_mapping.txt → MLflow artifact.

All model/dataset/physics classes live in neural_pbf.*; this script only
contains main(), argument parsing, and script-specific training batch logic.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from neural_pbf.data.patch_dataset import read_grid_attrs
from neural_pbf.eval.reporting.hardware import epoch0_profiler, log_gpu_telemetry, log_step_timing
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.losses import loss_panel
from neural_pbf.eval.viz.spatial import gallery_evolution, gallery_test
from neural_pbf.integrator.fm_stepper import euler_rollout_rope
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE, patch_center_coords_idx
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.physics.fm_physics import denorm_cond_batch, physics_heat_residual
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

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_DIM = 288
_MODEL_PATCH_SIZE = 4
_RUN_NAME = "v3_rope_patches_4"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="lpbf_surrogate_benchmark")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v3_rope_patches_4")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--test_n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics_loss_weight", type=float, default=0.0,
                        help="Weight for physics residual term. Set > 0 to enable PDE regularisation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    grid_attrs = read_grid_attrs(args.h5)
    logger.info("Grid: dx=%.3e m  dy=%.3e m  dz=%.3e m",
                grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"])

    split = build_patch_data(
        [args.h5], patch_size=64, batch_size=args.batch_size,
        seed=args.seed, use_origins=True,
    )
    ds_cfg = split.ds_cfg

    model = VelocityDiTRoPE(
        patch_size=_MODEL_PATCH_SIZE, in_channels=3, embed_dim=_DEFAULT_EMBED_DIM,
        depth=6, num_heads=8, cond_embed_dim=128,
    ).to(device)
    cond_encoder = ConditioningEncoder(len(ds_cfg.conditioning_keys), 128).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(split.train_loader), pct_start=0.1,
    )

    tracker, ckpt_dir = setup_run(args)

    # --- batch-level training logic (RoPE + static-lambda physics loss) ---

    def _train_batch(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].to(device).squeeze(1)
        T_in = batch["T_in"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond = batch["conditioning"].to(device)
        patch_origins = batch["patch_origin"].to(device)
        B, _, D, H, W = T_tgt.shape
        coords_idx = patch_center_coords_idx(
            patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
            (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE), device,
        )
        cond_emb = cond_encoder(cond)
        noise = sample_noise(T_tgt)
        tau = torch.rand(B, device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_idx)
        loss_fm = fm_loss(v_pred, noise, T_tgt)
        rho = denorm_cond_batch(cond, ds_cfg, "rho")
        cp_val = denorm_cond_batch(cond, ds_cfg, "cp")
        k_val = denorm_cond_batch(cond, ds_cfg, "k_s")
        phys = physics_heat_residual(
            v_pred, x_tau.detach(), tau, T_in, Q, rho, cp_val, k_val,
            grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"], ds_cfg,
        )
        loss = loss_fm + args.physics_loss_weight * phys
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.item()

    # --- per-epoch callbacks ---

    def _val_step(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond = batch["conditioning"].to(device)
        patch_origins = batch["patch_origin"].to(device)
        B, _, D, H, W = T_tgt.shape
        coords_idx = patch_center_coords_idx(
            patch_origins, _MODEL_PATCH_SIZE, grid_attrs,
            (D // _MODEL_PATCH_SIZE, H // _MODEL_PATCH_SIZE, W // _MODEL_PATCH_SIZE), device,
        )
        cond_emb = cond_encoder(cond)
        noise = sample_noise(T_tgt)
        tau = torch.rand(B, device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_idx)
        return fm_loss(v_pred, noise, T_tgt).item()

    def _train_epoch_fn(epoch: int) -> float:
        model.train()
        cond_encoder.train()
        t0 = time.perf_counter()
        if epoch == 0:
            with epoch0_profiler(len(split.train_loader), _RUN_NAME) as prof:
                avg = run_train_epoch(_train_batch, split.train_loader, epoch, profiler=prof)
        else:
            avg = run_train_epoch(_train_batch, split.train_loader, epoch)
        mlflow.log_metric("train_loss", avg, step=epoch)
        log_step_timing(epoch, (time.perf_counter() - t0) / max(len(split.train_loader), 1))
        log_gpu_telemetry(epoch)
        return avg

    def _val_epoch_fn(epoch: int) -> float:
        model.eval()
        cond_encoder.eval()
        avg_val = run_val_epoch(_val_step, split.val_loader)
        mlflow.log_metric("val_loss", avg_val, step=epoch)
        return avg_val

    val_batch_fixed = next(iter(split.val_loader))

    def _inference_rollout(val_batch: dict[str, Any]) -> Tensor:
        single = {
            k: v[:1].to(device) if isinstance(v, Tensor) else (v[:1] if isinstance(v, list) else v)
            for k, v in val_batch.items()
        }
        return euler_rollout_rope(model, cond_encoder, single, 25, device, grid_attrs, _MODEL_PATCH_SIZE)

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={"model": "VelocityDiTRoPE", "embed_dim": _DEFAULT_EMBED_DIM,
                "epochs": args.epochs, "lr": args.lr,
                "physics_loss_weight": args.physics_loss_weight, "seed": args.seed},
        tags={"architecture": "transformer_rope", "mode": "patches_physics"},
    ) as run:

        def _checkpoint_fn(epoch: int, val_loss: float) -> None:
            save_checkpoint(model, cond_encoder, ckpt_dir, epoch, val_loss, args)
            mlflow.log_metric("best_val_loss", val_loss, step=epoch)

        def _snapshot_fn(epoch: int) -> tuple[Tensor, Tensor] | None:
            if epoch % 10 != 0:
                return None
            return log_val_image_rope(
                model, cond_encoder, _inference_rollout,
                val_batch_fixed, epoch, run,
                filename_prefix="val_rope_epoch",
            )

        history: TrainHistory = run_train_loop(
            _train_epoch_fn, _val_epoch_fn, _checkpoint_fn, _snapshot_fn,
            epochs=args.epochs, val_every=args.val_every,
        )

        if history.train_losses and history.val_losses:
            log_figure(run, loss_panel(history.train_losses, history.val_losses,
                                       title=f"{_RUN_NAME} — Loss Panel"), "loss_panel.png")
        if history.T_gt_fixed is not None and history.pred_history:
            log_figure(run, gallery_evolution(
                history.T_gt_fixed, history.pred_history,
                epoch_labels=history.epoch_labels,
                title=f"{_RUN_NAME} — Training Evolution", dpi=150,
            ), "gallery_evolution.png")

        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            load_checkpoint(model, cond_encoder, best_ckpt, device)
            model.eval()
            cond_encoder.eval()

            def _test_rollout(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
                with torch.no_grad():
                    T_tgt = batch["T_target"].to(device).squeeze(1)
                    T_pred = euler_rollout_rope(
                        model, cond_encoder, batch,
                        args.test_n_steps, device, grid_attrs, _MODEL_PATCH_SIZE,
                    )
                return T_pred.cpu(), T_tgt.cpu()

            gallery_samples = run_test_phase(
                _test_rollout, split.test_loader, ckpt_dir, run,
                args.seed, tracker=tracker,
            )
            if gallery_samples:
                log_figure(run, gallery_test(gallery_samples[:5],
                                             title=f"{_RUN_NAME} — Test Samples", dpi=150),
                           "gallery_test.png")

        torch.save(
            {"model_state": model.state_dict(),
             "cond_encoder_state": cond_encoder.state_dict(),
             "ds_cfg": ds_cfg.model_dump(),
             "grid_attrs": grid_attrs},
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
