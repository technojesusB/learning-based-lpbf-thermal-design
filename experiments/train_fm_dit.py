"""train_fm_dit.py — 3D Diffusion Transformer (DiT) for LPBF thermal surrogate.

VelocityDiT uses patch-based 3D tokenization with Adaptive Layer Norm zero-init
modulation (adaLN-zero, Peebles & Xie 2022) conditioned on flow time and process
parameters.  Input channels: [x_tau, mask, Q].

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

from neural_pbf.eval.reporting.hardware import epoch0_profiler, log_gpu_telemetry, log_step_timing
from neural_pbf.eval.viz.logging import log_figure
from neural_pbf.eval.viz.losses import loss_panel
from neural_pbf.eval.viz.spatial import gallery_evolution, gallery_test
from neural_pbf.integrator.fm_stepper import euler_rollout
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiT
from neural_pbf.models.generative.fm.flow import fm_loss, interpolate, sample_noise
from neural_pbf.training import (
    TrainHistory,
    build_patch_data,
    load_checkpoint,
    run_test_phase,
    run_train_epoch,
    run_train_loop,
    run_val_epoch,
    save_checkpoint,
    setup_run,
)

logger = logging.getLogger(__name__)

_RUN_NAME = "v2_dit_patches_8"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mlflow_experiment", type=str, default="lpbf_surrogate_benchmark")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v2_dit_patches_8")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--test_n_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = build_patch_data(
        [args.h5], patch_size=64, batch_size=args.batch_size,
        seed=args.seed, use_origins=True,
    )
    ds_cfg = split.ds_cfg

    cond_embed_dim = 128
    model = VelocityDiT(
        patch_size=8, in_channels=3, embed_dim=256,
        depth=6, num_heads=8, cond_embed_dim=cond_embed_dim,
    ).to(device)
    cond_encoder = ConditioningEncoder(len(ds_cfg.conditioning_keys), cond_embed_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )

    tracker, ckpt_dir = setup_run(args)

    # --- batch-level training logic (DiT-specific: FM + TV regularisation) ---

    def _train_batch(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond_emb = cond_encoder(batch["conditioning"].to(device))
        noise = sample_noise(T_tgt)
        tau = torch.rand(T_tgt.shape[0], device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
        loss = fm_loss(v_pred, noise, T_tgt) + 0.001 * _tv_loss(v_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # --- per-epoch callbacks ---

    def _val_step(batch: dict[str, Any]) -> float:
        T_tgt = batch["T_target"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond_emb = cond_encoder(batch["conditioning"].to(device))
        noise = sample_noise(T_tgt)
        tau = torch.rand(T_tgt.shape[0], device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
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

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={
            "model": "VelocityDiT", "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "n_train": len(split.train_ds),
            "seed": args.seed, "patch_size": model.patch_size,
            "embed_dim": model.embed_dim, "depth": len(model.blocks),
            "tv_weight": 0.001, "pos_embed_scale": 0.02,
        },
        tags={"architecture": "transformer", "mode": "patches", "version": "v2_hotfix"},
    ) as run:

        def _checkpoint_fn(epoch: int, val_loss: float) -> None:
            save_checkpoint(model, cond_encoder, ckpt_dir, epoch, val_loss, args)
            mlflow.log_metric("best_val_loss", val_loss, step=epoch)

        def _snapshot_fn(epoch: int) -> tuple[Tensor, Tensor] | None:
            if epoch % 10 != 0:
                return None
            model.eval()
            cond_encoder.eval()
            try:
                with torch.no_grad():
                    T_tgt = val_batch_fixed["T_target"][:1].to(device).squeeze(1)
                    mask = val_batch_fixed["mask"][:1].to(device).squeeze(1)
                    Q = val_batch_fixed["Q"][:1].to(device).squeeze(1)
                    cond_emb = cond_encoder(val_batch_fixed["conditioning"][:1].to(device))
                    x = sample_noise(T_tgt)
                    dt = 1.0 / 25
                    for i in range(25):
                        tau = torch.full((1,), i * dt, device=device)
                        x = x + model(torch.cat([x, mask, Q], dim=1), tau, cond_emb) * dt
                T_gt_cpu = T_tgt.cpu()
                T_pred_cpu = x.cpu()
                log_figure(run, val_grid_2x2(T_gt_cpu, T_pred_cpu, epoch=epoch),
                           f"val_epoch_{epoch:03d}.png", dpi=120)
                return T_gt_cpu, T_pred_cpu
            finally:
                model.train()
                cond_encoder.train()

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
                    T_pred = euler_rollout(model, cond_encoder, batch, args.test_n_steps, device)
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
             "ds_cfg": ds_cfg.model_dump()},
            ckpt_dir / "latest.pt",
        )


# ---------------------------------------------------------------------------
# Script-specific helper (TV regularisation for DiT velocity field)
# ---------------------------------------------------------------------------


def _tv_loss(v: Tensor) -> Tensor:
    d = (v[:, :, 1:] - v[:, :, :-1]).pow(2).mean()
    h = (v[:, :, :, 1:] - v[:, :, :, :-1]).pow(2).mean()
    w = (v[:, :, :, :, 1:] - v[:, :, :, :, :-1]).pow(2).mean()
    return (d + h + w) / 3.0


if __name__ == "__main__":
    main()
