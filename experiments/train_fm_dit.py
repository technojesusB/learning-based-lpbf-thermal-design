"""train_fm_dit.py — 3D Diffusion Transformer (DiT) for LPBF thermal surrogate.

VelocityDiT uses patch-based 3D tokenization with Adaptive Layer Norm zero-init
modulation (adaLN-zero, Peebles & Xie 2022) conditioned on flow time and process
parameters.  Input channels: [x_tau, mask, Q].
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from experiments.train_fm_patches import PatchFMThermalDataset
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

logger = logging.getLogger(__name__)

# T_liquidus in normalised space: T_norm = (T_phys - 300) / 2000.
# TODO: calibrate to physical liquidus of SS316L (~1723 K → ~0.71).
_T_LIQUIDUS_NORM: float = 0.6
_RUN_NAME = "v2_dit_patches_8"


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------


def make_3d_sinusoidal_pos_embed(d: int, h: int, w: int, embed_dim: int) -> Tensor:
    """Fixed 3D sinusoidal position embeddings.

    Splits embed_dim into three equal per-axis parts (D/H/W), each rounded down
    to nearest even for clean sin/cos split.  Any remainder is zero-padded.

    Returns:
        (1, d*h*w, embed_dim) float32 tensor intended for buffer registration.
    """
    dim3 = (embed_dim // 3) & ~1  # per-axis even dim (e.g. 84 for embed_dim=256)
    extra = embed_dim - 3 * dim3   # remainder padded with zeros (e.g. 4)

    def _sincos(pos: Tensor, dim: int) -> Tensor:
        half = dim // 2
        freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
        angles = pos.float().reshape(-1, 1) * freq  # (N, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (N, dim)

    gd, gh, gw = torch.meshgrid(
        torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"
    )
    n = d * h * w
    parts = [
        _sincos(gd.reshape(-1), dim3),
        _sincos(gh.reshape(-1), dim3),
        _sincos(gw.reshape(-1), dim3),
    ]
    if extra > 0:
        parts.append(torch.zeros(n, extra))
    return torch.cat(parts, dim=-1).unsqueeze(0)  # (1, N, embed_dim)


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Fourier-feature time embedding.

    Args:
        t:   (B,) flow time ∈ [0, 1].
        dim: output dimensionality (must be even).

    Returns:
        (B, dim) float32 Fourier features.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    denom = max(half - 1, 1)
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / denom
    )
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# DiTBlock with adaLN-zero
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm zero-init (DiT paper §3.3)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        # Maps condition c → [shift1, scale1, gate1, shift2, scale2, gate2]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init makes the block an identity at initialisation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, dim) token sequence.
            c: (B, dim) condition vector (time + process params).

        Returns:
            (B, N, dim) modulated token sequence.
        """
        mods = self.adaLN_modulation(c)  # (B, 6*dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)

        # Unsqueeze for token-level broadcast: (B, 1, dim)
        shift1, scale1, gate1 = (v.unsqueeze(1) for v in (shift1, scale1, gate1))
        shift2, scale2, gate2 = (v.unsqueeze(1) for v in (shift2, scale2, gate2))

        h = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h)
        x = x + gate1 * attn_out

        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# VelocityDiT
# ---------------------------------------------------------------------------


class VelocityDiT(nn.Module):
    """3D patch-based Diffusion Transformer predicting the FM velocity field."""

    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_embed_dim: int = 128,
        time_emb_dim: int = 128,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self._time_emb_dim = time_emb_dim

        # Standard patch embedding: non-overlapping to reduce parameters and speed up learning.
        # Spatial coherence is handled by fixed 3D sinusoidal positional embeddings.
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        # Fixed 3D sinusoidal positional embeddings — non-trainable buffer.
        grid = input_size // patch_size  # 8 for 64^3 input with patch_size=8
        self.register_buffer(
            "pos_embed", make_3d_sinusoidal_pos_embed(grid, grid, grid, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.final_layer = nn.ConvTranspose3d(
            embed_dim, 1, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    (B, in_channels, D, H, W) packed [x_tau, mask, Q].
            t:    (B,) flow time ∈ [0, 1].
            cond: (B, cond_embed_dim) from external ConditioningEncoder.

        Returns:
            (B, 1, D, H, W) predicted velocity field.
        """
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_tokens, embed_dim)
        x = x + 0.02 * self.pos_embed # Scaled fixed 3D spatial information

        t_emb = sinusoidal_time_embedding(t, self._time_emb_dim)
        c = self.time_embed(t_emb) + self.cond_proj(cond)  # (B, embed_dim)

        for block in self.blocks:
            x = block(x, c)

        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return self.final_layer(x)


# ---------------------------------------------------------------------------
# Euler rollout helper
# ---------------------------------------------------------------------------


def _euler_rollout(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    batch: dict[str, Tensor],
    n_steps: int,
    device: torch.device,
) -> Tensor:
    """n-step Euler integration producing T_pred from noise.

    Args:
        model:        VelocityDiT in eval mode.
        cond_encoder: ConditioningEncoder in eval mode.
        batch:        Dataset batch dict.
        n_steps:      Number of Euler steps.
        device:       Compute device.

    Returns:
        T_pred: (B, 1, D, H, W).
    """
    T_in = batch["T_in"].to(device).squeeze(1)   # (B, 1, D, H, W)
    mask = batch["mask"].to(device).squeeze(1)
    Q = batch["Q"].to(device).squeeze(1)
    cond = batch["conditioning"].to(device)

    cond_emb = cond_encoder(cond)
    x = sample_noise(T_in)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        tau = torch.full((T_in.shape[0],), i * dt, device=device)
        v = model(torch.cat([x, mask, Q], dim=1), tau, cond_emb)
        x = x + v * dt

    return x


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def _infer_pred_dit(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    batch: dict,
    device: torch.device,
    n_steps: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler rollout for one DiT batch; returns (T_tgt, T_pred) on CPU.

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


def _tv_loss(v: Tensor) -> Tensor:
    """Mean L2 total variation of v (B, 1, D, H, W) across adjacent voxels."""
    d = (v[:, :, 1:] - v[:, :, :-1]).pow(2).mean()
    h = (v[:, :, :, 1:] - v[:, :, :, :-1]).pow(2).mean()
    w = (v[:, :, :, :, 1:] - v[:, :, :, :, :-1]).pow(2).mean()
    return (d + h + w) / 3.0


def _run_train_epoch(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    profiler: Any = None,
) -> float:
    model.train()
    cond_encoder.train()
    total = 0.0
    t0 = time.perf_counter()
    for batch in tqdm(loader, desc=f"Train {epoch}", leave=False):
        T_tgt = batch["T_target"].to(device).squeeze(1)
        mask = batch["mask"].to(device).squeeze(1)
        Q = batch["Q"].to(device).squeeze(1)
        cond_emb = cond_encoder(batch["conditioning"].to(device))
        noise = sample_noise(T_tgt)
        tau = torch.rand(T_tgt.shape[0], device=device)
        x_tau = interpolate(noise, T_tgt, tau)
        v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
        loss_fm = fm_loss(v_pred, noise, T_tgt)
        loss = loss_fm + 0.001 * _tv_loss(v_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        if profiler is not None:
            profiler.step()
    n = max(len(loader), 1)
    log_step_timing(epoch, (time.perf_counter() - t0) / n)
    log_gpu_telemetry(epoch)
    return total / n


def _run_val_epoch(
    model: VelocityDiT,
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
            cond_emb = cond_encoder(batch["conditioning"].to(device))
            noise = sample_noise(T_tgt)
            tau = torch.rand(T_tgt.shape[0], device=device)
            x_tau = interpolate(noise, T_tgt, tau)
            v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb)
            total += fm_loss(v_pred, noise, T_tgt).item()
    return total / max(len(loader), 1)


def _save_best_checkpoint(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "cond_encoder_state": cond_encoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        ckpt_dir / "best.pt",
    )


def _run_test_phase(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    loader: DataLoader,
    run: Any,
    device: torch.device,
    n_steps: int,
    ckpt_path: Path,
    seed: int,
    ckpt_dir: Path | None = None,
) -> tuple[list[float], list[float], list[tuple[torch.Tensor, torch.Tensor]]]:
    if not ckpt_path.exists():
        logger.warning("No best checkpoint at %s; skipping test phase.", ckpt_path)
        return [], [], []
    # weights_only=False required: checkpoint contains args dict (not pure tensors)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    cond_encoder.load_state_dict(ckpt["cond_encoder_state"])
    model.eval()
    cond_encoder.eval()
    torch.manual_seed(seed)
    mse_list: list[float] = []
    iou_list: list[float] = []
    phys_metrics: dict[str, list[float]] = {}
    gallery_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Test")):
            T_tgt = batch["T_target"].to(device).squeeze(1)
            T_pred = _euler_rollout(model, cond_encoder, batch, n_steps, device)
            mse_list.append(F.mse_loss(T_pred, T_tgt).item())
            iou_list.append(iou_melt_volumes(T_pred, T_tgt, _T_LIQUIDUS_NORM))
            for k, val in evaluate_physical_metrics(T_pred, T_tgt, _T_LIQUIDUS_NORM).items():
                phys_metrics.setdefault(k, []).append(val)
            if i < 4:
                log_figure(run, val_grid_2x2(T_tgt[:1].cpu(), T_pred[:1].cpu(), epoch=1000 + i),
                         f"test_sample_{i:03d}.png", dpi=120)
            if len(gallery_samples) < 5:
                gallery_samples.append((T_tgt[:1].cpu(), T_pred[:1].cpu()))
    if phys_metrics:
        mlflow.log_metrics({k: sum(v) / len(v) for k, v in phys_metrics.items()})
    if mse_list and ckpt_dir is not None:
        metrics_detail = {
            "samples": [
                {"idx": idx, "mse": mse_list[idx],
                 **{k: phys_metrics[k][idx] for k in phys_metrics}}
                for idx in range(len(mse_list))
            ],
            "summary": {
                "n_samples": len(mse_list),
                "mse_mean": sum(mse_list) / len(mse_list),
                "iou_mean": sum(iou_list) / len(iou_list) if iou_list else 0.0,
                **{k: sum(v) / len(v) for k, v in phys_metrics.items()},
            },
        }
        json_path = str(ckpt_dir / "test_metrics_detailed.json")
        with open(json_path, "w") as _jf:
            json.dump(metrics_detail, _jf, indent=2)
        run.log_artifact(json_path, artifact_path="eval")
    return mse_list, iou_list, gallery_samples


def _build_data(
    args: argparse.Namespace,
) -> tuple[FMDatasetConfig, DataLoader, DataLoader, DataLoader, int, int, int]:
    ds_cfg = FMDatasetConfig(h5_paths=[args.h5], Q_ref=1.35e15)
    patch_ds = PatchFMThermalDataset(FMThermalDataset(ds_cfg), patch_size=64)
    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        patch_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    logger.info("Dataset split — train: %d  val: %d  test: %d", n_train, n_val, n_test)
    return ds_cfg, train_loader, val_loader, test_loader, n_train, n_val, n_test


def _build_models_and_optimizer(
    args: argparse.Namespace,
    ds_cfg: FMDatasetConfig,
    device: torch.device,
) -> tuple[VelocityDiT, ConditioningEncoder, torch.optim.Optimizer]:
    cond_embed_dim = 128
    model = VelocityDiT(
        patch_size=8, in_channels=3, embed_dim=256,
        depth=6, num_heads=8, cond_embed_dim=cond_embed_dim,
    ).to(device)
    cond_encoder = ConditioningEncoder(
        len(ds_cfg.conditioning_keys), cond_embed_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    return model, cond_encoder, optimizer


def _setup_tracker(args: argparse.Namespace) -> tuple[Path, Any]:
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
    return ckpt_dir, tracker


def _train_loop(
    model: VelocityDiT,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_dir: Path,
    run: Any,
) -> tuple[torch.Tensor | None, list[torch.Tensor], list[str], list[float], list[float]]:
    best_val_loss = float("inf")
    val_batch_fixed = next(iter(val_loader))
    pred_history: list[torch.Tensor] = []
    epoch_labels: list[str] = []
    T_gt_fixed: torch.Tensor | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        if epoch == 0:
            with epoch0_profiler(len(train_loader), _RUN_NAME) as prof:
                avg_train = _run_train_epoch(
                    model, cond_encoder, optimizer, train_loader, device, epoch,
                    profiler=prof,
                )
        else:
            avg_train = _run_train_epoch(
                model, cond_encoder, optimizer, train_loader, device, epoch
            )
        mlflow.log_metric("train_loss", avg_train, step=epoch)
        if epoch % args.val_every != 0:
            continue
        avg_val = _run_val_epoch(model, cond_encoder, val_loader, device)
        mlflow.log_metric("val_loss", avg_val, step=epoch)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save_best_checkpoint(model, cond_encoder, ckpt_dir, epoch, avg_val, args)
            mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            logger.info("New best model at epoch %d: %.6f", epoch, best_val_loss)
        if epoch % 10 == 0:
            try:
                T_gt_fixed, T_pred_snap = _infer_pred_dit(model, cond_encoder, val_batch_fixed, device)
                pred_history.append(T_pred_snap)
                epoch_labels.append(f"Ep {epoch}")
                log_figure(run, val_grid_2x2(T_gt_fixed, T_pred_snap, epoch=epoch),
                         f"val_epoch_{epoch:03d}.png", dpi=120)
            finally:
                model.train()
                cond_encoder.train()

    return T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses


def _log_test_metrics(mse_list: list[float], iou_list: list[float]) -> None:
    if not mse_list:
        logger.warning("No test samples evaluated — skipping test metrics.")
        return
    avg_mse = sum(mse_list) / len(mse_list)
    avg_iou = sum(iou_list) / len(iou_list)
    mlflow.log_metric("test_mse_rollout", avg_mse)
    mlflow.log_metric("test_iou_rollout", avg_iou)
    logger.info("Test — MSE (rollout): %.6f  IoU: %.4f", avg_mse, avg_iou)


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

    ds_cfg, train_loader, val_loader, test_loader, n_train, n_val, n_test = _build_data(args)
    model, cond_encoder, optimizer = _build_models_and_optimizer(args, ds_cfg, device)
    ckpt_dir, tracker = _setup_tracker(args)

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={
            "model": "VelocityDiT", 
            "epochs": args.epochs, 
            "lr": args.lr,
            "batch_size": args.batch_size, 
            "n_train": n_train, 
            "seed": args.seed,
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "depth": len(model.blocks),
            "tv_weight": 0.001,
            "pos_embed_scale": 0.02
        },
        tags={"architecture": "transformer", "mode": "patches", "version": "v2_hotfix"},
    ) as run:
        T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses = _train_loop(
            model, cond_encoder, optimizer, train_loader, val_loader,
            device, args, ckpt_dir, run,
        )
        if train_losses and val_losses:
            log_figure(run, loss_panel(train_losses, val_losses, title=f"{_RUN_NAME} — Loss Panel"),
                     "loss_panel.png")

        if T_gt_fixed is not None and pred_history:
            log_figure(run, gallery_evolution(T_gt_fixed, pred_history, epoch_labels=epoch_labels,
                                            title=f"{_RUN_NAME} — Training Evolution", dpi=150),
                     "gallery_evolution.png")

        mse_list, iou_list, gallery_samples = _run_test_phase(
            model, cond_encoder, test_loader, run, device,
            args.test_n_steps, ckpt_dir / "best.pt", args.seed,
            ckpt_dir=ckpt_dir,
        )
        _log_test_metrics(mse_list, iou_list)
        if gallery_samples:
            log_figure(run, gallery_test(gallery_samples[:5], title=f"{_RUN_NAME} — Test Samples", dpi=150),
                     "gallery_test.png")

        torch.save(
            {"model_state": model.state_dict(),
             "cond_encoder_state": cond_encoder.state_dict(),
             "ds_cfg": ds_cfg.model_dump()},
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
