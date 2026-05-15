"""train_fm_dit_physics.py — Physics-informed 3D-DiT with Physical 3D-RoPE.

Extends train_fm_dit.py with:
  - Physical 3D-RoPE: patch-center coordinates in mm as positional encodings.
  - Physics-informed FM loss: heat PDE residual computed fully in SI units.
  - Test traceability: test_sample_mapping.txt → MLflow artifact.

Physics residual reconstructs T_pred from the FM flow (T_pred = x_tau + (1-tau)*v_pred),
de-normalises to SI [K], and evaluates ρ·cp·dT/dt ≈ ∇·(k·∇T) + Q using a fixed
physical time step dt_s = 5 µs.  ReLoBRaLo adaptively balances the resulting
~[W/m³]²-scale residual against the normalised FM loss.
"""

from __future__ import annotations

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
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from experiments.train_fm_dit import (
    _T_LIQUIDUS_NORM,
    VelocityDiT,
    sinusoidal_time_embedding,
)
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
from neural_pbf.physics.ops import div_k_grad
from neural_pbf.schemas.tracking import TrackingConfig
from neural_pbf.tracking.factory import build_tracker

logger = logging.getLogger(__name__)

# embed_dim=288: 288 / 8 heads = 36 head_dim, divisible by 6 for per-head 3D-RoPE.
_DEFAULT_EMBED_DIM = 288
_RUN_NAME = "v3_rope_patches_4"


# ---------------------------------------------------------------------------
# Dataset with patch origin and sample identity
# ---------------------------------------------------------------------------


class PatchFMThermalDatasetWithOrigin(PatchFMThermalDataset):
    """Extends PatchFMThermalDataset to expose patch_origin for 3D-RoPE.

    Adds to the returned dict:
        patch_origin: (3,) long tensor [z_start=0, y_start, x_start] in voxel units.
        sample_key:   HDF5 group key string.
        h5_path:      Path to the HDF5 file string.
    """

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from torch.utils.data import Subset
        sample = self.base_ds[idx]
        if isinstance(self.base_ds, Subset):
            actual_ds = self.base_ds.dataset
            actual_idx = self.base_ds.indices[idx]
            path, sample_key = actual_ds._keys[actual_idx]
        else:
            path, sample_key = self.base_ds._keys[idx]

        with h5py.File(path, "r") as f:
            try:
                Lx_m: float = float(f.attrs["Lx_m"])
                Ly_m: float = float(f.attrs["Ly_m"])
                Nx: int = int(f.attrs["Nx"])
                Ny: int = int(f.attrs["Ny"])
            except KeyError as exc:
                raise ValueError(
                    f"HDF5 file '{path}' missing required root attribute: {exc}"
                ) from exc
            dx = Lx_m / (Nx - 1)
            dy = Ly_m / (Ny - 1)
            grp = f["samples"][sample_key]
            try:
                x0: float = float(grp.attrs["x"])
                y0: float = float(grp.attrs["y"])
            except KeyError as exc:
                raise ValueError(
                    f"Sample '{sample_key}' in '{path}' missing laser-position attribute: {exc}"
                ) from exc

        ix = int(round(x0 / dx))
        iy = int(round(y0 / dy))
        half = self.patch_size // 2
        x_start = max(0, min(Nx - self.patch_size, ix - half))
        y_start = max(0, min(Ny - self.patch_size, iy - half))

        ps = self.patch_size
        patched = {
            key: sample[key][:, :, :, y_start : y_start + ps, x_start : x_start + ps]
            for key in ["T_in", "T_target", "Q", "mask"]
        }
        # Z is never cropped — origin is always 0
        return {
            **{k: v for k, v in sample.items() if k not in patched},
            **patched,
            "patch_origin": torch.tensor([0, y_start, x_start], dtype=torch.long),
            "sample_key": sample_key,
            "h5_path": path,
        }


# ---------------------------------------------------------------------------
# Grid attribute helpers
# ---------------------------------------------------------------------------


def read_grid_attrs(h5_path: str) -> dict[str, float]:
    """Read spatial grid attributes from HDF5.

    Returns dict with keys: Lx_m, Ly_m, Lz_m, Nx, Ny, Nz, dx_m, dy_m, dz_m.
    Infers dz/Lz via isotropic-voxel assumption (dz = dx) when Lz_m is absent.
    """
    with h5py.File(h5_path, "r") as f:
        try:
            Lx_m = float(f.attrs["Lx_m"])
            Ly_m = float(f.attrs["Ly_m"])
            Nx = int(f.attrs["Nx"])
            Ny = int(f.attrs["Ny"])
        except KeyError as exc:
            raise ValueError(
                f"HDF5 file '{h5_path}' is missing required root attribute: {exc}. "
                "Expected: Lx_m, Ly_m, Nx, Ny."
            ) from exc

        dx_m = Lx_m / (Nx - 1)
        dy_m = Ly_m / (Ny - 1)

        if "Nz" in f.attrs:
            Nz = int(f.attrs["Nz"])
        else:
            # Infer Nz from the first sample tensor shape
            first_key = next(iter(f.get("samples", {})), None)
            if first_key is None:
                raise ValueError(
                    f"HDF5 file '{h5_path}' has no 'Nz' attribute and no samples to infer it from."
                )
            Nz = int(f["samples"][first_key]["T_in"].shape[-3])  # (1,1,Nz,Ny,Nx)

        if "Lz_m" in f.attrs:
            Lz_m = float(f.attrs["Lz_m"])
            dz_m = Lz_m / max(Nz - 1, 1)
        else:
            dz_m = dx_m  # isotropic assumption
            Lz_m = dz_m * (Nz - 1)

    return {
        "Lx_m": Lx_m,
        "Ly_m": Ly_m,
        "Lz_m": Lz_m,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dz_m": dz_m,
    }


# ---------------------------------------------------------------------------
# Physical 3D-RoPE
# ---------------------------------------------------------------------------


def apply_rope_3d(x: Tensor, coords_mm: Tensor, base: float = 10000.0) -> Tensor:
    """Apply Physical 3D Rotary Position Embedding to Q or K tensors.

    head_dim is split into three equal axis partitions (z / y / x).  Within
    each partition the standard RoPE rotation is applied using the physical
    patch-centre coordinate along that axis.

    Args:
        x:          (B, num_heads, N, head_dim).  head_dim must be divisible by 6.
        coords_mm:  (B, N, 3) physical patch-centre coords in mm,
                    ordered (z_mm, y_mm, x_mm).
        base:       RoPE base frequency.

    Returns:
        (B, num_heads, N, head_dim) with 3D-RoPE applied.
    """
    B, H, N, D = x.shape
    if D % 6 != 0:
        raise ValueError(f"head_dim={D} must be divisible by 6 for 3D-RoPE")
    d_axis = D // 3   # features allocated per spatial axis
    half = d_axis // 2

    freqs = 1.0 / (
        base
        ** (torch.arange(half, dtype=torch.float32, device=x.device) / max(half, 1))
    )  # (half,)

    chunks = x.chunk(3, dim=-1)  # 3 × (B, H, N, d_axis)
    rotated: list[Tensor] = []

    for axis_idx, chunk in enumerate(chunks):
        coord = coords_mm[..., axis_idx]  # (B, N)
        # Unsqueeze for num_heads broadcast: (B, 1, N, 1) × (half,) → (B, 1, N, half)
        angles = coord.unsqueeze(1).unsqueeze(-1) * freqs
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        x1, x2 = chunk.chunk(2, dim=-1)  # each (B, H, N, half)

        r1 = x1 * cos_a - x2 * sin_a
        r2 = x1 * sin_a + x2 * cos_a

        rotated.append(torch.cat([r1, r2], dim=-1))

    return torch.cat(rotated, dim=-1)


def patch_center_coords_mm(
    patch_origins: Tensor,
    model_patch_size: int,
    grid_attrs: dict[str, float],
    token_grid_shape: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    """Compute patch-centre physical coordinates in mm for all tokens in a batch.

    Args:
        patch_origins:     (B, 3) long — [z_start, y_start, x_start] in voxels.
        model_patch_size:  DiT patch size (voxels per side).
        grid_attrs:        Output of read_grid_attrs.
        token_grid_shape:  (Nz_t, Ny_t, Nx_t) — token count per axis.
        device:            Compute device.

    Returns:
        (B, N_tokens, 3) float32 in mm, ordered (z_mm, y_mm, x_mm).
    """
    Nz_t, Ny_t, Nx_t = token_grid_shape
    dz_m: float = grid_attrs["dz_m"]
    dy_m: float = grid_attrs["dy_m"]
    dx_m: float = grid_attrs["dx_m"]

    # Crucial Fix: Use discrete patch indices (0, 1, 2...) instead of physical mm
    # to prevent RoPE rotation angles from collapsing to ~0.0 at the micro-scale.
    pz = torch.arange(Nz_t, device=device, dtype=torch.float32)
    py = torch.arange(Ny_t, device=device, dtype=torch.float32)
    px = torch.arange(Nx_t, device=device, dtype=torch.float32)

    grid_z, grid_y, grid_x = torch.meshgrid(pz, py, px, indexing="ij")
    local_coords_idx = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    N_tokens = Nz_t * Ny_t * Nx_t
    local_coords_idx = local_coords_idx.reshape(N_tokens, 3)

    # Convert the global voxel origin to a discrete patch index
    global_origin_idx = patch_origins.to(device).float() / model_patch_size  # (B, 3)

    return local_coords_idx.unsqueeze(0) + global_origin_idx.unsqueeze(1)


# ---------------------------------------------------------------------------
# DiT block with per-head 3D-RoPE applied to Q and K
# ---------------------------------------------------------------------------


class DiTBlockRoPE(nn.Module):
    """DiT block with custom multi-head attention that applies 3D-RoPE to Q and K."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor, coords_mm: Tensor) -> Tensor:
        """
        Args:
            x:          (B, N, dim) token sequence.
            c:          (B, dim) condition vector (time + process params).
            coords_mm:  (B, N, 3) patch-centre coords in mm.

        Returns:
            (B, N, dim) modulated token sequence.
        """
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim

        mods = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
        shift1, scale1, gate1 = (v.unsqueeze(1) for v in (shift1, scale1, gate1))
        shift2, scale2, gate2 = (v.unsqueeze(1) for v in (shift2, scale2, gate2))

        h = self.norm1(x) * (1.0 + scale1) + shift1
        # Project and reshape to (B, H, N, hd)
        qkv = self.qkv(h).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = apply_rope_3d(q, coords_mm)
        k = apply_rope_3d(k, coords_mm)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, hd)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + gate1 * self.out_proj(attn_out)

        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# VelocityDiT with 3D-RoPE
# ---------------------------------------------------------------------------


class VelocityDiTRoPE(VelocityDiT):
    """VelocityDiT with Physical 3D-RoPE applied inside each attention block."""

    def __init__(self, **kwargs: Any) -> None:
        embed_dim = int(kwargs.get("embed_dim", _DEFAULT_EMBED_DIM))
        num_heads = int(kwargs.get("num_heads", 8))
        head_dim = embed_dim // num_heads
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 6 for 3D-RoPE")
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 6 for 3D-RoPE")
        super().__init__(**kwargs)
        # Replace parent's DiTBlock list with RoPE-aware blocks
        depth = len(self.blocks)
        self.blocks = nn.ModuleList(
            [DiTBlockRoPE(embed_dim, num_heads) for _ in range(depth)]
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        t: Tensor,
        cond: Tensor,
        coords_mm: Tensor,
    ) -> Tensor:
        """
        Args:
            x:          (B, in_channels, D, H, W) packed [x_tau, mask, Q].
            t:          (B,) flow time ∈ [0, 1].
            cond:       (B, cond_embed_dim) from ConditioningEncoder.
            coords_mm:  (B, N_tokens, 3) patch-centre coordinates in mm.

        Returns:
            (B, 1, D, H, W) predicted velocity.
        """
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_tokens, embed_dim)

        t_emb = sinusoidal_time_embedding(t, self._time_emb_dim)
        c = self.time_embed(t_emb) + self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c, coords_mm)

        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return self.final_layer(x)


# ---------------------------------------------------------------------------
# Physics residual helpers
# ---------------------------------------------------------------------------


def denorm_cond_batch(
    cond_normalized: Tensor,
    ds_cfg: FMDatasetConfig,
    key: str,
) -> Tensor:
    """De-normalise a conditioning key, returning per-sample values.

    Args:
        cond_normalized: (B, D_cond) z-score normalised conditioning tensor.
        ds_cfg:          Dataset config carrying normalisation statistics.
        key:             Conditioning key name (e.g. "rho", "cp", "k_s").

    Returns:
        (B, 1, 1, 1, 1) float32 tensor in physical units, broadcast-ready.
    """
    idx = list(ds_cfg.conditioning_keys).index(key)
    mean = ds_cfg.cond_means[key]
    std = ds_cfg.cond_stds[key]
    vals = cond_normalized[:, idx] * std + mean  # (B,)
    return vals.view(-1, 1, 1, 1, 1)


def physics_heat_residual(
    v_pred: Tensor,
    x_tau_detached: Tensor,
    tau: Tensor,
    T_in: Tensor,
    Q: Tensor,
    rho: Tensor,
    cp: Tensor,
    k: Tensor,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    ds_cfg: FMDatasetConfig,
    dt_s: float = 5e-6,
) -> Tensor:
    """Heat-equation PDE residual computed fully in SI units.

    Reconstructs the predicted clean temperature field T_pred from the current
    noisy state x_tau and the predicted flow velocity v_pred (OT-FM identity:
    T_pred = x_tau + (1-tau)*v_pred), then de-normalises to SI and evaluates:

        ρ·cp·(T_pred_SI - T_in_SI)/dt_s  ≈  ∇·(k·∇T_pred_SI) + Q_SI

    x_tau must be detached from the gradient graph before calling so that
    gradients flow only through v_pred (avoids expensive second-order backprop).

    Args:
        v_pred:           (B, 1, D, H, W) predicted velocity (FM pseudo-time), with grad.
        x_tau_detached:   (B, 1, D, H, W) interpolated noisy field, detached.
        tau:              (B,) flow time ∈ [0, 1].
        T_in:             (B, 1, D, H, W) normalised initial temperature field.
        Q:                (B, 1, D, H, W) normalised heat source.
        rho:              (B, 1, 1, 1, 1) per-sample density [kg/m³].
        cp:               (B, 1, 1, 1, 1) per-sample specific heat [J/(kg·K)].
        k:                (B, 1, 1, 1, 1) per-sample conductivity [W/(m·K)].
        dx_m, dy_m, dz_m: Grid spacings [m].
        ds_cfg:           Dataset config carrying T_ref, T_ambient, Q_ref.
        dt_s:             Physical time step for the finite-difference dT/dt [s].

    Returns:
        Scalar MSE residual tensor (SI units: [W/m³]²).
    """
    # Reconstruct predicted clean field from current noisy state + velocity
    T_pred_norm = x_tau_detached + (1.0 - tau.view(-1, 1, 1, 1, 1)) * v_pred

    # De-normalise to SI [K]
    T_pred_SI = T_pred_norm * ds_cfg.T_ref + ds_cfg.T_ambient
    T_in_SI = T_in * ds_cfg.T_ref + ds_cfg.T_ambient
    Q_SI = Q * ds_cfg.Q_ref

    # Physical time derivative approximated over the scan-step interval
    dT_dt_SI = (T_pred_SI - T_in_SI) / dt_s

    lhs = rho * cp * dT_dt_SI  # [W/m³]
    k_field = k.expand_as(T_pred_SI)
    rhs = div_k_grad(T_pred_SI, k_field, dx_m, dy_m, dz_m) + Q_SI  # [W/m³]

    # Scale both sides by Q_ref to prevent bfloat16/float32 overflow during squaring!
    # (Values are ~1e15 W/m³, squaring them exceeds 3.4e38 and causes 'inf')
    lhs_scaled = lhs / ds_cfg.Q_ref
    rhs_scaled = rhs / ds_cfg.Q_ref
    
    return F.mse_loss(lhs_scaled, rhs_scaled)


# ---------------------------------------------------------------------------
# Euler rollout with RoPE
# ---------------------------------------------------------------------------


def _euler_rollout_rope(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    batch: dict[str, Any],
    n_steps: int,
    device: torch.device,
    grid_attrs: dict[str, float],
    model_patch_size: int,
) -> Tensor:
    """n-step Euler integration for VelocityDiTRoPE.

    Returns:
        T_pred: (B, 1, D, H, W).
    """
    T_in = batch["T_in"].to(device).squeeze(1)   # (B, 1, D, H, W)
    mask = batch["mask"].to(device).squeeze(1)
    Q = batch["Q"].to(device).squeeze(1)
    cond = batch["conditioning"].to(device)
    patch_origins = batch["patch_origin"].to(device)   # (B, 3)

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
# Collation & visualisation helpers
# ---------------------------------------------------------------------------


def _collate_with_strings(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Default collate but pass string fields through as lists."""
    import torch.utils.data._utils.collate as _collate

    string_keys = {k for k, v in batch[0].items() if isinstance(v, str)}
    result: dict[str, Any] = _collate.default_collate(
        [{k: v for k, v in sample.items() if k not in string_keys} for sample in batch]
    )
    for k in string_keys:
        result[k] = [sample[k] for sample in batch]
    return result


def _slice_batch_first(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Return a single-sample sub-batch (index 0) moved to device."""
    result: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            result[k] = v[:1].to(device)
        elif isinstance(v, list):
            result[k] = v[:1]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _parse_args_physics() -> argparse.Namespace:
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
    parser.add_argument(
        "--physics_loss_weight",
        type=float,
        default=0.0,
        help="Weight for physics residual term. Set > 0 to enable PDE regularisation.",
    )
    return parser.parse_args()


def _build_data_physics(
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
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_with_strings,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=_collate_with_strings)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=_collate_with_strings,
    )
    logger.info("Dataset split — train: %d  val: %d  test: %d", n_train, n_val, n_test)
    return ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, n_val, n_test


def _build_models_physics(
    args: argparse.Namespace,
    ds_cfg: FMDatasetConfig,
    device: torch.device,
) -> tuple[VelocityDiTRoPE, ConditioningEncoder, torch.optim.Optimizer, int]:
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
    ).to(device)
    cond_encoder = ConditioningEncoder(cond_dim, cond_embed_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr
    )
    return model, cond_encoder, optimizer, model_patch_size


def _setup_tracker_physics(args: argparse.Namespace) -> tuple[Path, Any]:
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


def _train_batch_physics(
    batch: dict[str, Any],
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
) -> tuple[float, float, float]:
    T_tgt = batch["T_target"].to(device).squeeze(1)
    T_in = batch["T_in"].to(device).squeeze(1)
    mask = batch["mask"].to(device).squeeze(1)
    Q = batch["Q"].to(device).squeeze(1)
    cond = batch["conditioning"].to(device)
    patch_origins = batch["patch_origin"].to(device)
    B, _, D, H, W = T_tgt.shape
    coords_mm = patch_center_coords_mm(
        patch_origins, model_patch_size, grid_attrs,
        (D // model_patch_size, H // model_patch_size, W // model_patch_size), device,
    )
    cond_emb = cond_encoder(cond)
    noise = sample_noise(T_tgt)
    tau = torch.rand(B, device=device)
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
    loss = loss_fm + args.physics_loss_weight * phys
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_fm.item(), phys.item()


def _run_train_epoch_physics(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    profiler: Any = None,
) -> float:
    model.train()
    cond_encoder.train()
    loss_total = loss_fm_total = phys_total = 0.0
    t0 = time.perf_counter()
    for batch in tqdm(loader, desc=f"Train {epoch}", leave=False):
        lt, lf, ph = _train_batch_physics(
            batch, model, cond_encoder, optimizer, device,
            args, grid_attrs, model_patch_size, ds_cfg,
        )
        scheduler.step()
        loss_total += lt
        loss_fm_total += lf
        phys_total += ph
        if profiler is not None:
            profiler.step()
    n = max(len(loader), 1)
    avg = loss_total / n
    mlflow.log_metric("train_loss", avg, step=epoch)
    mlflow.log_metric("train_loss_fm", loss_fm_total / n, step=epoch)
    mlflow.log_metric("physics_residual", phys_total / n, step=epoch)
    log_step_timing(epoch, (time.perf_counter() - t0) / n)
    log_gpu_telemetry(epoch)
    return avg


def _run_val_epoch_physics(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    grid_attrs: dict[str, float],
    model_patch_size: int,
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
            patch_origins = batch["patch_origin"].to(device)
            B, _, D, H, W = T_tgt.shape
            coords_mm = patch_center_coords_mm(
                patch_origins, model_patch_size, grid_attrs,
                (D // model_patch_size, H // model_patch_size, W // model_patch_size), device,
            )
            cond_emb = cond_encoder(cond)
            noise = sample_noise(T_tgt)
            tau = torch.rand(B, device=device)
            x_tau = interpolate(noise, T_tgt, tau)
            v_pred = model(torch.cat([x_tau, mask, Q], dim=1), tau, cond_emb, coords_mm)
            total += fm_loss(v_pred, noise, T_tgt).item()
    return total / max(len(loader), 1)


def _save_best_checkpoint_physics(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    # weights_only=False required when loading: checkpoint contains args dict
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


def _infer_pred_rope(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    n_steps: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-sample inference returning (T_gt_cpu, T_pred_cpu) as (1,1,D,H,W)."""
    model.eval()
    cond_encoder.eval()
    single = _slice_batch_first(batch, device)
    with torch.no_grad():
        T_pred = _euler_rollout_rope(
            model, cond_encoder, single, n_steps, device, grid_attrs, model_patch_size
        )
    T_gt = single["T_target"].squeeze(1)[:1].cpu()
    return T_gt, T_pred[:1].cpu()


def _train_loop_physics(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_dir: Path,
    run: Any,
    grid_attrs: dict[str, float],
    model_patch_size: int,
    ds_cfg: FMDatasetConfig,
) -> tuple[torch.Tensor | None, list[torch.Tensor], list[str], list[float], list[float]]:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
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
                avg_train = _run_train_epoch_physics(
                    model, cond_encoder, optimizer, train_loader, device, epoch,
                    args, grid_attrs, model_patch_size, ds_cfg, scheduler,
                    profiler=prof,
                )
        else:
            avg_train = _run_train_epoch_physics(
                model, cond_encoder, optimizer, train_loader, device, epoch,
                args, grid_attrs, model_patch_size, ds_cfg, scheduler,
            )
        if epoch % args.val_every != 0:
            continue
        avg_val = _run_val_epoch_physics(
            model, cond_encoder, val_loader, device, grid_attrs, model_patch_size
        )
        mlflow.log_metric("val_loss", avg_val, step=epoch)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save_best_checkpoint_physics(model, cond_encoder, ckpt_dir, epoch, avg_val, args)
            mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
            logger.info("New best model at epoch %d: %.6f", epoch, best_val_loss)
        if epoch % 10 == 0:
            try:
                T_gt_fixed, T_pred_snap = _infer_pred_rope(
                    model, cond_encoder, val_batch_fixed, device, grid_attrs, model_patch_size
                )
                pred_history.append(T_pred_snap)
                epoch_labels.append(f"Ep {epoch}")
                log_figure(run, val_grid_2x2(T_gt_fixed, T_pred_snap, epoch=epoch),
                         f"val_rope_epoch_{epoch:03d}.png", dpi=120)
            finally:
                model.train()
                cond_encoder.train()

    return T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses


def _run_test_phase_physics(
    model: VelocityDiTRoPE,
    cond_encoder: nn.Module,
    test_loader: DataLoader,
    test_ds: Any,
    full_ds: FMThermalDataset,
    ckpt_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    tracker: Any,
    run: Any,
    grid_attrs: dict[str, float],
    model_patch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    logger.info("Starting test evaluation with best checkpoint …")
    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        logger.warning("No best checkpoint at %s; skipping test phase.", best_ckpt)
        return []
    # weights_only=False required: checkpoint includes args dict (not pure tensors)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    cond_encoder.load_state_dict(ckpt["cond_encoder_state"])
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
            T_tgt = batch["T_target"].to(device).squeeze(1)
            T_pred = _euler_rollout_rope(
                model, cond_encoder, batch, args.test_n_steps,
                device, grid_attrs, model_patch_size,
            )
            mse_list.append(F.mse_loss(T_pred, T_tgt).item())
            iou_list.append(iou_melt_volumes(T_pred, T_tgt, _T_LIQUIDUS_NORM))
            for k, val in evaluate_physical_metrics(T_pred, T_tgt, _T_LIQUIDUS_NORM).items():
                phys_metrics.setdefault(k, []).append(val)
            if local_idx < 4:
                log_figure(run, val_grid_2x2(T_tgt[:1].cpu(), T_pred[:1].cpu(), epoch=1000 + local_idx),
                         f"test_sample_{local_idx:03d}.png", dpi=120)
            if len(gallery_samples) < 5:
                gallery_samples.append((T_tgt[:1].cpu(), T_pred[:1].cpu()))
    tracker.log_artifact(str(mapping_path))
    _log_test_metrics_physics(mse_list, iou_list)
    if phys_metrics:
        mlflow.log_metrics({k: sum(v) / len(v) for k, v in phys_metrics.items()})
    if mse_list:
        metrics_detail = {
            "samples": [
                {"idx": idx, "mse": mse_list[idx], "iou": iou_list[idx],
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
    return gallery_samples


def _log_test_metrics_physics(
    mse_list: list[float], iou_list: list[float]
) -> None:
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


def main() -> None:
    args = _parse_args_physics()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    grid_attrs = read_grid_attrs(args.h5)
    logger.info(
        "Grid: dx=%.3e m  dy=%.3e m  dz=%.3e m",
        grid_attrs["dx_m"], grid_attrs["dy_m"], grid_attrs["dz_m"],
    )

    ds_cfg, full_ds, test_ds, train_loader, val_loader, test_loader, n_train, _, _ = (
        _build_data_physics(args)
    )
    model, cond_encoder, optimizer, model_patch_size = _build_models_physics(
        args, ds_cfg, device
    )
    ckpt_dir, tracker = _setup_tracker_physics(args)

    with tracker.start_run(
        run_name=_RUN_NAME,
        config={"model": "VelocityDiTRoPE", "embed_dim": _DEFAULT_EMBED_DIM,
                "epochs": args.epochs, "lr": args.lr,
                "physics_loss_weight": args.physics_loss_weight, "seed": args.seed},
        tags={"architecture": "transformer_rope", "mode": "patches_physics"},
    ) as run:
        T_gt_fixed, pred_history, epoch_labels, train_losses, val_losses = _train_loop_physics(
            model, cond_encoder, optimizer, train_loader, val_loader,
            device, args, ckpt_dir, run, grid_attrs, model_patch_size, ds_cfg,
        )
        if train_losses and val_losses:
            log_figure(run, loss_panel(train_losses, val_losses, title=f"{_RUN_NAME} — Loss Panel"),
                     "loss_panel.png")

        if T_gt_fixed is not None and pred_history:
            log_figure(run, gallery_evolution(T_gt_fixed, pred_history, epoch_labels=epoch_labels,
                                            title=f"{_RUN_NAME} — Training Evolution", dpi=150),
                     "gallery_evolution.png")

        gallery_samples = _run_test_phase_physics(
            model, cond_encoder, test_loader, test_ds, full_ds, ckpt_dir,
            device, args, tracker, run, grid_attrs, model_patch_size,
        )
        if gallery_samples:
            log_figure(run, gallery_test(gallery_samples[:5], title=f"{_RUN_NAME} — Test Samples", dpi=150),
                     "gallery_test.png")

        torch.save(
            {"model_state": model.state_dict(),
             "cond_encoder_state": cond_encoder.state_dict(),
             "ds_cfg": ds_cfg.model_dump(), "grid_attrs": grid_attrs},
            ckpt_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
