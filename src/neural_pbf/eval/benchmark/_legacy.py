# ruff: noqa: E501
"""Physical fidelity benchmark and system comparison for LPBF thermal surrogates.

Public API:
    run_physical_fidelity_benchmark -- sweep models over a dataset, produce dashboard
    run_system_comparison           -- compare training runs by duration/GPU stats

Private helpers:
    _run_model_benchmark  -- per-model rollout loop (mockable in tests)
    _rollout_sample       -- Euler integration for a single batch/model type
    _model_palette        -- fixed color palette keyed by model name
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from mlflow.client import MlflowClient
from tqdm import tqdm

from neural_pbf.data.patch_dataset import PatchFMThermalDataset
from neural_pbf.eval.metrics.spectral import (
    compute_radial_psd,
    plot_spectral_comparison,
)
from neural_pbf.schemas.viz import THEME

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants & Normalisation (matches FMConfig priors)
# ---------------------------------------------------------------------------
_T_LIQUIDUS = 1600.0
_T_REF = 2000.0
_T_AMBIENT = 300.0


def _model_palette(names: list[str]) -> dict[str, str]:
    """Return a dict mapping each name to a hex color using THEME.physical.model_palette."""
    palette = {}
    fallback_colors = ["#e74c3c", "#3498db", "#f1c40f", "#00FFC8", "#95a5a6", "#9b59b6"]
    for i, name in enumerate(names):
        name_str = str(name)
        if name_str in THEME.physical.model_palette:
            palette[name_str] = THEME.physical.model_palette[name_str]
        else:
            palette[name_str] = fallback_colors[i % len(fallback_colors)]
    return palette


def _rollout_sample(
    model: nn.Module,
    cond_enc: nn.Module,
    batch: dict[str, Any],
    model_type: str,
    device: torch.device,
    n_steps: int = 25,
    grid_attrs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run Euler integration to produce T_pred for one batch.

    Args:
        model:       Velocity/DiT model.
        cond_enc:    Conditioning encoder.
        batch:       Dict with tensors already on *device*.
        model_type:  One of "net", "dit", "rope", "triton".
        device:      Compute device.
        n_steps:     Number of Euler steps.
        grid_attrs:  Grid attributes (required for rope/triton).

    Returns:
        T_pred tensor with shape matching batch["T_target"].squeeze(1).
    """
    with torch.no_grad():
        if model_type in ("rope", "triton"):
            from neural_pbf.integrator.fm_stepper import euler_rollout_rope

            ps = getattr(model, "patch_size", 4)
            if grid_attrs is None:
                grid_attrs = {"dx_m": 1.5e-5, "dy_m": 1.5e-5, "dz_m": 1.5e-5}
            return euler_rollout_rope(model, cond_enc, batch, n_steps, device, grid_attrs, ps)

        # Legacy path for non-rope models
        T_in = batch["T_in"]
        if T_in.ndim > 5:
            T_in = T_in.squeeze(2)
        mask = batch["mask"]
        if mask.ndim > 5:
            mask = mask.squeeze(2)
        Q = batch["Q"]
        if Q.ndim > 5:
            Q = Q.squeeze(2)

        cond = batch["conditioning"]
        cond_emb = cond_enc(cond)

        B = T_in.shape[0]
        x = torch.randn_like(T_in)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=device)
            inp = torch.cat([x, mask, Q], dim=1)
            v = model(inp, tau, cond_emb)
            x = x + v * dt

    return x


def _run_model_benchmark(
    name: str,
    ckpt_path: str,
    model_type: str,
    ds_cfg: Any,
    device: torch.device,
    test_indices: list[int] | None = None,
    collect_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    """Evaluate one model checkpoint and return a list of per-sample metric dicts.

    Loads the checkpoint, reconstructs the model, runs Euler-integration rollout on
    the last 10% of the dataset (test split), and computes IoU / depth / T_max_error
    / hotspot_offset metrics per sample.

    Args:
        name:       Display name for this model.
        ckpt_path:  Path to the ``best.pt`` checkpoint file.
        model_type: Architecture key — ``"net"``, ``"dit"``, or ``"rope"``.
        ds_cfg:     Dataset configuration (``FMDatasetConfig``).
        device:     Inference device.

    Returns:
        List of dicts with keys:
            Model, Sample, IoU, Depth_GT, Depth_Pred, T_max_Error, Offset_vox.
        Returns an empty list on any unrecoverable error (checkpoint missing, unknown
        model_type, dataset load failure).
    """
    from neural_pbf.data.fm_dataset import FMThermalDataset
    from neural_pbf.eval.metrics.geometry import (
        hotspot_offset_vox,
        iou_melt_volumes,
        melt_pool_extent,
    )
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
    from neural_pbf.models.generative.fm.config import FMConfig
    from neural_pbf.models.generative.fm.velocity_net import VelocityNet

    _T_REF = 2000.0

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        logger.error("Cannot load checkpoint %s: %s", ckpt_path, exc)
        return [], {}

    # Robust model loading: auto-detect patch_size via try/except (mirrors load_model_adaptive)
    model = None
    cond_enc = None
    model_patch_size = 4
    try:
        if model_type == "net":
            fm_cfg = FMConfig(**ckpt["fm_cfg"])
            model = VelocityNet(fm_cfg).to(device)
            cond_enc = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim).to(device)
        elif model_type == "dit":
            from neural_pbf.models.generative.fm.dit import VelocityDiT

            for p_size in [4, 8]:
                try:
                    model = VelocityDiT(
                        depth=6,
                        embed_dim=256,
                        num_heads=8,
                        patch_size=p_size,
                        input_size=64,
                        in_channels=3,
                        cond_embed_dim=128,
                    ).to(device)
                    model.load_state_dict(ckpt["model_state"])
                    model_patch_size = p_size
                    break
                except Exception:
                    continue
            cond_enc = ConditioningEncoder(12, 128).to(device)
        elif model_type in ("rope", "triton"):
            from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE

            for p_size in [4, 8]:
                try:
                    model = VelocityDiTRoPE(
                        depth=6,
                        embed_dim=288,
                        num_heads=8,
                        patch_size=p_size,
                        input_size=64,
                        in_channels=3,
                        cond_embed_dim=128,
                    ).to(device)
                    model.load_state_dict(ckpt["model_state"])
                    model_patch_size = p_size
                    break
                except Exception:
                    continue
            cond_enc = ConditioningEncoder(12, 128).to(device)
        else:
            logger.error("Unknown model_type %r for model %s", model_type, name)
            return [], {}
        if model is None:
            raise ValueError(f"Could not instantiate model {name} with any patch_size")
        if model_type == "net":
            model.load_state_dict(ckpt["model_state"])
        cond_enc.load_state_dict(ckpt["cond_encoder_state"])
        model.eval()
        cond_enc.eval()
        logger.info("Loaded %s (type=%s, patch_size=%d)", name, model_type, model_patch_size)
    except Exception as exc:
        logger.error("Failed to build/load model %s (%s): %s", name, model_type, exc)
        return [], {}

    # Use PatchFMThermalDatasetWithOrigin for rope/triton — required for patch_origin in batch
    try:
        base_ds = FMThermalDataset(ds_cfg)
        if model_type in ("rope", "triton"):
            from neural_pbf.data.patch_dataset import PatchFMThermalDatasetWithOrigin

            ds = PatchFMThermalDatasetWithOrigin(base_ds, patch_size=64)
            logger.info("Using PatchFMThermalDatasetWithOrigin for %s", name)
        else:
            # Patch-based evaluation to match training (64x64x64)
            ds = PatchFMThermalDataset(base_ds, patch_size=64)
    except Exception as exc:
        logger.error("Failed to load dataset for %s: %s", name, exc)
        return [], {}

    n = len(ds)
    # Default to 90/10 split if no specific indices provided
    if test_indices is None:
        test_indices = list(range(int(n * 0.9), n))

    rows: list[dict[str, Any]] = []
    collected_vols: dict[int, torch.Tensor] = {}
    # Reset peak memory counter for accurate per-model measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i in tqdm(test_indices, desc=f"Benchmark {name}", leave=False):
        # Live memory tracking every 5 samples
        if device.type == "cuda" and i % 5 == 0:
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            res = torch.cuda.memory_reserved(device) / 1024**2
            logger.info(
                "  [Mem Check Sample %d] Allocated: %.0f MB | Reserved: %.0f MB",
                i,
                alloc,
                res,
            )

        try:
            sample = ds[i]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            T_pred = _rollout_sample(
                model,
                cond_enc,
                batch,
                model_type,
                device,
                grid_attrs=ckpt.get("grid_attrs"),
            )
            T_tgt = batch["T_target"].squeeze(1)

            if collect_indices and i in collect_indices:
                collected_vols[i] = T_pred.squeeze().cpu()

            # Prepare 5D tensors for metrics (B, C, Nz, Ny, Nx)
            T_pred_5d = T_pred.reshape(1, 1, 64, 64, 64)
            T_tgt_5d = T_tgt.reshape(1, 1, 64, 64, 64)

            # Un-normalise for physics metrics (matching evaluate_physical_testset.py)
            T_pred_phys = T_pred_5d * _T_REF + _T_AMBIENT
            T_tgt_phys = T_tgt_5d * _T_REF + _T_AMBIENT

            iou = iou_melt_volumes(T_pred_phys, T_tgt_phys, _T_LIQUIDUS)
            d_pred = melt_pool_extent(T_pred_phys, _T_LIQUIDUS)["D"]
            d_gt = melt_pool_extent(T_tgt_phys, _T_LIQUIDUS)["D"]
            offset = hotspot_offset_vox(T_pred_5d, T_tgt_5d)
            t_max_err = abs(T_pred.max().item() - T_tgt.max().item()) * _T_REF

            rows.append(
                {
                    "Model": name,
                    "Sample": f"s{i:04d}",
                    "IoU": iou,
                    "Depth_GT": d_gt,
                    "Depth_Pred": d_pred,
                    "T_max_Error": t_max_err,
                    "Offset_vox": offset,
                }
            )
        except Exception as exc:
            logger.warning("Error on sample %d for model %s: %s", i, name, exc)

    # Record peak tensor memory (actual allocations, not allocator pool)
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info("Model %s — peak VRAM allocated: %.0f MB", name, peak_vram_mb)

    # Explicit CUDA cleanup — prevents VRAM accumulation across sequential models
    del model, cond_enc
    torch.cuda.empty_cache()

    return rows, collected_vols


def run_physical_fidelity_benchmark(
    models: list[tuple[str, str, str]],
    ds_cfg: Any,
    device: torch.device,
    output_path: Path | str = "docs/assets/physical_fidelity_benchmark_detailed.png",
    mlflow_run_id: str | None = None,
    test_indices: list[int] | None = None,
    collect_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[int, torch.Tensor]]]:
    """Evaluate a list of model checkpoints on physical fidelity metrics.

    Args:
        models:       List of (display_name, ckpt_path, model_type) tuples.
        ds_cfg:       Dataset configuration object (passed to _run_model_benchmark).
        device:       Torch device for inference.
        output_path:  Path to save the 4-panel dashboard PNG.
        mlflow_run_id: If not None, log the output PNG to this MLflow run.

    Returns:
        DataFrame with columns: Model, Sample, IoU, Depth_GT, Depth_Pred,
        T_max_Error, Offset_vox.
    """
    warnings.warn(
        "run_physical_fidelity_benchmark is deprecated — use run_physics_sweep "
        "from neural_pbf.eval.benchmark.runner instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_vols: dict[str, dict[int, torch.Tensor]] = {}
    for name, ckpt_path, model_type in models:
        rows, vols = _run_model_benchmark(
            name,
            ckpt_path,
            model_type,
            ds_cfg,
            device,
            test_indices=test_indices,
            collect_indices=collect_indices,
        )
        all_rows.extend(rows)
        all_vols[name] = vols

    df = (
        pd.DataFrame(
            all_rows,
            columns=[
                "Model",
                "Sample",
                "IoU",
                "Depth_GT",
                "Depth_Pred",
                "T_max_Error",
                "Offset_vox",
            ],
        )
        if all_rows
        else pd.DataFrame(
            columns=[
                "Model",
                "Sample",
                "IoU",
                "Depth_GT",
                "Depth_Pred",
                "T_max_Error",
                "Offset_vox",
            ]
        )
    )

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=200)
        fig.patch.set_facecolor(THEME.physical.bg_figure)

        if not df.empty:
            model_names = df["Model"].unique().tolist()
            palette = _model_palette(model_names)

            # [0,0] IoU violin
            sns.violinplot(
                data=df,
                x="Model",
                y="IoU",
                hue="Model",
                ax=axes[0, 0],
                palette=palette,
                inner="box",
                legend=False,
                density_norm="width",  # Standardize violin widths
            )
            axes[0, 0].set_title("Meltpool IoU Distribution", pad=15)
            axes[0, 0].set_ylim(0, 1.2)  # Increased to prevent clipping
            axes[0, 0].set_facecolor(THEME.physical.bg_axis)
            axes[0, 0].grid(
                THEME.physical.grid.enabled,
                which=THEME.physical.grid.which,
                axis=THEME.physical.grid.axis,
                linestyle=THEME.physical.grid.linestyle,
                alpha=THEME.physical.grid.alpha,
                color=THEME.physical.grid.color,
            )

            # [0,1] Depth scatter
            axes[0, 1].set_facecolor(THEME.physical.bg_axis)
            for mname, grp in df.groupby("Model"):
                axes[0, 1].scatter(
                    grp["Depth_GT"],
                    grp["Depth_Pred"],
                    label=str(mname),
                    color=palette.get(str(mname), "#ffffff"),
                    alpha=0.8,
                    s=50,  # Increased size
                    edgecolor="w",  # Add outline for visibility
                    linewidth=0.5,
                )
            # Ideal line
            all_vals = pd.concat([df["Depth_GT"], df["Depth_Pred"]]).dropna()
            if len(all_vals) > 0:
                lo, hi = float(all_vals.min()), float(all_vals.max())
                pad = (hi - lo) * 0.1
                axes[0, 1].plot(
                    [lo - pad, hi + pad],
                    [lo - pad, hi + pad],
                    "w--",
                    linewidth=1.5,
                    label="ideal",
                )
                axes[0, 1].set_xlim(lo - pad, hi + pad)
                axes[0, 1].set_ylim(lo - pad, hi + pad)
            axes[0, 1].set_xlabel("GT Depth (vox)")
            axes[0, 1].set_ylabel("Pred Depth (vox)")
            axes[0, 1].set_title("Meltpool Depth: GT vs Pred")
            axes[0, 1].legend(fontsize=7)
            axes[0, 1].set_facecolor(THEME.physical.bg_axis)
            axes[0, 1].grid(
                THEME.physical.grid.enabled,
                which=THEME.physical.grid.which,
                axis="both",  # Scatter needs both usually, but let's stick to subclass logic if needed
                linestyle=THEME.physical.grid.linestyle,
                alpha=THEME.physical.grid.alpha,
                color=THEME.physical.grid.color,
            )

            # [1,0] T_max error boxplot
            sns.boxplot(
                data=df,
                x="Model",
                y="T_max_Error",
                hue="Model",
                ax=axes[1, 0],
                palette=palette,
                legend=False,
                whis=100,  # Full range whiskers
            )
            axes[1, 0].set_title("T_max Error [K]")
            axes[1, 0].set_facecolor(THEME.physical.bg_axis)
            axes[1, 0].grid(
                THEME.physical.grid.enabled,
                which=THEME.physical.grid.which,
                axis=THEME.physical.grid.axis,
                linestyle=THEME.physical.grid.linestyle,
                alpha=THEME.physical.grid.alpha,
                color=THEME.physical.grid.color,
            )

            # [1,1] Hotspot offset strip
            sns.stripplot(
                data=df,
                x="Model",
                y="Offset_vox",
                hue="Model",
                ax=axes[1, 1],
                palette=palette,
                jitter=True,
                size=4,
                legend=False,
            )
            axes[1, 1].set_title("Hotspot Offset [vox]")
            axes[1, 1].set_facecolor(THEME.physical.bg_axis)
            axes[1, 1].grid(
                THEME.physical.grid.enabled,
                which=THEME.physical.grid.which,
                axis=THEME.physical.grid.axis,
                linestyle=THEME.physical.grid.linestyle,
                alpha=THEME.physical.grid.alpha,
                color=THEME.physical.grid.color,
            )

        else:
            for ax in axes.flatten():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="white",
                )

        # Final layout adjustments for clean research-grade visualization
        fig.tight_layout(pad=5.0)  # Increased pad
        fig.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Fidelity dashboard saved to %s", output_path)

    if mlflow_run_id is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(output_path), artifact_path="eval")
        except Exception:
            pass  # Silently ignore MLflow errors in benchmark context
    return df, all_vols


def run_system_comparison(
    runs: dict[str, str],
    output_path: Path | str = "docs/assets/system_comparison.png",
    mlflow_run_id: str | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> pd.DataFrame:
    """Compare training runs by duration, GPU utilisation, and final loss.

    Args:
        runs:        Dict mapping run version label → MLflow run ID.  Each entry is
                     looked up via ``MlflowClient``; on any failure the row falls back
                     to NaN values so the chart still renders.
        output_path: Path to save the 2-panel figure.
        mlflow_run_id: If not None, log the PNG to this MLflow run.
        mlflow_tracking_uri: MLflow tracking server URI (default ``sqlite:///mlflow.db``).

    Returns:
        DataFrame with columns: Version, Duration (h), GPU Util (%),
        GPU Mem (MB), s/it, Final Loss.
        Returns empty DataFrame when *runs* is empty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs:
        df = pd.DataFrame(
            columns=[
                "Version",
                "Duration (h)",
                "GPU Util (%)",
                "GPU Mem (MB)",
                "s/it",
                "Final Loss",
            ]
        )
    else:
        rows: list[dict[str, Any]] = []
        client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        for version, run_id in runs.items():
            row: dict[str, Any] = {
                "Version": version,
                "Duration (h)": float("nan"),
                "GPU Util (%)": float("nan"),
                "GPU Mem (MB)": float("nan"),
                "s/it": float("nan"),
                "Final Loss": float("nan"),
            }
            try:
                run = client.get_run(run_id)
                start_ms = run.info.start_time or 0
                end_ms = run.info.end_time or start_ms
                row["Duration (h)"] = round((end_ms - start_ms) / 3_600_000.0, 3)

                util_hist = client.get_metric_history(run_id, "system/gpu_0_utilization_percentage")
                mem_hist = client.get_metric_history(run_id, "system/gpu_0_memory_usage_megabytes")
                if util_hist:
                    row["GPU Util (%)"] = round(float(np.mean([m.value for m in util_hist])), 1)
                if mem_hist:
                    row["GPU Mem (MB)"] = round(float(np.mean([m.value for m in mem_hist])), 1)

                loss_hist = client.get_metric_history(run_id, "train_loss")
                if len(loss_hist) > 1:
                    total_ms = loss_hist[-1].timestamp - loss_hist[0].timestamp
                    total_epochs = loss_hist[-1].step - loss_hist[0].step
                    if total_epochs > 0:
                        s_per_epoch = (total_ms / 1000.0) / total_epochs
                        batch_size = int(run.data.params.get("batch_size", 4))
                        n_train = int(run.data.params.get("n_train", 105))
                        its = max(n_train // batch_size, 1)
                        row["s/it"] = round(s_per_epoch / its, 4)

                row["Final Loss"] = run.data.metrics.get("val_loss", run.data.metrics.get("train_loss", float("nan")))
            except Exception as exc:
                logger.warning("Could not fetch MLflow run %r for %s: %s", run_id, version, exc)
            rows.append(row)
        df = pd.DataFrame(rows)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

        if not df.empty:
            palette = _model_palette(df["Version"].tolist())
            colors = [palette[v] for v in df["Version"]]

            # [0] Duration bar + s/it twin axis
            ax0 = axes[0]
            valid_dur = df["Duration (h)"].dropna()
            if not valid_dur.empty:
                ax0.bar(df["Version"], df["Duration (h)"].fillna(0), color=colors, alpha=0.8)
            ax0.set_title("Training Duration and Step Timing")
            ax0.set_ylabel("Duration (h)")
            ax0_twin = ax0.twinx()
            valid_sit = df["s/it"].dropna()
            if not valid_sit.empty:
                ax0_twin.plot(
                    df["Version"],
                    df["s/it"].fillna(0),
                    color="#f39c12",
                    marker="o",
                    linewidth=2,
                    label="s/it",
                )
                ax0_twin.set_ylabel("s/it")

            # [1] GPU scatter (util % vs mem MB)
            ax1 = axes[1]
            valid_gpu = df[["GPU Util (%)", "GPU Mem (MB)"]].dropna()
            if not valid_gpu.empty:
                for _, row in df.iterrows():
                    util = row["GPU Util (%)"]
                    mem = row["GPU Mem (MB)"]
                    version = row["Version"]
                    if not (np.isnan(util) or np.isnan(mem)):
                        ax1.scatter(util, mem, color=palette[str(version)], s=100, zorder=5)
                        ax1.annotate(
                            str(version),
                            (util, mem),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                            color="white",
                        )
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 16384)
            ax1.set_xlabel("GPU Util (%)")
            ax1.set_ylabel("GPU Mem (MB)")
            ax1.set_title("GPU Utilisation vs Memory")
        else:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="white",
                )

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if mlflow_run_id is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(output_path), artifact_path="eval")
        except Exception:
            pass

    return df


def run_spectral_benchmark(
    models: list[tuple[str, str, str]],
    ds_cfg: Any,
    device: torch.device,
    output_path: Path | str = "docs/assets/spectral_fidelity_benchmark.png",
    mlflow_run_id: str | None = None,
) -> None:
    """Compare spectral fidelity (aliasing) across multiple models.

    Generates a log-log PSD plot comparing models to Ground Truth.
    """
    warnings.warn(
        "run_spectral_benchmark is deprecated — use the spectral pipeline in experiments/benchmark_suite.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from neural_pbf.data.fm_dataset import FMThermalDataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = FMThermalDataset(ds_cfg)
        # Use the same representative sample for all models for fair comparison
        sample_idx = int(len(ds) * 0.95)  # Deep in test set
        sample = ds[sample_idx]
        gt_volume = sample["T_target"].squeeze(0)  # (D, H, W)
        gt_freqs, gt_psd = compute_radial_psd(gt_volume.to(device))
    except Exception as exc:
        logger.error("Spectral benchmark failed to load data: %s", exc)
        return

    model_psds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, ckpt_path, model_type in models:
        try:
            # Re-use logic to load model (could be refactored, but keeping it simple)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            # This is a bit redundant with _run_model_benchmark,
            # but allows for a cleaner isolated spectral run.
            if model_type == "net":
                from neural_pbf.models.generative.fm.config import FMConfig
                from neural_pbf.models.generative.fm.velocity_net import VelocityNet

                fm_cfg = FMConfig(**ckpt["fm_cfg"])
                model = VelocityNet(fm_cfg).to(device)
            elif model_type == "dit":
                from neural_pbf.models.generative.fm.dit import VelocityDiT

                model = VelocityDiT(depth=6, embed_dim=256, num_heads=8, patch_size=8, input_size=64).to(device)
            elif model_type in ("rope", "triton"):
                from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE

                # v3/v4 used patch_size 8, v5 used patch_size 4
                p_size = 4 if "triton" in model_type else 8
                model = VelocityDiTRoPE(
                    depth=6,
                    embed_dim=288,
                    num_heads=8,
                    patch_size=p_size,
                    input_size=64,
                ).to(device)
            else:
                continue

            from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder

            cond_enc = ConditioningEncoder(12, 128).to(device)

            model.load_state_dict(ckpt["model_state"])
            cond_enc.load_state_dict(ckpt["cond_encoder_state"])
            model.eval()
            cond_enc.eval()

            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            T_pred = _rollout_sample(model, cond_enc, batch, model_type, device)

            freqs, psd = compute_radial_psd(T_pred[0])
            model_psds[name] = (freqs, psd)

        except Exception as exc:
            logger.warning("Failed to compute PSD for %s: %s", name, exc)

    # Plot using standardized look
    plot_spectral_comparison(
        model_results=model_psds,
        gt_result=(gt_freqs, gt_psd),
        output_path=output_path,
        patch_sizes=[4, 8],
    )

    if mlflow_run_id is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(output_path), artifact_path="eval")
        except Exception:
            pass
