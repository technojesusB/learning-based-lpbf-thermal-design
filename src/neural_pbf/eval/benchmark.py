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

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Fixed color palette — cycles if more than 5 models
_PALETTE_COLORS = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def _model_palette(names: list[str]) -> dict[str, str]:
    """Return a dict mapping each name to a hex color.

    Colors cycle through a fixed set of 5 values if more names are provided.
    """
    return {
        name: _PALETTE_COLORS[i % len(_PALETTE_COLORS)] for i, name in enumerate(names)
    }


def _rollout_sample(
    model: nn.Module,
    cond_enc: nn.Module,
    batch: dict[str, Any],
    model_type: str,
    device: torch.device,
    n_steps: int = 25,
) -> torch.Tensor:
    """Run Euler integration to produce T_pred for one batch.

    Args:
        model:       Velocity/DiT model.
        cond_enc:    Conditioning encoder.
        batch:       Dict with tensors already on *device*.
        model_type:  One of "net", "dit", "rope".
        device:      Compute device.
        n_steps:     Number of Euler steps.

    Returns:
        T_pred tensor with shape matching batch["T_target"].squeeze(1).
    """
    T_in = batch["T_in"].squeeze(1)
    mask = batch["mask"].squeeze(1)
    Q = batch["Q"].squeeze(1)
    cond = batch["conditioning"]
    cond_emb = cond_enc(cond)

    B = T_in.shape[0]
    x = torch.randn_like(T_in)
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=device)
            inp = torch.cat([x, mask, Q], dim=1)
            if model_type == "rope":
                # For benchmark purposes pass zero coords when grid_attrs not available
                D, H, W = T_in.shape[-3], T_in.shape[-2], T_in.shape[-1]
                N_tokens = D * H * W  # token count (simplified: no patch splitting)
                coords_mm = torch.zeros(B, N_tokens, 3, device=device)
                v = model(inp, tau, cond_emb, coords_mm)
            else:
                v = model(inp, tau, cond_emb)
            x = x + v * dt

    return x


def _run_model_benchmark(
    name: str,
    ckpt_path: str,
    model_type: str,
    ds_cfg: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
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

    _T_LIQUIDUS = 0.6
    _T_REF = 2000.0

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        logger.error("Cannot load checkpoint %s: %s", ckpt_path, exc)
        return []

    try:
        if model_type == "net":
            fm_cfg = FMConfig(**ckpt["fm_cfg"])
            model = VelocityNet(fm_cfg).to(device)
            cond_enc = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim).to(
                device
            )
        elif model_type == "dit":
            from experiments.train_fm_dit import (
                VelocityDiT,  # lazy: experiments/ not a package
            )

            model = VelocityDiT(
                depth=6,
                embed_dim=256,
                num_heads=8,
                patch_size=8,
                input_size=64,
                in_channels=3,
                cond_embed_dim=128,
            ).to(device)
            cond_enc = ConditioningEncoder(12, 128).to(device)
        elif model_type in ("rope", "triton"):
            from experiments.train_fm_dit_rope import VelocityDiTRoPE

            model = VelocityDiTRoPE(
                depth=6,
                embed_dim=288,
                num_heads=8,
                patch_size=4,
                input_size=64,
                in_channels=3,
                cond_embed_dim=128,
            ).to(device)
            cond_enc = ConditioningEncoder(12, 128).to(device)
        else:
            logger.error("Unknown model_type %r for model %s", model_type, name)
            return []
        model.load_state_dict(ckpt["model_state"])
        cond_enc.load_state_dict(ckpt["cond_encoder_state"])
        model.eval()
        cond_enc.eval()
    except Exception as exc:
        logger.error("Failed to build/load model %s (%s): %s", name, model_type, exc)
        return []

    try:
        ds = FMThermalDataset(ds_cfg)
    except Exception as exc:
        logger.error("Failed to load dataset for %s: %s", name, exc)
        return []

    n = len(ds)
    test_start = int(n * 0.9)
    rows: list[dict[str, Any]] = []

    for i in range(test_start, n):
        try:
            sample = ds[i]
            batch = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            T_pred = _rollout_sample(model, cond_enc, batch, model_type, device)
            T_tgt = batch["T_target"].squeeze(1)

            T_pred_5d = T_pred.unsqueeze(1)
            T_tgt_5d = T_tgt.unsqueeze(1)

            iou = iou_melt_volumes(T_pred_5d, T_tgt_5d, _T_LIQUIDUS)
            d_pred = melt_pool_extent(T_pred_5d, _T_LIQUIDUS)["D"]
            d_gt = melt_pool_extent(T_tgt_5d, _T_LIQUIDUS)["D"]
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

    return rows


def run_physical_fidelity_benchmark(
    models: list[tuple[str, str, str]],
    ds_cfg: Any,
    device: torch.device,
    output_path: Path | str = "docs/assets/physical_fidelity_benchmark_detailed.png",
    mlflow_run_id: str | None = None,
) -> pd.DataFrame:
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for name, ckpt_path, model_type in models:
        rows = _run_model_benchmark(name, ckpt_path, model_type, ds_cfg, device)
        all_rows.extend(rows)

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
            )
            axes[0, 0].set_title("Meltpool IoU Distribution")
            axes[0, 0].set_ylim(0, 1)

            # [0,1] Depth scatter
            for mname, grp in df.groupby("Model"):
                axes[0, 1].scatter(
                    grp["Depth_GT"],
                    grp["Depth_Pred"],
                    label=str(mname),
                    color=palette.get(str(mname), "#ffffff"),
                    alpha=0.7,
                    s=20,
                )
            # Ideal line
            all_vals = pd.concat([df["Depth_GT"], df["Depth_Pred"]]).dropna()
            if len(all_vals) > 0:
                lo, hi = float(all_vals.min()), float(all_vals.max())
                axes[0, 1].plot([lo, hi], [lo, hi], "w--", linewidth=1, label="ideal")
            axes[0, 1].set_xlabel("GT Depth (vox)")
            axes[0, 1].set_ylabel("Pred Depth (vox)")
            axes[0, 1].set_title("Meltpool Depth: GT vs Pred")
            axes[0, 1].legend(fontsize=7)

            # [1,0] T_max error boxplot
            sns.boxplot(
                data=df,
                x="Model",
                y="T_max_Error",
                hue="Model",
                ax=axes[1, 0],
                palette=palette,
                legend=False,
            )
            axes[1, 0].set_title("T_max Error [K]")

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

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if mlflow_run_id is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(output_path), artifact_path="eval")
        except Exception:
            pass  # Silently ignore MLflow errors in benchmark context

    return df


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

                util_hist = client.get_metric_history(
                    run_id, "system/gpu_0_utilization_percentage"
                )
                mem_hist = client.get_metric_history(
                    run_id, "system/gpu_0_memory_usage_megabytes"
                )
                if util_hist:
                    row["GPU Util (%)"] = round(
                        float(np.mean([m.value for m in util_hist])), 1
                    )
                if mem_hist:
                    row["GPU Mem (MB)"] = round(
                        float(np.mean([m.value for m in mem_hist])), 1
                    )

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

                row["Final Loss"] = run.data.metrics.get(
                    "val_loss", run.data.metrics.get("train_loss", float("nan"))
                )
            except Exception as exc:
                logger.warning(
                    "Could not fetch MLflow run %r for %s: %s", run_id, version, exc
                )
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
                ax0.bar(
                    df["Version"], df["Duration (h)"].fillna(0), color=colors, alpha=0.8
                )
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
                        ax1.scatter(
                            util, mem, color=palette[str(version)], s=100, zorder=5
                        )
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
