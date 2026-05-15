"""benchmark_suite.py — Unified LPBF benchmark CLI orchestrator."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import mlflow

from neural_pbf.eval.benchmark.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TRAINING_RUN_IDS,
)
from neural_pbf.schemas.viz import THEME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical Model Registry & Global Color Stability
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "Baseline":      ("checkpoints/fm/best.pt",            "net"),
    "DiT v1":        ("checkpoints/dit_v1/best.pt",        "dit"),
    "DiT v2":        ("checkpoints/dit_v2/best.pt",        "dit"),
    "DiT v3":        ("checkpoints/dit_physics/best.pt",   "rope"),
    "DiT v4":        ("checkpoints/dit_accelerate/best.pt","rope"),
    "Hero Run (v5)": ("checkpoints/dit_triton/best.pt",    "triton"),
}

CANONICAL_MODELS = ["Baseline", "DiT v1", "DiT v2", "DiT v3", "DiT v4", "Hero Run (v5)"]

def get_master_color_map() -> dict[str, str]:
    pal = THEME.physical.color_palette
    return {name: pal[i % len(pal)] for i, name in enumerate(CANONICAL_MODELS)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").lower()


def run_benchmark(args: argparse.Namespace) -> None:
    from neural_pbf.eval.benchmark.constants import N_EULER_STEPS, PATCH_SIZE
    from neural_pbf.eval.benchmark.dataset import build_test_dataset
    from neural_pbf.eval.benchmark.io import mlflow_log_paths, save_metrics_csv
    from neural_pbf.eval.benchmark.mlflow_data import fetch_mlflow_data
    from neural_pbf.eval.benchmark.model_loader import load_model_adaptive
    from neural_pbf.eval.benchmark.runner import run_physics_sweep
    from neural_pbf.eval.metrics.spectral import (
        calculate_boundary_discontinuity,
        calculate_total_variation,
        compute_axis_psd,
        compute_radial_psd,
        plot_spectral_full_analysis,
    )
    from neural_pbf.eval.viz.physical_dashboard import plot_physical_dashboard
    from neural_pbf.eval.viz.structural import plot_tv_pbd_comparison
    from neural_pbf.eval.viz.system_dashboard import plot_system_dashboard
    from neural_pbf.schemas.run_meta import RunMeta
    from neural_pbf.tracking.run_context import RunContext
    from neural_pbf.data.fm_dataset import FMThermalDataset, FMDatasetConfig

    # Enable system metrics for inference with high frequency (1s)
    import os
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        logger.warning("Could not enable MLflow system metrics logging.")

    device = torch.device(args.device)
    n_steps = args.n_euler_steps or N_EULER_STEPS
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    overrides = _parse_run_id_overrides(args.run_ids)
    effective_run_ids = {**TRAINING_RUN_IDS, **overrides}
    color_map = get_master_color_map()

    run_meta = RunMeta(
        seed=args.seed, device=str(device), dtype="float32",
        started_at=str(time.time()), dx=0.0, dy=0.0, dz=0.0, dt=0.0,
        grid_shape=[0, 0, 0],
    )

    with RunContext.with_full_tracking(
        mlflow_uri=args.mlflow_uri,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=f"BenchmarkSuite-{time.strftime('%Y%m%dT%H%M%S')}",
        tags={"metrics": ",".join(args.metrics), "models": ",".join(args.models)},
        out_dir=output_dir / "benchmark_run",
        run_meta=run_meta,
    ) as ctx:

        torch.manual_seed(args.seed)

        # 1. Dataset Setup
        valid_paths = [p for p in args.dataset if Path(p).exists()]
        base_ds = FMThermalDataset(FMDatasetConfig(h5_paths=valid_paths, Q_ref=1.35e15))
        
        from torch.utils.data import random_split
        n_total = len(base_ds)
        n_train, n_val = int(n_total * 0.7), int(n_total * 0.2)
        _, _, test_ds_raw = random_split(base_ds, [n_train, n_val, n_total - n_train - n_val], generator=torch.Generator().manual_seed(42))
        test_indices = [idx for idx in test_ds_raw.indices if "offline_dataset_notebook.h5" not in str(base_ds._keys[idx][0])]
        
        spectral_ref_idx = test_indices[len(test_indices) // 2]
        collect_indices = [spectral_ref_idx]

        # 2. Physics & Model Sweep (with Nested Runs for Inference Metrics)
        all_vols: dict[str, dict[int, torch.Tensor]] = {}
        physics_df = pd.DataFrame()
        master_throughput: dict[str, dict[str, float]] = {}

        logger.info("=== Starting Model Evaluation Sweep ===")
        for model_name in args.models:
            if model_name not in _MODEL_REGISTRY: continue
            ckpt_path, model_type = _MODEL_REGISTRY[model_name]
            
            # Use nested run for each model to capture isolated GPU metrics
            with mlflow.start_run(run_name=f"Eval-{model_name}", nested=True) as nested:
                mlflow.set_tag("model_version", model_name)
                try:
                    model, cond_enc, patch_size, grid_attrs = load_model_adaptive(model_name, ckpt_path, model_type, device)
                    ds = build_test_dataset(base_ds, patch_size=PATCH_SIZE, model_type=model_type)
                    
                    sub_df, sub_vols, throughput = run_physics_sweep(
                        models=[(model_name, model, cond_enc, model_type) + ((grid_attrs,) if grid_attrs else ())],
                        dataset=ds, device=device, test_indices=test_indices, collect_indices=collect_indices, n_steps=n_steps
                    )
                    
                    physics_df = pd.concat([physics_df, sub_df], ignore_index=True)
                    all_vols[model_name] = sub_vols.get(model_name, {})
                    if model_name in throughput:
                        master_throughput[model_name] = throughput[model_name]
                        for k, v in throughput[model_name].items():
                            mlflow.log_metric(f"inf_{k}", v)
                except Exception as exc:
                    logger.warning("Failed evaluating %s: %s", model_name, exc)

        # 3. Persistence & Plotting (Physical)
        if not physics_df.empty:
            physics_png = output_dir / "benchmark_dashboard_physics.png"
            plot_physical_dashboard(physics_df, physics_png, color_map=color_map)
            mlflow_log_paths(ctx, [physics_png], artifact_subdir="plots")

        # 4. Spectral & Structural
        if "spectral" in args.metrics:
            gt_volume = base_ds[spectral_ref_idx]["T_target"].squeeze().to(device)
            gt_res = {"radial": compute_radial_psd(gt_volume), "axes": compute_axis_psd(gt_volume)}
            model_spectral_results = {m: {"radial": compute_radial_psd(v[spectral_ref_idx].to(device)), "axes": compute_axis_psd(v[spectral_ref_idx].to(device))} 
                                     for m, v in all_vols.items() if spectral_ref_idx in v}
            structural_df = pd.DataFrame([{"Model": m, "TV": calculate_total_variation(v[spectral_ref_idx].to(device)), 
                                         "PBD": calculate_boundary_discontinuity(v[spectral_ref_idx].to(device), patch_size=8)} 
                                        for m, v in all_vols.items() if spectral_ref_idx in v])

            if model_spectral_results:
                spectral_png = output_dir / "benchmark_spectral_analysis.png"
                plot_spectral_full_analysis(model_results=model_spectral_results, gt_result=gt_res, output_path=spectral_png, color_map=color_map)
                struct_png = output_dir / "benchmark_structural_metrics.png"
                plot_tv_pbd_comparison(structural_df, struct_png, color_map=color_map)
                mlflow_log_paths(ctx, [spectral_png, struct_png], artifact_subdir="plots")

        # 5. System Dashboard (using bench_run_id to find nested inf stats)
        if "system" in args.metrics:
            logger.info("=== Fetching MLflow System Stats (Train & Inf) ===")
            selected_ids = {k: v for k, v in effective_run_ids.items() if k in args.models}
            
            # Get active run ID via mlflow directly
            active_run = mlflow.active_run()
            bench_run_id = active_run.info.run_id if active_run else None
            
            train_df = fetch_mlflow_data(
                selected_ids, 
                tracking_uri=args.mlflow_uri, 
                bench_run_id=bench_run_id
            )

            inf_df = pd.DataFrame([{"Model": m, "s_per_sample": t.get("s_per_sample", float("nan"))} 
                                 for m, t in master_throughput.items()])
            
            system_png = output_dir / "benchmark_system_comparison.png"
            plot_system_dashboard(train_df, inf_df, system_png, color_map=color_map)
            mlflow_log_paths(ctx, [system_png], artifact_subdir="plots")

    logger.info("Benchmark complete. Artifacts in %s", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LPBF surrogate benchmark suite")
    p.add_argument("--models", nargs="+", default=list(_MODEL_REGISTRY.keys()))
    p.add_argument("--metrics", nargs="+", default=["physics", "spectral", "system"])
    p.add_argument("--dataset", nargs="+", default=["data/offline_dataset_test.h5", "data/offline_dataset_notebook.h5"])
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--output-dir", default="docs/assets")
    p.add_argument("--raw-dir", default="docs/assets/raw_metrics")
    p.add_argument("--mlflow-uri", default=MLFLOW_TRACKING_URI)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-ids", default=None)
    p.add_argument("--n-euler-steps", type=int, default=None)
    return p


def _parse_run_id_overrides(run_ids_str: str | None) -> dict[str, str]:
    if not run_ids_str:
        return {}
    return {k.strip(): v.strip() for part in run_ids_str.split(",") if ":" in part for k, _, v in [part.partition(":")]}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
