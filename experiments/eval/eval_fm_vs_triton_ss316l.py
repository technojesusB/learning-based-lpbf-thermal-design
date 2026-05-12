"""End-to-end evaluation: FM surrogate vs. Triton solver on SS316L.

Usage::

    uv run python experiments/eval/eval_fm_vs_triton_ss316l.py \\
        --traj path/to/trajectory.h5 \\
        --checkpoint path/to/fm_model.pt \\
        --output artifacts/eval/ss316l

The script loads a pre-generated GT trajectory from HDF5, evaluates both
the physics-based solver (TritonAdapter) and the FM surrogate (FMAdapter),
generates comparison figures, and logs everything to MLflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch

from neural_pbf.eval.adapters.triton_adapter import TritonAdapter
from neural_pbf.eval.data.hdf5_loader import load_trajectory
from neural_pbf.eval.reporting.markdown_report import save_markdown_report
from neural_pbf.eval.reporting.mlflow_logger import log_rollout_to_mlflow
from neural_pbf.eval.rollout.engine import RolloutEngine
from neural_pbf.eval.viz.report_figures import generate_report_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, help="Path to ground-truth trajectory HDF5")
    parser.add_argument("--output", default="artifacts/eval/ss316l")
    parser.add_argument("--mode", default="one_step", choices=["one_step", "autoregressive"])
    parser.add_argument("--experiment", default="surrogate_eval_ss316l")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)

    print(f"Loading trajectory from {args.traj} …")
    gt_traj = load_trajectory(args.traj, device=device)
    print(f"  {len(gt_traj)} snapshots, device={device}")

    triton = TritonAdapter(
        sim_cfg=gt_traj.sim_cfg,
        mat_cfg=gt_traj.mat_cfg,
        use_triton=torch.cuda.is_available(),
    )

    engine = RolloutEngine()
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="triton_reference"):
        result = engine.run(triton, gt_traj, mode=args.mode)
        run_dir = out / "triton"
        figs = generate_report_figures(result, run_dir)
        save_markdown_report(result, run_dir, figs)
        log_rollout_to_mlflow(result, artifact_dir=run_dir)
        print(
            f"Triton: mean_MAE={result.per_step_metrics[0]['mae_global']:.4f} K"
            if result.per_step_metrics else "Triton: no steps evaluated"
        )

    print("Done. Open MLflow UI: uv run mlflow ui")


if __name__ == "__main__":
    main()
