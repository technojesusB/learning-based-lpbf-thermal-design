"""Material Zoo benchmark sweep.

Evaluates a stepper (Triton by default) across all materials in the zoo and
writes a comparison DataFrame as CSV + prints a summary table.

Usage::

    uv run python experiments/eval/eval_material_zoo.py \\
        --traj-dir data/trajectories/ \\
        --output artifacts/eval/zoo

Each material is expected to have a trajectory file named
``<material_name>.h5`` in *traj-dir*.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_pbf.eval.adapters.triton_adapter import TritonAdapter
from neural_pbf.eval.data.hdf5_loader import load_trajectory
from neural_pbf.eval.zoo.benchmark import run_zoo_benchmark
from neural_pbf.eval.zoo.materials import MaterialZoo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", required=True, help="Directory with <name>.h5 trajectories")
    parser.add_argument("--output", default="artifacts/eval/zoo")
    parser.add_argument("--mode", default="one_step", choices=["one_step", "autoregressive"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_dir = Path(args.traj_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    zoo = MaterialZoo.default()
    trajectories = {}
    for entry in zoo.all():
        h5_path = traj_dir / f"{entry.name}.h5"
        if h5_path.exists():
            print(f"Loading {entry.name} from {h5_path} …")
            trajectories[entry.name] = load_trajectory(h5_path, device=device)
        else:
            print(f"  Skipping {entry.name}: {h5_path} not found")

    if not trajectories:
        print("No trajectories found. Exiting.")
        return

    # Use the first trajectory's cfg to build the adapter
    first_traj = next(iter(trajectories.values()))
    stepper = TritonAdapter(
        sim_cfg=first_traj.sim_cfg,
        mat_cfg=first_traj.mat_cfg,
        use_triton=torch.cuda.is_available(),
    )

    df = run_zoo_benchmark(stepper, trajectories, zoo=zoo, mode=args.mode)
    csv_path = out / "zoo_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
