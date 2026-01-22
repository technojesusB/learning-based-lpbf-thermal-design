import sys
from pathlib import Path

import torch

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))  # noqa: E402

from experiments.run_3d import run_3d_experiment  # noqa: E402
from src.neural_pbf.core.config import LengthUnit, SimulationConfig  # noqa: E402
from src.neural_pbf.physics.material import MaterialConfig  # noqa: E402
from src.neural_pbf.schemas.artifacts import ArtifactConfig  # noqa: E402


def run_comparison():
    # 1. 'Sweet Spot' Resolution Config
    # Grid: 1024 x 512 x 128 = 67,108,864 nodes
    # Sub-micron resolution: 1mm / 1024 ~= 0.97 um
    sim_cfg = SimulationConfig(
        Lx=1.0,
        Ly=0.5,
        Lz=0.125,
        Nx=1024,
        Ny=512,
        Nz=128,
        dt_base=1e-6,  # Smaller dt for extreme stability at sub-micron grid
        length_unit=LengthUnit.MILLIMETERS,
    )

    # 2. Artifact Config (High-res artifacts)
    art_cfg = ArtifactConfig(
        enabled=True,
        png_every_n_steps=100,
        html_every_n_steps=200,
        make_report=True,
        downsample=2,  # Downsample 3D volume for frontend performance
    )

    materials = {
        "Ti64": MaterialConfig.ti64_preset(),
        "AlSi10Mg": MaterialConfig.alsi10mg_preset(),
        "IN718": MaterialConfig.in718_preset(),
    }

    print("\n" + "=" * 60)
    print(">>> STARTING EXTREME-FIDELITY MATERIAL COMPARISON <<<")
    print(
        f"Grid: {sim_cfg.Nx}x{sim_cfg.Ny}x{sim_cfg.Nz} "
        f"({sim_cfg.Nx * sim_cfg.Ny * sim_cfg.Nz / 1e6:.1f}M nodes)"
    )
    print("=" * 60)

    for name, mat in materials.items():
        print(f"\n[Run {name}] Starting simulation...")

        # Optimization: Use Triton if CUDA available
        # Note: run_3d_experiment defaults to Triton if not specified?
        # Actually it uses step_adaptive which uses Triton internally if enabled.
        # But we need to ensure the stepper uses Triton.

        run_3d_experiment(
            run_name=f"compare_hi_fid_{name.lower()}",
            mat_cfg=mat,
            sim_cfg=sim_cfg,
            art_cfg=art_cfg,
            total_time=0.3e-3,  # 0.3ms to capture stable state
            scan_speed=1.0,  # 1 m/s
        )

    print("\n" + "=" * 60)
    print(">>> COMPARISON EXPERIMENT COMPLETE <<<")
    print("=" * 60)


if __name__ == "__main__":
    # Safety: Ensure CUDA
    if not torch.cuda.is_available():
        print(
            "CRITICAL: CUDA not detected. "
            "This experiment requires Triton and GPU memory."
        )
        sys.exit(1)

    run_comparison()
