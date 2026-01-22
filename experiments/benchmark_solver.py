import sys
from pathlib import Path

import torch

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.neural_pbf.core.config import LengthUnit, SimulationConfig
from src.neural_pbf.core.state import SimulationState
from src.neural_pbf.integrator.stepper import TimeStepper
from src.neural_pbf.physics.material import MaterialConfig
from src.neural_pbf.utils.profiling import PerformanceTracker


def run_benchmark(
    nx=512, ny=256, nz=128, steps=50, material="default", use_triton=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: CUDA not available. Benchmarking on CPU is not representative for GPU kernels."
        )

    # 1. Config
    if material == "ss316l":
        mat_cfg = MaterialConfig.ss316l_preset()
    elif material == "tdep_linear":
        mat_cfg = MaterialConfig(
            k_powder=0.2,
            k_solid=25.0,
            k_liquid=45.0,
            cp_base=500.0,
            rho=7900.0,
            T_solidus=1650.0,
            T_liquidus=1700.0,
            latent_heat_L=2.7e5,
            use_T_dep=True,
            k_solid_T_coeff=0.0005,
            cp_T_coeff=0.0002,
        )
    else:
        mat_cfg = MaterialConfig(
            k_powder=0.2,
            k_solid=25.0,
            k_liquid=45.0,
            cp_base=500.0,
            rho=7900.0,
            T_solidus=1650.0,
            T_liquidus=1700.0,
            latent_heat_L=2.7e5,
            use_T_dep=False,
        )

    sim_cfg = SimulationConfig(
        Lx=nx * 1e-2,
        Ly=ny * 1e-2,
        Lz=nz * 1e-2,  # mm
        Nx=nx,
        Ny=ny,
        Nz=nz,
        dt_base=2e-6,
        length_unit=LengthUnit.MILLIMETERS,
    )

    stepper = TimeStepper(sim_cfg, mat_cfg)

    # 2. State Initialization
    T = torch.full((1, 1, nx, ny, nz), 300.0, device=device, dtype=torch.float32)
    mask = torch.ones_like(T, dtype=torch.uint8)  # Fully solid/powder for benchmark
    state = SimulationState(T=T, t=0.0, material_mask=mask)

    Q_ext = torch.zeros_like(T)
    Q_ext[..., nx // 2, ny // 2, -1] = 1e11

    # 3. Warm-up
    print(f"Propagating {nx}x{ny}x{nz} grid ({nx * ny * nz / 1e6:.1f}M cells)...")
    for _ in range(5):
        state = stepper.step_adaptive(
            state, dt_target=sim_cfg.dt_base, Q_ext=Q_ext, use_triton=use_triton
        )

    torch.cuda.synchronize()

    # 4. Actual Benchmark
    tag_mat = material
    tag_solver = "Triton" if use_triton else "PyTorch"
    tag = f"{tag_solver} ({tag_mat})"
    print(f"Starting Benchmark: {tag} for {steps} steps...")

    tracker = PerformanceTracker(tag, device=device)
    with tracker:
        for _i in range(steps):
            state = stepper.step_explicit_euler(
                state, sim_cfg.dt_base, Q_ext=Q_ext, use_triton=use_triton
            )

    res = tracker.result
    print(f"DONE: {res.elapsed_ms:.2f} ms total | {res.elapsed_ms / steps:.2f} ms/step")
    print(f"Peak VRAM: {res.max_vram_mb:.1f} MB")
    return res


if __name__ == "__main__":
    print("--- Performance Benchmark (3D Solver: Realistic LUT) ---")

    NX, NY, NZ = 512, 256, 128
    STEPS = 20

    # Case 1: Baseline PyTorch (Constant)
    res_base = run_benchmark(nx=NX, ny=NY, nz=NZ, steps=STEPS, material="default")

    # Case 2: PyTorch (SS316L LUT)
    res_tdep = run_benchmark(
        nx=NX, ny=NY, nz=NZ, steps=STEPS, material="ss316l", use_triton=False
    )

    # Case 3: Triton (SS316L LUT)
    res_triton = run_benchmark(
        nx=NX, ny=NY, nz=NZ, steps=STEPS, material="ss316l", use_triton=True
    )

    print("\n" + "=" * 40)
    print(f"SUMMARY ({NX}x{NY}x{NZ} Grid)")
    print(
        f"PyTorch (Const):   {res_base.elapsed_ms / STEPS:.2f} ms/step, {res_base.max_vram_mb:.1f} MB"
    )
    print(
        f"PyTorch (SS316L): {res_tdep.elapsed_ms / STEPS:.2f} ms/step, {res_tdep.max_vram_mb:.1f} MB"
    )
    print(
        f"Triton  (SS316L): {res_triton.elapsed_ms / STEPS:.2f} ms/step, {res_triton.max_vram_mb:.1f} MB"
    )

    speedup = res_tdep.elapsed_ms / res_triton.elapsed_ms
    vram_save = res_tdep.max_vram_mb - res_triton.max_vram_mb

    print("-" * 40)
    print(f"Speedup Triton vs PyTorch (LUT): {speedup:.2f}x")
    print(f"VRAM Saved: {vram_save:.1f} MB")

    overhead = (res_tdep.elapsed_ms / res_base.elapsed_ms - 1) * 100
    print(f"PyTorch LUT Overhead: {overhead:.1f}%")
    print("=" * 40)
