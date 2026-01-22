import torch
import numpy as np
import time
from pathlib import Path
import sys
import datetime

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.neural_pbf.core.config import SimulationConfig, LengthUnit
from src.neural_pbf.core.state import SimulationState
from src.neural_pbf.physics.material import MaterialConfig
from src.neural_pbf.integrator.stepper import TimeStepper
from src.neural_pbf.utils.profiling import PerformanceTracker

def run_benchmark(nx=512, ny=256, nz=128, steps=50, use_t_dep=False, use_triton=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Benchmarking on CPU is not representative for GPU kernels.")

    # 1. Config
    mat_cfg = MaterialConfig(
        k_powder=0.2, k_solid=25.0, k_liquid=45.0,
        cp_base=500.0, rho=7900.0,
        T_solidus=1650.0, T_liquidus=1700.0,
        latent_heat_L=2.7e5,
        use_T_dep=use_t_dep,
        k_solid_T_coeff=0.0005, # Example 0.05% per K
        cp_T_coeff=0.0002
    )
    
    sim_cfg = SimulationConfig(
        Lx=1.0, Ly=0.5, Lz=0.25,
        Nx=nx, Ny=ny, Nz=nz,
        dt_base=2e-6,
        length_unit=LengthUnit.MILLIMETERS
    )
    
    stepper = TimeStepper(sim_cfg, mat_cfg)
    
    # 2. State Initialization
    T = torch.full((1, 1, nx, ny, nz), 300.0, device=device, dtype=torch.float32)
    mask = torch.zeros_like(T, dtype=torch.uint8)
    state = SimulationState(T=T, t=0.0, material_mask=mask)
    
    # Pre-generate a dummy Q_ext to avoid measuring source calculation overhead
    Q_ext = torch.zeros_like(T)
    # Add a "hot spot" to trigger material property lookups (melt fraction etc)
    Q_ext[..., nx//2, ny//2, -1] = 1e11 

    # 3. Warm-up
    print(f"Propagating {nx}x{ny}x{nz} grid ({nx*ny*nz/1e6:.1f}M cells)...")
    for _ in range(5):
        state = stepper.step_adaptive(state, dt_target=sim_cfg.dt_base, Q_ext=Q_ext)
    
    torch.cuda.synchronize()
    
    # 4. Actual Benchmark
    tag_tdep = "T-Dep" if use_t_dep else "Const"
    tag_solver = "Triton" if use_triton else "PyTorch"
    tag = f"{tag_solver} ({tag_tdep})"
    print(f"Starting Benchmark: {tag} for {steps} steps...")
    
    with PerformanceTracker(tag, device=device) as tracker:
        for i in range(steps):
            state = stepper.step_adaptive(state, dt_target=sim_cfg.dt_base, Q_ext=Q_ext, use_triton=use_triton)
            
    res = tracker.result
    print(f"DONE: {res.elapsed_ms:.2f} ms total | {res.elapsed_ms/steps:.2f} ms/step")
    print(f"Peak VRAM: {res.max_vram_mb:.1f} MB")
    return res

if __name__ == "__main__":
    # Small test first to verify everything works
    print("--- Performance Benchmark (3D Solver) ---")
    
    # Case 1: Baseline
    res_base = run_benchmark(nx=512, ny=256, nz=128, steps=20, use_t_dep=False)
    
    # Case 2: T-Dependent PyTorch
    res_tdep = run_benchmark(nx=512, ny=256, nz=128, steps=20, use_t_dep=True, use_triton=False)
    
    # Case 3: Triton (Fused & T-Dependent)
    res_triton = run_benchmark(nx=512, ny=256, nz=128, steps=20, use_t_dep=True, use_triton=True)
    
    print("\n" + "="*40)
    print("SUMMARY (512x256x128 Grid)")
    print(f"PyTorch (Const):   {res_base.elapsed_ms/20:.2f} ms/step, {res_base.max_vram_mb:.1f} MB")
    print(f"PyTorch (T-Dep):   {res_tdep.elapsed_ms/20:.2f} ms/step, {res_tdep.max_vram_mb:.1f} MB")
    print(f"Triton  (T-Dep):   {res_triton.elapsed_ms/20:.2f} ms/step, {res_triton.max_vram_mb:.1f} MB")
    
    speedup = res_tdep.elapsed_ms / res_triton.elapsed_ms
    vram_save = res_tdep.max_vram_mb - res_triton.max_vram_mb
    
    print("-" * 40)
    print(f"Speedup Triton vs PyTorch (T-Dep): {speedup:.2f}x")
    print(f"VRAM Saved: {vram_save:.1f} MB")
    
    diff_speed_pytorch = (res_tdep.elapsed_ms / res_base.elapsed_ms - 1) * 100
    print(f"PyTorch T-Dependency Overhead: {diff_speed_pytorch:.1f}%")
    print("="*40)
