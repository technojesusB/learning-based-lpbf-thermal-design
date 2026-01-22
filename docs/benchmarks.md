# Solver Benchmarks & Performance Analysis

This document tracks the performance characteristics of the thermal solver across different architectures and grid sizes.

## 1. Baseline Performance (Vanilla PyTorch)
*Date: 2026-01-22*
*Hardware: NVIDIA GPU (CUDA)*
*Software: PyTorch 2.x, 3D Finite Difference Solver (Explicit Euler)*

### Comparison: Resolution Scaling

| Grid Size ($N_x \times N_y \times N_z$) | Total Cells | PyTorch (Const) | PyTorch (SS316L LUT) | Triton (SS316L LUT) |
| :--- | :--- | :--- | :--- | :--- |
| $256 \times 128 \times 64$ | ~2.1 Million | 4.80 ms | 7.90 ms | - |
| $512 \times 256 \times 128$ | ~16.8 Million | 41.65 ms | 62.81 ms | **5.91 ms** |

### Key Observations
1. **Unprecedented Performance**: 
   - The Triton fused kernel with realistic SS316L Lookup Tables achieves a **10.6x speedup** over the PyTorch implementation.
   - For a 16.8 Million cell domain, a single solver step takes only **~6 ms** on current hardware.
2. **Computational Efficiency**:
   - While PyTorch baseline scaled linearly with cell count (factor 8x), the memory-bound property lookups in PyTorch (LUT interpolation) added a **50% overhead**.
   - **Triton Advantage**: The fused kernel eliminates this overhead entirely by performing the LUT interpolation in registers during the same pass as the stencil calculation.
3. **Memory Savings**:
   - Triton reduces peak VRAM usage by **70%** (1.76 GB -> 0.53 GB) by fusing intermediate tensor fields (conductivity, melt fraction, indices).

## 2. Optimization Strategy

The goal is to break the non-linear scaling identified above using **Kernel Fusion**.

### Planned Improvements
- [ ] **Triton Fused Kernel**: Combine diffusion, heat source evaluation, and material property lookups into a single GPU pass.
- [ ] **VRAM Reduction**: By fusing operations, we eliminate the need for large intermediate tensors (like `k_eff` or `cp_eff` fields) in global VRAM.
- [ ] **Linear Scaling Target**: Attempt to bring the scaling factor back towards the theoretical 8.0x for the larger grid.

## 3. Benchmarking Methodology
Benchmarks are executed using `experiments/benchmark_solver.py` which utilizes `PerformanceTracker` from `src/neural_pbf/utils/profiling.py`.
- **Timing**: Measured via `torch.cuda.Event` for accurate GPU synchronization.
- **Memory**: Measured via `torch.cuda.max_memory_allocated()`.
