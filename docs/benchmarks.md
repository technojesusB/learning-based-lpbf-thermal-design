# Solver Benchmarks & Performance Analysis

This document tracks the performance characteristics of the thermal solver across different architectures and grid sizes.

## 1. Baseline Performance (Vanilla PyTorch)
*Date: 2026-01-22*
*Hardware: NVIDIA GPU (CUDA)*
*Software: PyTorch 2.x, 3D Finite Difference Solver (Explicit Euler)*

### Comparison: Resolution Scaling

| Grid Size ($N_x \times N_y \times N_z$) | Total Cells | PyTorch (Const) | PyTorch (T-Dep) | Triton (T-Dep) |
| :--- | :--- | :--- | :--- | :--- |
| $256 \times 128 \times 64$ | ~2.1 Million | 62.65 ms | 69.68 ms | - |
| $512 \times 256 \times 128$ | ~16.8 Million | 1887.43 ms | 2160.78 ms | **316.96 ms** |

### Key Observations
1. **Geometric vs. Computational Scaling**: 
   - While the cell count increased by factor **8.0**, the PyTorch computation time increased by factor **~30.0**. 
   - **Triton Solution**: By using a fused kernel, we reduced the scaling penalty significantly. The Triton solver achieves a **6.8x speedup** over PyTorch for the 16.8M cell domain.
2. **Memory Efficiency**:
   - Triton reduced VRAM consumption by **~70%** (1764 MB -> 528 MB) by eliminating intermediate tensor fields.
3. **Temperature Dependency Overhead**:
   - Enabling T-dependent material properties adds ~14% overhead in PyTorch, but is virtually "free" within the Triton fused kernel compared to the standard PyTorch baseline.

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
