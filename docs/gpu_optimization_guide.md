# GPU Optimization Guide: Broadcasting & Memory Management

This guide explains the technical strategies used to stabilize high-resolution thermal simulations (67M+ voxels) on consumer-grade GPUs, specifically focusing on Broadcasting and VRAM fragmentation.

## 1. Broadcasting: The VRAM Savior

### The Problem: Memory Explosion
In a 3D simulation with $1024 \times 512 \times 128$ voxels, a single float32 grid occupies **268 MB**. 
A typical laser intensity calculation involves several intermediate steps:
1. $(X - x_0)$
2. $(X - x_0)^2$
3. $(Y - y_0)^2$
4. $r^2 = (X - x_0)^2 + (Y - y_0)^2$
5. $\exp(-r^2 / 2\sigma^2)$

If each step creates a full 3D tensor, we allocate over **1.3 GB of temporary VRAM per simulation step**. At 10 steps per second, this causes massive pressure on the GPU memory allocator and often leads to driver timeouts (TDR) in WSL.

### The Solution: Broadcasting
Broadcasting allows performing operations on arrays with different shapes without replicating the data. Instead of full 3D grids, we use 1D "views":
- **X-View**: `[1, 1, 1, 1, 1024]`
- **Y-View**: `[1, 1, 1, 512, 1]`
- **Z-View**: `[1, 1, 128, 1, 1]`

**How it works:**
PyTorch conceptually "stretches" these 1D arrays during calculation within the GPU registers. The full 3D shape is only realized in the final result.
- **Memory Impact**: Intermediate tensors stay 1D or 2D (a few KB instead of 268 MB).
- **Speed**: Massive reduction in data movement between VRAM and GPU compute cores.

---

## 2. VRAM Fragmentation: The "Parking Lot" Analogy

### What is Fragmentation?
Fragmentation occurs when free memory is split into small, non-contiguous blocks. 
**Analogy**: Imagine a parking lot with 10 free spaces, but every single one is separated by a parked car. A large truck (your 3D tensor) cannot park there, even though 10 spaces are technically "free".

### Why it happens in our Pipeline:
1. **Caching Allocator**: PyTorch keeps memory "reserved" to speed up future allocations. Over thousands of steps, these reserved blocks can become scattered.
2. **WSL/Windows Watchdog (TDR)**: Windows monitors the GPU. If a process triggers too many large allocations or blocks the GPU for too long, Windows resets the driver to stay responsive. This manifests as a "GPU Crash".
3. **Accumulated "Ghost" Objects**: Between simulation runs, old `SimulationState` or `TimeStepper` objects might stay in memory if Python's Garbage Collector hasn't run yet.

### Prevention Strategies:
1. **Broadcasting**: Stop creating "truck-sized" intermediate tensors; use "car-sized" views instead.
2. **Explicit Cleanup**: At the end of each trajectory run, use:
   ```python
   del state, stepper, beam  # Remove references
   torch.cuda.empty_cache()  # Force PyTorch to release reserved memory
   ```
3. **Half Precision (FP16)**: Using `half()` for storage in HDF5 reduces the memory footprint of saved snapshots by 50% without significant loss in thermal accuracy.

---

## 3. Summary for Developers
| Strategy | Benefit | Cost |
| :--- | :--- | :--- |
| **Broadcasting** | Saves 90%+ VRAM on intermediates | Higher code complexity (`.view()` management) |
| **Manual Cleanup** | Prevents crashes over long runs | Minimal (slight overhead after each run) |
| **FP16 Storage** | 50% smaller datasets | Small precision loss in extreme ranges |
