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

---

## 4. WSL & Windows Stability

Two additional issues appear specifically under WSL2 (Windows Subsystem for Linux) when running high-resolution multi-run dataset generation.

### 4.1 Time-Chunking (prevents GPU TDR timeouts)

**The Problem:** Windows monitors GPU activity via the Timeout Detection and Recovery (TDR) watchdog. If a single GPU kernel occupies the device for more than ~2 seconds, Windows forcibly resets the driver. High-diffusivity materials (e.g., Ti-6Al-4V) trigger many more CFL sub-steps per macro-step, making individual `step_adaptive` calls hit this limit during long exposure times (≥50 µs).

**The Solution:** The script now auto-detects WSL2 (via `/proc/version`) and enables chunked stepping automatically. Instead of calling `stepper.step_adaptive` once with the full `exposure_time`, the script iterates in chunks of at most **5 µs** each, with an explicit `torch.cuda.synchronize()` after every chunk:

```python
# _chunked_step: active when --wsl-safe is True (auto-enabled on WSL2)
t_remaining = exposure_time
while t_remaining > 0.0:
    dt = min(t_remaining, 5e-6)          # ≤ 5 µs per GPU call
    state = stepper.step_adaptive(state, dt_target=dt, ...)
    if torch.cuda.is_available():
        torch.cuda.synchronize()         # force CPU to wait for GPU completion
    t_remaining -= dt
```

**Why `cuda.synchronize()` is required:** PyTorch kernel launches are asynchronous — `step_adaptive` returns to Python the moment kernels are *queued*, not when they *finish*. Without a sync point, the while loop enqueues the next chunk's kernels before the GPU has finished the previous chunk. Windows TDR measures time from when a GPU packet is submitted until the GPU acknowledges completion; without sync, consecutive chunks appear as continuous uninterrupted GPU work and TDR fires. The synchronize call gives Windows the "breathing point" it needs to reset its TDR timer between slices.

**Auto-detection:** On WSL2, `--wsl-safe` is enabled automatically. The log will show:
```
WSL2 detected — enabling --wsl-safe automatically (pass --no-wsl-safe to opt out)
```

**Usage:**
```bash
# Automatic on WSL2 (no flags needed):
uv run experiments/generate_offline_dataset.py --runs 20

# Explicit on non-WSL2:
uv run experiments/generate_offline_dataset.py --wsl-safe --runs 20

# Opt out of auto-detection on WSL2 (advanced use only):
uv run experiments/generate_offline_dataset.py --no-wsl-safe --runs 20
```

### 4.2 HDF5 File Locking (prevents BlockingIOError)

**The Problem:** h5py uses POSIX advisory file locks by default. Under WSL2, when one worker subprocess exits and releases the lock, the next subprocess may encounter a `BlockingIOError: [Errno 11] Resource temporarily unavailable` before the kernel has fully released it.

**The Solution (two layers):**

1. **Environment variable** — set before `import h5py` in the script so the library never tries to acquire POSIX locks:
   ```python
   os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
   import h5py
   ```
   This is the primary fix. It is safe for single-writer-per-file access patterns (which the orchestrator enforces by running one worker at a time).

2. **Retry loop** — the `_open_hdf5` helper retries up to 3 times with a 0.5 s sleep on `OSError`, providing a safety net if the env-var approach is insufficient on a specific kernel version:
   ```python
   for attempt in range(3):
       try:
           return h5py.File(path, mode)
       except OSError:
           if attempt < 2:
               time.sleep(0.5)
           else:
               raise
   ```

### 4.3 Orchestrator Resilience (prevents cascade failure after TDR)

**The Problem (observed 2026-05-03):** When a TDR event reset the Windows GPU driver mid-run (Run 1 of 20), every subsequent subprocess also failed immediately with `RuntimeError: Found no NVIDIA driver on your system`. A single crash cascaded into a total loss of the entire 20-run batch.

**Root cause:** After a TDR reset, the CUDA driver needs time to fully re-initialize. The orchestrator was spawning the next subprocess immediately after the failed one exited, before the driver had recovered.

**The Solution (two mechanisms):**

**1. Post-failure GPU cooldown sleep** (`--post-failure-delay`, default: 30 s)

After any failed subprocess, the orchestrator sleeps before the next attempt, giving Windows time to complete the driver reset:

```bash
# Default: 30-second cooldown after each failure
uv run experiments/generate_offline_dataset.py --runs 20

# Reduce for faster iteration (risky on TDR-prone setups):
uv run experiments/generate_offline_dataset.py --post-failure-delay 10 --runs 20
```

The sleep is linearly scaled per retry attempt: the first retry waits 1×delay, the second waits 2×delay, etc.

**2. Per-run retry with linear backoff** (`--max-retries`, default: 2)

Before marking a run as permanently failed, the orchestrator retries it up to `--max-retries` times. This recovers from one-off TDR events without manual re-invocation:

```bash
# Default: up to 2 retries per run
uv run experiments/generate_offline_dataset.py --runs 20

# Disable retries (fail fast):
uv run experiments/generate_offline_dataset.py --max-retries 0 --runs 20
```

**3. Defensive sample-write guard**

The per-sample HDF5 write block is now wrapped in `try/except (RuntimeError, OSError)`. On a CUDA error mid-write, the worker logs the failure, aborts the current run cleanly (exits non-zero), and lets the orchestrator's retry logic handle recovery. This prevents silent partial writes and produces a deterministic exit code.

### 4.4 Updated Strategy Summary

| Strategy | Benefit | Cost |
| :--- | :--- | :--- |
| **Broadcasting** | Saves 90%+ VRAM on intermediates | Higher code complexity (`.view()` management) |
| **Manual Cleanup** | Prevents crashes over long runs | Minimal (slight overhead after each run) |
| **FP16 Storage** | 50% smaller datasets | Small precision loss in extreme ranges |
| **Time-Chunking (auto on WSL2)** | Prevents Windows GPU TDR resets | ~5% overhead from loop + sync bookkeeping |
| **HDF5 Lock Disable** | Eliminates `BlockingIOError` between workers | Safe only with one writer per file |
| **Post-failure delay (30 s)** | Lets GPU driver recover after TDR before next run | Increases wall-clock time after failures |
| **Per-run retry (×2)** | Recovers from one-off TDR without manual re-run | Linear backoff capped at `max_retries` attempts |
| **Sample-write guard** | Clean abort on mid-write CUDA error → triggers retry | None (defensive only) |
