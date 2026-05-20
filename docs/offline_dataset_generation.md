# High-Fidelity Offline Dataset Generation (Phase 8)

## The Strategy: "Diversity over Quantity"
The goal of the offline dataset is to transition from a single-material surrogate to a **Universal Thermal Surrogate** that generalizes across the entire LPBF material space. To achieve this on consumer-grade hardware, we employ a "Quality-over-Quantity" strategy.

## 1. Domain Randomization (Material Zoo)
Instead of training on discrete material points, we treat material properties as a continuous probability space.
*   **Base Presets**: We start from 5 core physical anchors: `SS316L`, `Ti-6Al-4V`, `IN718`, `AlSi10Mg`, and `CuCrZr`.
*   **Property Perturbation**: For every simulation run, properties ($k$, $c_p$, $\rho$, $L$) are randomly scaled by **±10-20%**.
*   **LUT Scaling**: Temperature-dependent Look-Up Tables (LUTs) are scaled globally to preserve the physical "shape" of the material's thermal response while varying the absolute magnitude.
*   **Result**: The surrogate learns the underlying PDE relationships rather than memorizing specific alloy signatures.

## 2. Complex Scan Paths (Stress-Testing)
To prevent the model from overfitting to simple linear hatches, we introduce "Stress-Test" trajectories:
*   **Curved Hatches**: Sine-wave modulated paths that force the model to handle rotating temperature gradients.
*   **Spiral Patterns**: Centripetal scans to test heat accumulation in confined zones.
*   **Island Strategies**: Patch-based scanning with frequent "jumps" to capture rapid transient cooling and re-heating.
*   **Acceleration Modeling**: Random variation of `exposure_time` and `point_distance` to simulate laser scanner dynamics and varying energy density.

## 3. The "Aluminium Bottleneck" Management
High-diffusivity materials like Aluminium require extremely small internal sub-steps for stability in explicit solvers.
*   **Adaptive Sampling**: For expensive high-k materials, we run fewer trajectories but extract **10x more snapshots** per run (200+ samples).
*   **Modular Appending**: Data is generated in batches. Fast materials (Steel/Titanium) are processed first to enable early training, while expensive materials are appended to the HDF5 archive later.

## 4. Monitoring & Dataset Metrics
Every generation run is documented with a sidecar `.metrics.json` file:
*   **`s/it`**: Real-world compute cost per macro-step.
*   **`substeps_total`**: Numerical complexity of the run.
*   **Path Preview**: A `path_run_XXX.png` is generated for visual verification of the scan strategy.

## 5. Storage Efficiency & Data Stack (HDF5)
*   **Lazy Loading**: Datasets are streamed from disk using `h5py` to keep RAM usage minimal.
*   **FP16 Compression**: Fields are stored in half-precision with GZIP compression.
*   **The Sample Stack**: Each HDF5 group (`sample_XXXXXX`) contains:
    *   **`T_in`**: Input temperature field [K].
    *   **`Q`**: Laser heat source field [W/m³].
    *   **`T_target`**: Ground truth temperature field after $dt$ [K].
    *   **`T_lf`**: Low-fidelity reference field (optional).
    *   **`mask`**: Phase mask field (0=Powder, 1=Solid/Liquid).
    *   **Attributes**: Randomized parameters used for this specific step:
        *   Material: `k_s`, `cp`, `rho`, `L`, etc. (perturbed values).
        *   Laser: `power`, `sigma`, `eta`, `depth`.
        *   Timing/Path: `exposure_time`, `point_distance`, `x`, `y`.
    *   **Material LUTs**: Temperature-dependent Look-Up Tables (`T_lut`, `k_lut`, `cp_lut`) are stored as datasets to allow exact reconstruction of the temperature-dependent physics context during training.

## 6. WSL2 & eGPU Execution Workflow

Due to the intensive nature of high-resolution simulations on eGPU/WSL2 setups, follow this specific execution pattern to prevent driver timeouts (TDR).

### Execution Command
Run the orchestrated simulation with auto-detected WSL-safety and MLflow diagnostics:

```bash
nohup uv run experiments/generate_offline_dataset.py \
    --materials "SS316L,Ti64,IN718" \
    --runs 20 \
    --out data/universal_offline_dataset.h5 \
    --nx 512 --ny 256 --nz 64 \
    --samples-per-run 50 \
    --mlflow \
    --viz > dataset_gen.log 2>&1 &
```

### Monitoring & Diagnostics
Track progress and identify bottlenecks (Bus saturation vs. Physics load):

1. **Terminal Log**: `tail -f dataset_gen.log`
2. **MLflow UI**: `./scripts/mlflow_ui.sh`
   - Check the **`lpbf-dataset-gen-diagnostics`** experiment.
   - Monitor `time_sim`, `time_transfer`, and `time_io`.

### Troubleshooting
- **CUDA unknown error**: A TDR reset occurred. Check if `time_io` spiked (Bus saturation). Ensure `TdrDelay` is set to 60s in Windows.
- **No NVIDIA driver**: Driver lost after crash. Run `wsl --shutdown` in PowerShell and restart.
- **BlockingIOError (Errno 11)**: Handled automatically by `HDF5_USE_FILE_LOCKING=FALSE` and retry logic.
