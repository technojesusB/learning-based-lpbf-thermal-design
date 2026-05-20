# LPBF Thermal Surrogate Benchmarking Suite

The benchmarking suite has been refactored into a modular, production-ready library designed for rigorous physical evaluation and experiment tracking.

## Core Components

1.  **Unified Orchestrator (`experiments/benchmark_suite.py`):**
    *   A CLI-driven tool that manages model loading, data iteration, and metric collection.
    *   Supports flags for selecting models (`--models`), metric suites (`--metrics`), and overriding MLflow Run IDs (`--run-ids`).
    *   Automatically handles physical scaling ($2000 \cdot T_{norm} + 300$) and liquidus thresholding ($1600 K$).

2.  **Modular Metric Suites (`src/neural_pbf/eval/metrics/`):**
    *   `physics`: Meltpool geometry (IoU, Depth, Hotspot Offset) and thermal accuracy ($T_{max}$ error).
    *   `spectral`: High-frequency fidelity via Power Spectral Density (PSD) and Total Variation (TV).
    *   `system`: Training and inference efficiency (s/sample, parameters, convergence).

3.  **Automated Dashboards (`src/neural_pbf/eval/viz/`):**
    *   Generates high-fidelity PNG dashboards in `docs/assets/`.
    *   `benchmark_dashboard_physics.png`: Violin plots for IoU and boxplots for thermal residuals.
    *   `benchmark_system_comparison.png`: Comparison of training vs. inference performance.

## Execution

To run a full comparison between the Baseline and the Hero Run (v5):

```bash
uv run experiments/benchmark_suite.py \
    --models Baseline "Hero Run (v5)" \
    --metrics physics spectral system \
    --dataset data/offline_dataset_test.h5 \
    --save-raw
```

## Experiment Tracking (MLflow)

All benchmark runs are logged to the **"Model-Evaluation"** experiment. 
*   **Metrics:** Average IoU, Mean Thermal Error, and Inference Throughput are logged as scalars.
*   **Artifacts:** The final dashboards and raw CSV/NPZ files are saved for full reproducibility.

## Key Design Decisions

*   **Adaptive Loading:** The suite automatically detects `patch_size` (4 or 8) and model type (U-Net, DiT, RoPE, or Triton).
*   **Grid Meta-Data Fallback:** If a checkpoint lacks gitter metadata, the suite falls back to the standard $15\mu m$ spacing, ensuring backward compatibility.
*   **Persistence:** Raw metrics are saved to `docs/assets/raw_metrics/` before plotting to allow for future re-analysis without re-running expensive rollouts.
