# Status: Hyperparameter Optimization (HPO) Strategy (planned)

## Current State
We have designed a modular tuning pipeline based on **Optuna** and **MLflow**. The logic is currently "on hold" while we focus on building the **Material Zoo baseline**.

## Key Architecture Decisions
- **Modularity**: The optimization logic is built into a standalone script (`scripts/tune_surrogate.py`) that can be optionally called from the notebook.
- **Agnostic Objective**: The script suggests a `LossConfig` instead of a static weight. This ensures it won't break when we transition from `Static` to `ReLoBRaLo` weighting.
- **Pruning**: We've selected `MedianPruner` (or `Hyperband`) to kill underperforming trials early, saving GPU resources.
- **Metrics**: The tuning objective has been moved from "Training Loss" to **"Generalization Error"** (using a small unseen OOD validation set) to prevent overfitting.

## Advanced Optimization Goals
- **Generalization-First Tuning**: The objective for Optuna will not be the training loss but the **OOD Validation MAE** (evaluating on a completely unseen scan path). This ensures we find a "Universal" model, not one that overfits to a specific laser path.
- **Physical Sanity Score**: Every trial will log an `energy_conservation_error`. We will prioritize models that satisfy the heat equation residuals, even if their MSE is slightly higher, to ensure thermodynamic consistency.
- **Pareto Efficiency**: We will track the tradeoff between **Inference Latency (FPS)** and **Accuracy**. This allows us to select the "Sweet Spot" model that is fast enough for real-time digital twins while remaining scientifically accurate.

## Future Pick-up Points
1.  **ReLoBRaLo Implementation**: Once the baseline multi-material model is running, we will implement the adaptive weighting inside `src/neural_pbf/models/loss.py`.
2.  **Pareto Search**: Using Optuna to find the best tradeoff between **Prediction Accuracy (MAE)** and **Inference Speed (FPS)**.
3.  **Cross-Material Tuning**: Finding a single hyperparameter set that is "robustly sub-optimal" (good enough) for all materials in the library.
