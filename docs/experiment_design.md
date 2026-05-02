# Experiment Design: Physics-Regularized Thermal Surrogates

This document outlines the systematic experiments planned to optimize and validate neural surrogates for LPBF thermal prediction.

## Overview
The goal is to develop a surrogate model that is:
1.  **Stable**: No exploding gradients over long sequences (autoregressive stability).
2.  **Physics-Informed**: Satisfies the 3D heat conduction PDE with high fidelity.
3.  **Generalizable**: Performs well on unseen scan paths and power levels (OOD).

---

## Experiment 1: Strategy Benchmark (Direct vs. Residual)
*   **Objective**: Compare absolute temperature mapping vs. multi-fidelity correction.
*   **Hypothesis**: The Residual strategy will have higher OOD robustness because it is anchored to a physically consistent (though coarse) prior.
*   **Design**:
    *   **Baseline**: Direct UNet-3D (Inputs: $T_t, Q$).
    *   **MF-UNet**: Residual UNet-3D (Inputs: $T_t, Q, T_{lf}$).
    *   **Params**: 300 epochs, $\lambda=0.1$, `base_channels=16`.
*   **Success Metric**: Lower MAE on a 280W diagonal scan path (held-out data).

## Experiment 2: Physics Regularization Sensitivity ($\lambda$-Sweep)
*   **Objective**: Quantify the trade-off between Data MSE (accuracy) and PDE Residual (physicality).
*   **Hypothesis**: There is a "sweet spot" for $\lambda$ beyond which the model sacrifices too much accuracy to satisfy the PDE.
*   **Design**:
    *   Sweep $\lambda \in \{0.0, 0.01, 0.1, 0.5, 1.0\}$.
    *   Monitor `mse_loss` vs `pde_loss` curves.
*   **Success Metric**: Achieving PDE residual $< 10^{-4}$ while maintaining MAE $< 30K$.

## Experiment 3: Resolution & Kernel Optimization (The Triton Step)
*   **Objective**: Transition from toy $32^3$ grids to production-relevant resolutions.
*   **Hypothesis**: Larger domains will require higher model capacity (deeper UNet) to maintain the same spatial accuracy.
*   **Design**:
    *   **Scale**: $32^3 \to 64^3 \to 128^3$.
    *   **Backend**: Enable `use_triton=True` for physics loss computation to ensure batch-processing speed.
*   **Success Metric**: Inference time per step $< 5ms$ for a $64^3$ domain.

## Experiment 4: Cross-Material Generalization
*   **Objective**: Validate if the surrogate can learn material-agnostic physics.
*   **Design**:
    *   Train on SS316L.
    *   Inference on Ti-6Al-4V (using different $k(T)$ and $c_p(T)$ in the PDE loss).
*   **Success Metric**: Meaningful prediction of melt pool shape differences between materials without retraining.
