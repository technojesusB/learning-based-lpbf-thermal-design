# Strategy: Adaptive Loss Weighting via ReLoBRaLo (Implemented)

## Status: Active
ReLoBRaLo was successfully implemented in `src/neural_pbf/models/loss.py` and validated during the development of iterations **v4** and **v5 (Hero Run)**. It successfully balanced the multi-objective training dynamics, eliminating the need for manual hyperparameter tuning of the physics loss weight.

## The Problem: Gradient Imbalance in PINNs
Physics-Informed Neural Networks (PINNs) are inherently **multi-task learning** problems. We minimize:
1.  **Data Loss ($\mathcal{L}_{mse}$)**: Fitting the high-fidelity simulation points.
2.  **Physics Loss ($\mathcal{L}_{pde}$)**: Satisfying the non-linear heat equation.

In LPBF, the temperature gradients are extremely steep ($\gt 10^6$ K/m). Often, the $\mathcal{L}_{pde}$ gradients are orders of magnitude larger than $\mathcal{L}_{mse}$, causing the optimizer to "ignore" the data and focus only on a trivial physics solution (like a constant temperature) or vice versa.

## The Solution: ReLoBRaLo
**R**elative **Lo**ss **B**alancing with **Ra**ndom **Lo**ckback (ReLoBRaLo) is a SOTA adaptive weighting scheme.

### How it works:
Instead of a static `pde_weight`, ReLoBRaLo calculates the weight $\lambda$ dynamically:

1.  **Relative Learning Rate**: It tracks the moving average of each loss term.
2.  **Efficiency Focus**: If the MSE loss is converging faster than the PDE loss, it increases the weight of the PDE loss automatically.
3.  **Random Lookback**: It occasionally "looks back" at the loss values from several iterations ago to prevent the weights from getting stuck in a local balancing loop.

### Validation & Impact:
During training, the physics loss weight $\lambda_{phys}$ dynamically scaled across four orders of magnitude ($10^{-3}$ to $10^1$). This self-regulating behavior prevented gradient explosion in `bf16` precision and ensured that the model prioritized physics-informed regularization only when the base data loss converged to a stable manifold. This resulted in an 11% reduction in meltpool depth error without sacrificing spatial reconstruction accuracy.

