# Strategy: Adaptive Loss Weighting via ReLoBRaLo (planned)

## The Problem: Gradient Imbalance in PINNs
Physics-Informed Neural Networks (PINNs) are inherently **multi-task learning** problems. We minimize:
1.  **Data Loss ($\mathcal{L}_{mse}$)**: Fitting the high-fidelity simulation points.
2.  **Physics Loss ($\mathcal{L}_{pde}$)**: Satisfying the non-linear heat equation.

In LPBF, the temperature gradients are extremely steep ($> 10^6$ K/m). Often, the $\mathcal{L}_{pde}$ gradients are orders of magnitude larger than $\mathcal{L}_{mse}$, causing the optimizer to "ignore" the data and focus only on a trivial physics solution (like a constant temperature) or vice versa.

## The Solution: ReLoBRaLo
**R**elative **Lo**ss **B**alancing with **Ra**ndom **Lo**ckback (ReLoBRaLo) is a SOTA adaptive weighting scheme.

### How it works:
Instead of a static `pde_weight`, ReLoBRaLo calculates the weight $\lambda$ dynamically:

1.  **Relative Learning Rate**: It tracks the moving average of each loss term.
2.  **Efficiency Focus**: If the MSE loss is converging faster than the PDE loss, it increases the weight of the PDE loss automatically.
3.  **Random Lookback**: It occasionally "looks back" at the loss values from several iterations ago to prevent the weights from getting stuck in a local balancing loop.

### Why this is the "Logic Reasoning" for Phase 2:
By first using **Optuna** to find the best *global* hyperparameters (Learning Rate, Network Depth), we establish a strong baseline. Then, by introducing **ReLoBRaLo**, we remove the need to manually tune the `pde_weight`. The model becomes a "self-tuning" physics engine that balances accuracy and thermodynamics automatically.

