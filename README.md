# Learning-Based LPBF Thermal Design

Learning-based inverse thermal design for LPBF-inspired scan strategies using physics-regularized neural surrogates.

This repository contains a **research prototype** that explores how machine learning,
differentiable physics, and gradient-based inverse design can be combined to optimize
laser scan strategies under simplified thermal assumptions.

---

## 🚀 View the Showcase

### Simulation Result (High-Fidelity)

![High Fidelity Simulation](docs/assets/simulation_hi_fid.gif)

*High-Fidelity Multi-Hatch Simulation (SS316L)*:
- **Grid Resolution**: $1024 \times 512 \times 128$ nodes (~67.1M voxels, ~1 $\mu$m spatial resolution).
- **Physical Domain**: $1.0 \times 0.5 \times 0.125$ mm.
- **Material**: SS316L (Temperature-dependent properties via LUT).
- **Performance**: ~3.9 s/step (on single GPU) using Triton kernels.

---

## Table of Contents
- [Motivation](#motivation)
- [Scope and Assumptions](#scope-and-assumptions)
- [Problem Definition](#problem-definition)
- [Forward Thermal Model](#forward-thermal-model)
- [Machine Learning Approach](#machine-learning-approach)
- [Surrogate Development & Ablation History](#surrogate-development--ablation-history)
- [Inverse Design](#inverse-design)
- [Physics & Critical Assessment](#physics--critical-assessment)
- [Development & Usage](#development--usage)
- [Roadmap](#roadmap--future-work)

---

## Motivation
Laser Powder Bed Fusion (LPBF) is governed by complex, highly transient thermal processes.
Scan strategies strongly influence peak temperatures and cooling rates, which in turn affect
melt pool behavior and microstructure.

The goal of this project is **not** high-fidelity industrial prediction, but the development
of a **research-level prototype** that demonstrates how **machine learning, differentiable physics,
and inverse design** can be combined to optimize LPBF-inspired scan strategies.

> [!TIP]
> **Research Journal**: Explore the detailed experimental journey, dataset engineering struggles, and the complete ablation narrative in the [DiT Flow Matching Development Journal](docs/journals/dit_flow_matching_journal.md).

---

## Scope and Assumptions
- 2D & 3D thermal domains
- Transient heat conduction as governing physics
- Physical material models (Powder, Solid, Liquid phases)
- No melt pool fluid flow, free surface, or keyholing (standard thermal assumption)
- Focus on differentiable methodology for inverse design

---

## Problem Definition
Given target thermal descriptors over a 2D layer, determine a structured scan strategy
(e.g. hatch lines or zig-zag patterns) that reproduces these descriptors as closely as possible.

### Control Variables (Scan Strategy)
- Scan pattern type: zig-zag / hatch lines (serpentine)
- Global orientation angle θ
- Hatch spacing h
- Scan speed v (global or per line)
- Laser power P (global or per line)

---

## Forward Thermal Model

### Governing Equation
Transient heat conduction in 3D:

$$\rho(T) c_p(T) \frac{\partial T}{\partial t} = \nabla \cdot (k(T) \nabla T) + Q(r, z, t)$$

Where:
- $\rho c_p$ is the volumetric heat capacity, incorporating Latent Heat of Fusion via the Enthalpy method.
- $\nabla \cdot (k \nabla T)$ represents Fourier conduction with temperature-dependent conductivity $k(T)$ modeled via Look-up Tables (LUTs).

### Heat Source Model
Each laser is modeled as a moving Gaussian heat input:

$$Q_i(x, y, z, t) = \frac{\eta P_i(t)}{2 \pi \sigma^2 L_z} \exp\left(-\frac{r^2}{2\sigma^2}\right)$$

---

## Machine Learning Approach

The repository implements a **Physics-Informed Generative Surrogate** utilizing Conditional Flow Matching (CFM) and a 3D Diffusion Transformer (DiT) backbone to model the thermal field progression.

### 1. Conditional Flow Matching (CFM)
Instead of standard Diffusion SDE paths, the surrogate is trained using Conditional Flow Matching. CFM learns deterministic, optimal-transport-like ODE trajectories. This significantly reduces the Number of Function Evaluations (NFEs) required for inference, allowing for rapid 3D field rollout.

### 2. 3D Diffusion Transformer (DiT)
The backbone architecture is a volumetric Diffusion Transformer. The $64^3$ spatial thermal grid is tokenized into non-overlapping patches (e.g., $4 \times 4 \times 4$ voxels), projected, and modulated via adaptive LayerNorm (`adaLN-zero`) conditioning on process parameters and material characteristics.

### 3. Translation-Invariant 3D-RoPE
To enforce physical coordinate consistency and spatial translation invariance, Query ($Q$) and Key ($K$) representations in the Self-Attention blocks are rotated using **3D Rotary Positional Embeddings (3D-RoPE)**. Coordinate frequencies are mapped to discrete patch grid indexes to prevent rotation collapse at micro-scales.

### 4. Physics-Informed Regularization (PINN)
The training is regularized using the residual of the governing non-linear Heat Equation:

$$\mathcal{L}_{pde} = \left\| \rho(T) c_p(T) \frac{\partial T}{\partial t} - \nabla \cdot (k(T) \nabla T) - Q \right\|^2$$

- **Loss Balancing (ReLoBRaLo):** Relative Loss Balancing with Random Lookback dynamically weights the generative data loss and physical PDE loss ($\mathcal{L}_{pde}$), preventing the physics constraint from dominating early training.
- **Triton Acceleration:** The PDE loss calculation and its backward pass are implemented as custom fused Triton GPU kernels. This maximizes arithmetic intensity, avoids PyTorch Autodiff memory overhead, and allows for direct physics-regularized backpropagation.

---

## Surrogate Development & Ablation History

The architecture was developed systematically through five main iterations:

1. **v1: U-Net Baseline:** A local convolutional model trained on localized patches, showing fast initial convergence but limited by local inductive bias.
2. **v2: DiT with Absolute Positional Embeddings (APE):** Tokenized transformer backbone. Suffered from spatial fragmentation and "checkerboard" artifacts due to coordinate entanglement.
3. **v3: DiT with Sinusoidal Embeddings:** Transitioned to 3D sinusoidal coordinate embeddings and $8^3$ patch sizes, resolving grid fragmentation but stalling on deep convergence.
4. **v4: DiT with 3D-RoPE & OneCycleLR:** Upgraded to Rotary Position Embeddings and OneCycle learning rate scheduler, yielding sharp melt pool boundaries.
5. **v5: Triton-PINN DiT (Hero Run):** Scaled to a full 650-sample corpus with 3D-RoPE, a custom Triton physics backend, and ReLoBRaLo dynamic loss scaling, establishing a stable, physically consistent surrogate.

---

## Inverse Design
Scan parameters are treated as trainable tensors.
Gradient-based optimization (Adam / L-BFGS) is used to minimize the objective by backpropagating through the surrogate model.

### Teacher–Student and Knowledge Distillation
- Teacher: higher-resolution, slower thermal model (this simulator)
- Student: lightweight surrogate for fast optimization

---

## Physics & Critical Assessment

### Implemented Physics
- **Full 3D Transient Solver**: Solves the heat equation in 3D using finite differences.
- **Triton-Accelerated Kernels**: Custom High-performance kernels for 60M+ node grids.
- **Temperature-Dependent Parameters**: $k(T)$ and $c_p(T)$ modeled via per-material Lookup Tables (LUT).
- **Phase Transition**: Models the transition between Powder, Solid, and Liquid phases including latent heat effects.
- **Irreversible State**: Correctly handles the physical transformation from Powder to Solid.

### Visualization Breakdown
- **Surface Plot (Top-Down)**: Visualizes the temperature distribution on the top surface.
- **3D Block Plot**: Volumetric representation for inspecting heat penetration depth.
- **Orthogonal Cross-Sections**: XY, XZ, and YZ planes showing internal thermal structure.

### Current Limitations:
- **Basic Boundary Conditions**: Limited to adiabatic or linear cooling losses; lacks radiation and gas flow convection.
- **Memory Overhead**: Large 3D domains are constrained by VRAM (partially solved by Triton).

---

## Development & Usage

### Setup & Prerequisites
This project requires [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Sync dependencies
uv sync
```

### Running Experiments

**1. Generate Offline Dataset:**
Generate the physically consistent 3D datasets for training:
```bash
# Smoke test (rapid validation)
uv run experiments/generate_offline_dataset.py --runs 2 --samples-per-run 50

# Full corpus generation
uv run experiments/generate_offline_dataset.py --runs 20 --samples-per-run 50
```

**2. Train Surrogate Model (v5 Triton-PINN Hero Run):**
```bash
uv run experiments/train_fm_dit_triton.py --h5 data/offline_dataset.h5
```

**3. Run Benchmarks:**
Evaluate surrogate predictions against solver rollouts across accuracy, performance, and physics audit metrics:
```bash
uv run experiments/benchmark_suite.py
```

- **Documentation**: See [Usage Guide](docs/usage.md) and [Physics Model](docs/physics.md).
- **Experiment Tracking**: Run `uv run mlflow ui` to view logs and artifact lineage.

---

## 📈 Project Status & Milestones

### Phase 7: Volumetric Flow Matching Surrogate [COMPLETED]
- Successfully integrated Conditional Flow Matching (CFM) framework.
- Evaluated U-Net convolutional baseline on spatial thermal patches.

### Phase 8: Scaling & Physics-Informed Transformers [COMPLETED]
- **3D-DiT Architecture:** Implemented 3D Diffusion Transformer with multi-head attention and adaptive LayerNorm conditioning.
- **Relative Coordinate Embedding:** Integrated axial-decoupled 3D Rotary Positional Embeddings (3D-RoPE).
- **Physics Regularization:** Implemented SI-unit normalized PDE loss with ReLoBRaLo dynamic scaling.
- **Hardware Cores & Triton:** Fused stencil computations into custom Triton forward/backward kernels for direct backpropagation of physics loss.
- **Validation & Benchmarks:** Standardized evaluation suite comparing structural, temporal, spectral, and physical (melt pool geometry) metrics.

---

## Roadmap & Future Work
1. **Universal Surrogate**: Training on multi-material HDF5 datasets with randomized properties.
2. **Advanced Architecture**: Transition from CNN-based surrogates to Fourier Neural Operators (FNO) for resolution-independent prediction.
3. **Inverse Design Loop**: Using the trained surrogate for gradient-based optimization of 2D energy density distributions.
4. **Physical Validation**: Comparison against open-source experimental datasets.

---

## Intended Outcome
Physics-regularized machine learning without full CFD/FEM pipelines, bridging mechanical engineering and modern ML via a scalable research prototype.
