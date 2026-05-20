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
- [Machine Learning Approach & Inverse Design](#machine-learning-approach--inverse-design)
- [Unified Benchmarking Suite](#unified-benchmarking-suite)
- [Surrogate Development & Ablation History](#surrogate-development--ablation-history)
- [Physics & Critical Assessment](#physics--critical-assessment)
- [Development & Usage](#development--usage)
- [Project Status & Milestones](#-project-status--milestones)

---

## Motivation
Laser Powder Bed Fusion (LPBF) is governed by complex, highly transient thermal processes.
Scan strategies strongly influence peak temperatures and cooling rates, which in turn affect
melt pool behavior and microstructure.

The goal of this project is **not** high-fidelity industrial prediction, but the development
of a **research-level prototype** that demonstrates how **machine learning, differentiable physics,
and inverse design** can be combined to optimize LPBF-inspired scan strategies.

For a detailed account of the design journey of the Diffusion Transformer based approach, implementation challenges (such as WSL2 timeout dogging and thermal hardware ceilings), and the complete ablation narrative, refer to the [DiT Flow Matching Development Journal](docs/journals/dit_flow_matching_journal.md).

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

## Machine Learning Approach & Inverse Design

This repository serves as a research testbed exploring multiple machine learning paradigms for LPBF surrogate modeling and inverse design:

### 1. Direct Differentiable Surrogates (Standard Optimization Loop)
* **Concept:** A direct neural network (e.g., U-Net or 3D CNN) acts as a regression model, mapping process parameters (laser power, scan speed, patterns) directly to the transient thermal field: $T = S_{\theta}(P, v, \text{pattern})$.
* **Inverse Design Workflow:** Because the neural surrogate $S_{\theta}$ is fully differentiable, we can define a target thermal field $T_{target}$ (e.g., a specific meltpool footprint or uniform cooling rate) and backpropagate the design loss directly to the input parameters:
  $$\mathcal{L}_{design} = \left\| S_{\theta}(P, v, \text{pattern}) - T_{target} \right\|_2^2$$
  Using gradient-based optimizers (Adam or L-BFGS), we directly update the scan strategy parameters without running slow physics simulations.
* **Physics Regularization (PINN):** The surrogate's predictions are constrained to satisfy the governing 3D heat equation by incorporating a PDE residual loss during training:
  $$\mathcal{L}_{pde} = \left\| \rho(T) c_p(T) \frac{\partial T}{\partial t} - \nabla \cdot (k(T) \nabla T) - Q \right\|_2^2$$

### 2. Physics-Informed Generative Surrogates (Flow Matching & DiT)
* **Concept:** To capture complex, multi-modal thermal trajectories without blurring gradients, we frame surrogate modeling as a generative task. We train a **3D Diffusion Transformer (DiT)** using **Conditional Flow Matching (CFM)** to model the time-evolution of the thermal field as a vector field.
* **Flow Matching Dynamics:** Instead of SDE noising, CFM learns a deterministic ODE vector field $v_{\theta}(T_t, t | Q, \text{params})$ that pushes a simple initial distribution (e.g., historical state or noise) to the physical target state. During inference, we roll out the 3D grid by integrating this vector field using standard ODE solvers (e.g., Euler or RK4), requiring far fewer function evaluations (NFEs) than traditional diffusion models.
* **Translation-Invariant 3D-RoPE:** Self-Attention query/key representations are rotated using **3D Rotary Positional Embeddings** mapped to patch-grid coordinates, preserving strict translation invariance across boundaries.
* **Triton-Fused PINN Pass:** To enforce the heat equation constraint $\mathcal{L}_{pde}$ during vector field integration without exploding memory, the PDE loss and its backward pass are calculated via custom fused Triton kernels. Relative Loss Balancing (ReLoBRaLo) dynamically scales the physics tax ($\mathcal{L}_{pde}$) against the data loss.
* **Inverse Design Integration:** In this paradigm, inverse design is achieved by optimizing the conditioning parameters $Q$ (the heat source distribution) through the integrated ODE trajectory, enabling advanced, path-based scan strategy optimization.


![DiT v5 Hero Test Samples](docs/assets/DiT-v5-hero-testsamples.png)
*Figure 1: Volumetric temperature predictions of the v5 Flow-Matching DiT surrogate against the high-fidelity simulator reference across test samples.*


![DiT v5 Hero Loss Curves](docs/assets/DiT-v5-hero-losses-detailed.png)
*Figure 2: ReLoBRaLo-balanced training and validation losses showing data (FM) and physical (PDE) loss components over 500 epochs.*

---

## Unified Benchmarking Suite

To evaluate and compare different surrogate architectures against the high-fidelity Triton physical solver, the repository includes a standardized benchmarking pipeline (`experiments/benchmark_suite.py`). It groups metrics into four key domains:

| Metric Group | Metric | Description |
| :--- | :--- | :--- |
| **Statistical Accuracy** | **MSE / MAE / $R^2$** | Standard pixel-wise temperature field reconstruction errors. |
| **Physical Geometry** | **Meltpool IoU** | Intersection-over-Union of the predicted vs. reference active melt pools ($T > T_{liquidus}$). |
| | **Meltpool Depth Error** | Discrepancy in thermal penetration depth along the Z-axis. |
| | **Hotspot Offset** | Euclidean distance between the predicted and reference peak temperature locations. |
| **Spectral Fidelity** | **FFT Power Spectrum** | Evaluates spatial frequency distributions to penalize high-frequency checkerboard and patch boundary artifacts. |
| **System Performance** | **Throughput (steps/s)** | Rollout speed of the surrogate compared to the solver. |
| | **Peak VRAM / Speedup** | Memory footprint and Triton kernel speedup ratios. |


![Benchmark System Comparison](docs/assets/benchmark_system_comparison.png)
*Figure 3: Benchmark throughput (steps/s) and peak VRAM usage comparing the raw PyTorch solver, optimized Triton solver, and surrogate inference rollouts.*


![Benchmark Physics Dashboard](docs/assets/benchmark_dashboard_physics.png)
*Figure 4: Physics validation metrics (meltpool dimensions, temperature profile along scan paths) audited against the physical reference solver.*


* **Automated Reporting:** The suite automatically logs experiment runs, artifacts, and comparative charts to **MLflow** and compiles a comprehensive Markdown report summarizing structural, physical, and hardware metrics.

---

## Surrogate Development & Ablation History

The surrogate architecture evolved through a baseline and five sequential DiT iterations so far:

* **Baseline (3D U-Net):** ConvNet-based regression model trained on local spatial patches. Fast convergence but lacked long-range global context.
* **DiT v1 (APE):** First generative attempt. 3D Diffusion Transformer with Absolute Positional Embeddings and $4^3$ patches. Suffered from severe checkerboard fragmentation and optimization instability.
* **DiT v2 (Sinusoidal):** Switched to fixed 3D sinusoidal embeddings and larger $8^3$ patches. Eliminated checkerboard noise but training stalled on an optimization plateau.
* **DiT v3 (3D-RoPE):** Pragmatic upgrade to 3D Rotary Position Embeddings, custom FlashAttention attention blocks, and a `OneCycleLR` scheduler. Drastically improved spatial detail and convergence.
* **DiT v4 (PINN-Reg):** Introduced physics-informed regularization (PINN) via de-normalizing PDE losses and ReLoBRaLo dynamic loss balancing in `bfloat16`.
* **v5 (Triton-PINN / Hero Run):** Flagship model utilizing 3D-RoPE, custom fused Triton GPU forward/backward kernels for direct PDE backpropagation, and full-scale dataset training (650 samples).

> [!NOTE]
> In the physical evaluation (Table 1) and inference performance (Table 4) benchmarks, **DiT v1** and **DiT v2** are omitted because these early models failed to achieve numerical convergence and could not generate physically coherent melt pools.

### Quantitative Ablation Results

To benchmark these iterations, all models were evaluated on the unified validation set across key physical, structural, and performance metrics:

#### 1. Physical Fidelity & Isotherm Accuracy
Measures the surrogate's capability to capture the thermodynamic geometry of the active melt pool ($T > T_{liquidus}$):

| Model Architecture | Features | Meltpool IoU (↑) | Depth Error (↓) | $T_{max}$ Error [K] (↓) | Hotspot Offset [vox] (↓) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (3D U-Net)** | ConvNet | $0.627 \pm 0.367$ | 8.6 vox | $1159.2 \pm 1595.3$ | $35.1 \pm 14.9$ |
| **DiT v3 (3D-RoPE)** | RoPE + p=4 | $0.804 \pm 0.254$ | 7.3 vox | $1347.3 \pm 1944.5$ | **$19.2 \pm 11.9$** |
| **DiT v4 (PINN-Reg)** | PINN-Reg | $0.737 \pm 0.303$ | 6.5 vox | $1375.5 \pm 2053.0$ | $20.7 \pm 12.5$ |
| **v5 (Triton-PINN)** | **Triton-PINN** | **$0.819 \pm 0.266$** | **$5.8 \text{ vox}$** | **$617.3 \pm 818.8$** | $22.9 \pm 17.1$ |

---

#### 2. Structural & Spectral Noise Analysis
Quantifies patch fragmentation and checkerboard high-frequency noise using Total Variation (TV) error and the Seam Index (Patch Boundary Discontinuity - PBD):

| Model Architecture | TV Error % (Noise) ↓ | PBD (Seam Index) ↓ | Status |
| :--- | :---: | :---: | :--- |
| *Ground Truth (Solver)* | *0.0 %* | *1.000* | *Reference* |
| **Baseline (3D U-Net)** | 204.8 % | 0.987 | High Jitter |
| **DiT v2 (Sinusoidal)** | 10298.4 % | 1.018 | Unstable (Sinusoidal, p=8) |
| **DiT v3 (3D-RoPE)** | 170.4 % | 1.336 | Structural Noise |
| **DiT v4 (PINN-Reg)** | 165.8 % | 1.684 | Patch Aliasing |
| **v5 (Triton-PINN)** | **59.9 %** | **1.382** | **Physics-Stabilized** |

---

#### 3. Training Cost & Hardware Footprint
Details resources required to train each version to convergence:

| Version | Duration (h) | Peak VRAM (GB) | GPU Util (%) | Final Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (3D U-Net)** | 1.39 | 7.03 | 16.1 % | 0.00715 | Baseline |
| **DiT v1 (APE)** | 2.94 | 16.68 | 58.7 % | 0.05824 | Unstable |
| **DiT v2 (Sinusoidal)** | 1.16 | 2.42 | 9.5 % | 0.60054 | Stalled |
| **DiT v3 (3D-RoPE)** | 5.02 | 4.83 | 40.4 % | 0.01592 | Cosine Dec. |
| **DiT v4 (PINN-Reg)** | 3.08 | 4.26 | 15.8 % | 0.04193 | PINN-Baseline |
| **v5 (Triton-PINN)** | **29.36** | **16.63** | **18.9 %** | **0.03472** | **Hero Run** |

---

#### 4. Inference Throughput & Latency
Rollout speed evaluated on a single GPU (representing speedups for inverse design optimization loops):

| Version | Latency [s/sample] | VRAM (GB) | Throughput | GPU Util (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (3D U-Net)** | 1.92 | ~2.8 | 17.2 s/s | ~18 % |
| **DiT v3 (3D-RoPE)** | **1.79** | ~2.1 | 19.4 s/s | ~24 % |
| **DiT v4 (PINN-Reg)** | **1.79** | ~2.1 | 19.1 s/s | ~25 % |
| **v5 (Triton-PINN)** | 1.86 | ~2.4 | **20.5 s/s** | **~45 %** |

---

## Physics & Critical Assessment

### Implemented Physics
- **Full 3D Transient Solver**: Solves the heat equation in 3D using finite differences.
- **Triton-Accelerated Kernels**: Custom High-performance kernels for 60M+ node grids.
- **Temperature-Dependent Parameters**: $k(T)$ and $c_p(T)$ modeled via per-material Lookup Tables (LUT).
- **Phase Transition**: Models the transition between Powder, Solid, and Liquid phases including latent heat effects.
- **Irreversible State**: Correctly handles the physical transformation from Powder to Solid.

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
Generate physically consistent 3D datasets for surrogate training:
```bash
# Smoke test (rapid validation)
uv run experiments/generate_offline_dataset.py --runs 2 --samples-per-run 50

# Full corpus generation
uv run experiments/generate_offline_dataset.py --runs 20 --samples-per-run 50
```

**2. Train Surrogate Model:**
```bash
# Train v5 Triton-PINN DiT model
uv run experiments/train_fm_dit_triton.py --h5 data/offline_dataset.h5
```

**3. Run Benchmarking Suite:**
```bash
# Evaluate surrogate models against solver
uv run experiments/benchmark_suite.py
```

- **Documentation**: See [Usage Guide](docs/usage.md) and [Physics Model](docs/physics.md).
- **Experiment Tracking**: Run `uv run mlflow ui` to view logs and artifact lineage.

---

## 📈 Project Status & Milestones

The project is structured around five core research pillars:

### Pillar 1: Physical Solver & Triton Core [COMPLETED]
- Implemented a 3D FDM transient solver with LUT-based material properties and phase transitions.
- Fused simulator operations into Triton GPU kernels, achieving ~10x speedups.

### Pillar 2: Differentiable Surrogates & Direct Inverse Design [COMPLETED]
- Developed a 3D U-Net forward model for direct regression.
- Integrated the surrogate into a gradient-based optimization loop to perform inverse design on scan paths.

### Pillar 3: Generative CFM & 3D Diffusion Transformers [COMPLETED]
- Implemented Conditional Flow Matching (CFM) for physical trajectory generation.
- Integrated 3D Diffusion Transformer (DiT) with 3D Rotary Positional Embeddings (3D-RoPE).

### Pillar 4: Fused Physics & Unified Benchmarking [COMPLETED]
- Built custom Triton forward/backward kernels for de-normalized PDE residual calculations.
- Integrated ReLoBRaLo dynamic loss scaling.
- Implemented a comprehensive evaluation suite (`experiments/benchmark_suite.py`) testing accuracy, geometry, performance, and spectral artifacts.

### Pillar 5: Multi-Material & Universal Generalization [ROADMAP]
- Train surrogates on multi-material datasets with randomized thermal properties.
- Transition from fixed-grid transformers to resolution-independent Fourier Neural Operators (FNO).

---

## Intended Outcome
Physics-regularized machine learning without full CFD/FEM pipelines, bridging mechanical engineering and modern ML via a scalable research prototype.
