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

ρ c_p ∂T/∂t = ∇·(k ∇T) + Q(x, y, z, t)

### Heat Source Model
Each laser is modeled as a moving Gaussian heat input:

Q_i(x, y, z, t) = η P_i(t) exp( -||[x,y,z] - r_i(t)||² / (2σ²) )

---

## Machine Learning Approach

### Surrogate Model
A neural surrogate (e.g. CNN, U-Net, or Neural Operator) learns the mapping from scan parameters
or heat input representations to thermal descriptors.

### Physics Loss
A physics residual loss enforces consistency with the governing heat equation:

L_phys = || ρ c_p ∂T/∂t - ∇·(k ∇T) - Q ||²

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
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

### Running Experiments
```bash
# SS316L High-Fidelity Multi-Hatch Simulation
uv run python experiments/ss316l_multi_hatch.py

# General 3D Simulation
uv run python experiments/run_3d.py
```

- **Documentation**: See [Usage Guide](docs/usage.md) and [Physics Model](docs/physics.md).
- **Experiment Tracking**: Run `uv run mlflow ui` to view logs and artifact lineage.

---

## Roadmap & Future Work
1. **Advanced Physics**: Implementation of radiation and convection at the top surface.
2. **Domain & Geometry**: Multi-layer domains (substrate + powder) and pre-heating.
3. **Performance**: Domain Decomposition for large-scale part simulation.
4. **Scan Design**: Support for arbitrary complex scan patterns (space-filling curves).

---

## Intended Outcome
Physics-regularized machine learning without full CFD/FEM pipelines, bridging mechanical engineering and modern ML via a scalable research prototype.
