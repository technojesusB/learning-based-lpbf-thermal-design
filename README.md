# Learning-Based LPBF Thermal Design

Learning-based inverse thermal design for LPBF-inspired scan strategies using physics-regularized neural surrogates.

This repository contains a **research prototype** that explores how machine learning,
differentiable physics, and gradient-based inverse design can be combined to optimize
laser scan strategies under simplified thermal assumptions.

---

## Table of Contents
- [Motivation](#motivation)
- [Scope and Assumptions](#scope-and-assumptions)
- [Problem Definition](#problem-definition)
- [Control Variables (Scan Strategy)](#control-variables-scan-strategy)
- [Forward Thermal Model](#forward-thermal-model)
- [Target Thermal Descriptors](#target-thermal-descriptors)
  - [Peak Temperature Map](#peak-temperature-map)
  - [Peak Temperature and Cooling Rate](#peak-temperature-and-cooling-rate)
- [Optimization Problem](#optimization-problem)
- [Constraints and Regularization](#constraints-and-regularization)
- [Machine Learning Approach](#machine-learning-approach)
  - [Surrogate Model](#surrogate-model)
  - [Physics Loss](#physics-loss)
- [Inverse Design](#inverse-design)
- [Teacher–Student and Knowledge Distillation](#teacher–student-and-knowledge-distillation)
- [Extensions and Future Work](#extensions-and-future-work)
- [Intended Outcome](#intended-outcome)
- [Documentation](#documentation)
  - [Usage Guide](#usage-guide)
  - [Physics Model & Numerics](#physics-model-numerics)
- [Development](#development)
  - [Testing](#testing)

---

## Documentation

- [Usage Guide](docs/usage.md)
- [Physics Model & Numerics](docs/physics.md)
- [Project Wiki](https://github.com/technojesusB/learning-based-lpbf-thermal-design/wiki) (Synced with `docs/`)

## Development

### Testing
This project uses `pytest`. To run the unit and integration tests:
```bash
uv run pytest
```
Tests cover:
- Unit correctness (mm vs m conversion)
- Energy conservation checks
- Physics gradient operators
- Simulation pipeline stability

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

---

## Control Variables (Scan Strategy)
- Scan pattern type: zig-zag / hatch lines (serpentine)
- Global orientation angle θ
- Hatch spacing h
- Scan speed v (global or per line)
- Laser power P (global or per line)
- Optional smooth power ramps at line start and end

---

## Forward Thermal Model

### Governing Equation
Transient heat conduction in 2D:

ρ c_p ∂T/∂t = ∇·(k ∇T) + Q(x, y, t)

### Heat Source Model
Each laser is modeled as a moving Gaussian heat input:

Q_i(x, y, t) = η P_i(t) exp( -||[x,y] - r_i(t)||² / (2σ²) )

For multiple lasers:
Q_total = Σ_i Q_i

---

## Target Thermal Descriptors

### Peak Temperature Map
T_max(x, y) = max_t T(x, y, t)

Target: T_max*(x, y)

---

### Peak Temperature and Cooling Rate
To introduce temporal structure without enforcing full trajectories, a cooling-rate descriptor
is added.

1. Determine time of peak temperature:
   t_max(x, y) = argmax_t T(x, y, t)

2. Define local cooling rate:
   R(x, y) ≈ [ T_max(x, y) - T(x, y, t_max + Δt) ] / Δt

Targets:
- T_max*(x, y)
- R*(x, y)

---

## Optimization Problem
The inverse design problem is formulated as:

min_u  ||T_max(u) - T_max*||²
     + β ||R(u) - R*||²
     + constraint penalties

where u denotes the scan strategy parameters.

---

## Constraints and Regularization
- Power bounds: P_min ≤ P(t) ≤ P_max
- Speed bounds: v_min ≤ v(t) ≤ v_max
- Smoothness penalties on P(t) and v(t)
- Energy or total scan time regularization
- (Optional) multi-laser separation and scheduling constraints

---

## Machine Learning Approach

### Surrogate Model
A neural surrogate (e.g. CNN, U-Net, or Neural Operator) learns the mapping from scan parameters
or heat input representations to thermal descriptors.

### Physics Loss
A physics residual loss enforces consistency with the governing heat equation:

L_phys = || ρ c_p ∂T/∂t - ∇·(k ∇T) - Q ||²

The full training loss is:

L = L_data + λ L_phys

---

## Inverse Design
Scan parameters are treated as trainable tensors.
Gradient-based optimization (Adam / L-BFGS) is used to minimize the objective by backpropagating
through the surrogate model.

---

## Teacher–Student and Knowledge Distillation
- Teacher: higher-resolution, slower thermal model
- Student: lightweight surrogate for fast optimization
- Knowledge distillation transfers thermal behavior from teacher to student

This enables rapid inverse design iterations without repeated expensive simulations.

---

## Development & CI

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and task execution, ensuring parity between local development and CI.

### Prerequisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```bash
# Sync dependencies (creates .venv based on uv.lock)
uv sync
```

### Running Experiments

You can run the main simulation experiment either via the command line or using the provided notebook.

**Command Line:**
The repository includes several experiment scripts for different use cases:

```bash
# General 3D Simulation (Single Track)
uv run python experiments/run_3d.py

# High-Fidelity Bidirectional Hatch Pattern (Used for the GIF below)
uv run python experiments/hatch_pattern.py

# Fidelity Comparison (Low-res vs. High-res)
uv run python experiments/compare_fidelity.py
```

Each run will generate artifacts (plots, interactive HTMLs, and logs) in the `artifacts/` directory.

#### Experiment Tracking (MLflow)
The project is integrated with [MLflow](https://mlflow.org/) for tracking parameters, metrics, and artifact lineage.
- **Local Dashboard**: Run `uv run mlflow ui` to view the experiment dashboard.
- **More Info**: See the [Project Wiki](https://github.com/technojesusB/learning-based-lpbf-thermal-design/wiki) for detailed MLflow setup and usage.

**Notebooks:**
- `notebooks/01_view_states.ipynb`: Basic state visualization.
- `notebooks/02_verify_physics_and_viz.ipynb`: Advanced 3D visualization, cross-sections, and physics validation.

## Current State & Results

The simulator now supports full 3D transient thermal analysis with automated artifact generation and advanced visualization.

### Simulation Result (High-Fidelity)

![High Fidelity Simulation](docs/assets/simulation_hi_fid.gif)

*3D Transient Thermal Analysis:*
- **Solver**: Dense grid ($512 \times 256 \times 128$ nodes, ~16.7M cells), physical domain of $1.0 \times 0.5 \times 0.25$ mm.
- **Physics**: Realistic material conductivity and latent heat effects for Stainless Steel / Ti64.
- **Capabilities**: Captures high-frequency thermal gradients and accurate melt pool morphology.

---

### Visualization Breakdown

- **Surface Plot (Top-Down)**: Visualizes the temperature distribution on the top surface, highlighting the laser's path and the immediate thermal footprint.
- **3D Block Plot**: Provides a volumetric representation of the temperature field, allowing for the inspection of heat penetration depth and internal thermal gradients.
- **Orthogonal Cross-Sections**: XY, XZ, and YZ planes are extracted to show the internal structure of the thermal field, critical for understanding melt pool morphology and cooling rates at different depths.
- **Phase State Overlay**: (Coming soon) Enhanced plotting to distinguish between Powder, Solid, and Liquid phases with clear boundaries.

---

## Critical Assessment

### What the simulator can do now:
- **Full 3D Transient Solver**: Solves the heat equation in 3D using finite differences.
- **Differentiable Physics**: All operations are implemented in PyTorch, enabling backpropagation for inverse design.
- **Phase Transition**: Models the transition between Powder, Solid, and Liquid phases including latent heat effects.
- **Irreversible State**: Correctly handles the physical transformation from Powder to Solid.
- **Automated Artifacts**: Generates high-quality 3D visualizations, cross-sections, and interactive logs for every run.

### Current Limitations:
- **Temperature-Independent Parameters**: Material properties ($k, c_p$) are currently constant values rather than functions of temperature.
- **Basic Boundary Conditions**: Limited to adiabatic or linear cooling losses; lacks radiation and gas flow convection.
- **Memory Overhead**: Large 3D domains are constrained by VRAM due to PyTorch's memory management (to be solved by custom CUDA kernels).
- **Domain Uniformity**: Currently assumes a homogeneous material block (to be expanded to substrate+powder layer systems).


### Continuous Integration

CI is hosted on GitHub Actions:
- **Main Pipeline**: Runs on PRs and main push. Uses `uv` for speed and consistency.
- **Packaging Check**: Verifies that the package can be installed via `pip` (ensures `uv` didn't produce something exotic).
- **Release**: Automatically builds and creates a GitHub Release on `v*` tags with artifacts attached.

---


## Roadmap & Future Work

The following milestones are planned to increase physical fidelity and scalability:

### 1. Advanced Physics & Materials
- **Temperature-Dependent Parameters**: Transition from constant material properties to $T$-dependent thermal conductivity $k(T)$ and heat capacity $c_p(T)$.
- **Phase State Refinement**: Improved tracking and visualization of phase transitions, including latent heat effects and irreversible powder-to-solid transformation.
- **Surface Boundary Conditions**: Implementation of radiation and convection (argon flow) at the top surface.

### 2. Domain & Geometry
- **Customizable Domains**: Setup for multi-layer domains, including a base substrate material with a thin powder layer on top.
- **Pre-heating**: Initialization of the domain with a prescribed base temperature to simulate build-plate heating.

### 3. Scalability & Performance
- **Custom CUDA Kernels**: Fused kernels for the forward solver to drastically reduce memory overhead and enable simulation of significantly larger domains.
- **Domain Decomposition**: Implementation of "local-high-fidelity / far-field-low-fidelity" coupling to simulate large parts without sacrificing detail near the melt pool.
- **Surrogate-Guided Optimization**: Leveraging the neural surrogate to optimize scan patterns and parameters across large parts.

### 4. Scan Pattern Design
- **Complex Scan Strategies**: Support for arbitrary scan patterns beyond simple hatches, including space-filling curves and multi-laser strategies.

---

## Intended Outcome
This project demonstrates:
- Physics-regularized machine learning without full CFD/FEM pipelines
- Gradient-based inverse design for LPBF-inspired processes
- A scalable research prototype bridging mechanical engineering and modern ML

It is not intended as an industrial LPBF simulator.
