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
- 2D powder bed layer (single-layer abstraction)
- Transient heat conduction as governing physics
- Synthetic / toy data only
- No melt pool fluid flow, free surface, or keyholing
- Focus on methodology rather than predictive accuracy

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
```bash
# Run the stateful time-resolved experiment
uv run python experiments/stateful_time_resolved.py
```
This will generate artifacts in `artifacts/run_<timestamp>/`.

**Notebook:**
Open `notebooks/01_view_states.ipynb` to run the experiment interactively and visualize the thermal states (Temperature, Cooling Rate, etc.) directly in the notebook.


### Running Checks

The following commands are defined in `pyproject.toml` and mimic the CI pipeline:

```bash
# Linting (Ruff) - check only
uv run lint

# Linting (Ruff) - auto-fix issues
uv run ruff check --fix .

# Formatting Check (Ruff)
uv run format

# Type Checking (Pyright)
uv run typecheck

# Tests (Pytest)
uv run test

# Run tests with coverage
uv run test -- --cov=neural_pbf
```

### Continuous Integration

CI is hosted on GitHub Actions:
- **Main Pipeline**: Runs on PRs and main push. Uses `uv` for speed and consistency.
- **Packaging Check**: Verifies that the package can be installed via `pip` (ensures `uv` didn't produce something exotic).
- **Release**: Automatically builds and creates a GitHub Release on `v*` tags with artifacts attached.

---


## Extensions and Future Work
- Multi-laser scan strategies
- Separate lasers for melting vs. thermal conditioning
- Patch-based / domain-decomposed surrogates
- 2.5D (thin-slab) or local 3D moving-window models
- Learned closure terms for unresolved physics

---

## Intended Outcome
This project demonstrates:
- Physics-regularized machine learning without full CFD/FEM pipelines
- Gradient-based inverse design for LPBF-inspired processes
- A scalable research prototype bridging mechanical engineering and modern ML

It is not intended as an industrial LPBF simulator.
