# Learning-Based LPBF Thermal Design

Welcome to the **Learning-Based LPBF Thermal Design** wiki!

This project explores machine learning and differentiable physics for optimizing laser scan strategies in Laser Powder Bed Fusion (LPBF).

## Documentation

<!-- AUTO-GENERATED from docs/ index — do not edit this section manually -->

### Core
- **[Usage](usage.md)**: How to run experiments, generate datasets, and use the codebase.
- **[Formulation](formulation.md)**: Mathematical formulation of the thermal model and optimization problem.
- **[Physics Model](physics.md)**: Governing equations, boundary conditions, and phase transitions.

### Surrogate & ML Pipeline
- **[Offline Dataset Generation](offline_dataset_generation.md)**: HDF5 pipeline schema, domain randomization, and material zoo strategy (Phase 8).
- **[Experiment Design](experiment_design.md)**: Experimental design for universal surrogate training.
- **[ReLoBRaLo Strategy](relobralo_strategy.md)**: Relative loss balancing via residual-aware loss scheduling.
- **[Resolution Trap Strategy](resolution_trap_strategy.md)**: Curriculum approach for escaping coarse-grid local minima.

### Tracking & Observability
- **[Tracking Architecture](tracking_architecture.md)**: Details on the tracking and state management architecture.
- **[MLflow Backend](mlflow_backend.md)**: How experiments are tracked using MLflow.
- **[Diagnostics & Metrics](diagnostics_metrics.md)**: Explanation of the diagnostics and metrics used.
- **[Artifacts & Reports](artifacts_and_reports.md)**: Structure of generated artifacts and reports.
- **[Benchmarks](benchmarks.md)**: Performance benchmarks for the Triton-accelerated kernels.

### Materials Reference
- **[Materials Gallery](materials/index.md)**: Thermophysical properties for 11 LPBF alloys.

<!-- END AUTO-GENERATED -->

## Overview from README

*(See the main repository [README](https://github.com/technojesusB/learning-based-lpbf-thermal-design) for the most up-to-date project overview)*

### Motivation
LPBF is governed by complex, highly transient thermal processes. Scan strategies strongly influence peak temperatures and cooling rates. This project aims to optimize these strategies using ML.

### Scope and Assumptions
- 2D powder bed layer
- Transient heat conduction
- Synthetic data
- Focus on methodology

## Quick Links
- [GitHub Repository](https://github.com/technojesusB/learning-based-lpbf-thermal-design)
- [Issues](https://github.com/technojesusB/learning-based-lpbf-thermal-design/issues)
