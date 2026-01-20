# Tracking Architecture

The simulation tracking pipeline is designed to be future-proof, robust, and minimally intrusive. It separates concerns between:
1.  **Tracking Core**: Managing the lifecycle of experiments and logging to backends (MLflow, etc.).
2.  **Diagnostics**: Calculating metrics, checking stability, and computing "proxy losses".
3.  **Artifacts**: Generating visualizations (PNG, HTML) and reports.
4.  **RunContext**: The single integration point that coordinates these components.

## Core Components

### ExperimentTracker
Defined in `neural_pbf.tracking.base`, this protocol defines the standard API for any tracking backend:
- `start_run()`, `end_run()`
- `log_metrics()`, `log_params()`
- `log_artifact()`, `log_text()`

### DiagnosticsRecorder
Located in `neural_pbf.diagnostics.recorder`, this component:
- Computes per-step metrics from simulation state (e.g., Min/Max T, Energy, L2 Norm).
- Performs stability checks (NaN/Inf detection, large gradients).
- Raises errors if `strict` mode is enabled and thresholds are breached.

### ArtifactBuilder
Located in `neural_pbf.viz.artifacts_base`, this abstract base class defines how complex artifacts are generated. The `TemperatureArtifactBuilder` implementation handles:
- Static PNG plots (Matplotlib).
- Interactive HTML plots (Plotly).
- Simulation Reports (HTML).

### RunContext
The glue code in `neural_pbf.tracking.run_context`. The simulation loop only interacts with `RunContext`.

```python
ctx = RunContext(tracking_cfg, artifact_cfg, diagnostics_cfg, run_meta, out_dir)
ctx.start()

for step in range(steps):
    # ... physics ...
    ctx.on_step_start(step, state)
    metrics = ctx.log_step(step, state)
    ctx.maybe_snapshot(step, state)

ctx.end(final_state)
```

## Directory Structure
Artifacts are stored in a canonical structure:
```
artifacts/
  <run_name>/
    config/             # Saved configs
    diagnostics/        # Diagnostic logs/plots
    plots/
      png/              # Static heatmaps
      interactive/      # Plotly HTMLs
    report/             # Summary report.html
    video/              # MP4 animations (optional)
    states/             # Raw PyTorch state dumps
    checkpoints/        # Reserved for training
```
