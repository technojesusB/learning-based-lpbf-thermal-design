# MLflow Integrations

The tracking pipeline supports MLflow as a backend for experiment tracking.

## Configuration
Enable MLflow by setting `backend="mlflow"` in `TrackingConfig`.

```python
tracking_cfg = TrackingConfig(
    enabled=True,
    backend="mlflow",
    experiment_name="lpbf-thermal",
    mlflow_tracking_uri="http://localhost:5000" # or local ./mlruns
)
```

## Environment Variables
The factory respects standard MLflow environment variables if not overridden in config:
- `MLFLOW_TRACKING_URI`: URI of the MLflow tracking server.
- `MLFLOW_EXPNAME`: (Custom) can be used if you extend config parsing.

## gracefully Degradation
If `mlflow` is not installed or the server is unreachable:
- By default (`strict=False`), the system falls back to a `NullTracker` (no-op) and logs a warning.
- Logic is resilient: simulation will likely continue even if logging fails.

## Artifact Logging
Artifacts generated locally (PNGs, HTMLs) are also logged to MLflow artifacts store if enabled.
- Takes advantage of `RunContext` batching.
