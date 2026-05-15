"""LPBF benchmark library.

Public API (new):
    load_model_adaptive      -- load any model checkpoint, auto-detecting patch_size
    run_euler_rollout        -- Euler integration rollout for one batch
    compute_physics_metrics  -- per-sample IoU / depth / T_max_err / offset
    fetch_mlflow_data        -- fetch training stats from MLflow run IDs
    run_physics_sweep        -- full per-model sweep (no side effects)
    build_test_dataset       -- dataset factory (net / rope / triton)
    save_metrics_csv         -- persist DataFrame to CSV
    save_npz                 -- persist arrays to NPZ
    mlflow_log_paths         -- log file paths to MLflow via RunContext

Legacy API (deprecated — callers should migrate to the above):
    run_physical_fidelity_benchmark
    run_system_comparison
    run_spectral_benchmark
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Legacy re-exports (backwards compat — kept until callers are migrated)
# ---------------------------------------------------------------------------
from neural_pbf.eval.benchmark._legacy import (  # noqa: F401
    _model_palette,
    _rollout_sample,
    _run_model_benchmark,
    run_physical_fidelity_benchmark,
    run_spectral_benchmark,
    run_system_comparison,
)

# ---------------------------------------------------------------------------
# New public API
# ---------------------------------------------------------------------------
from neural_pbf.eval.benchmark.dataset import build_test_dataset  # noqa: F401
from neural_pbf.eval.benchmark.io import (  # noqa: F401
    mlflow_log_paths,
    save_metrics_csv,
    save_npz,
)
from neural_pbf.eval.benchmark.metrics_physics import (
    compute_physics_metrics,  # noqa: F401
)
from neural_pbf.eval.benchmark.mlflow_data import fetch_mlflow_data  # noqa: F401
from neural_pbf.eval.benchmark.model_loader import load_model_adaptive  # noqa: F401
from neural_pbf.eval.benchmark.rollout import run_euler_rollout  # noqa: F401
from neural_pbf.eval.benchmark.runner import run_physics_sweep  # noqa: F401
