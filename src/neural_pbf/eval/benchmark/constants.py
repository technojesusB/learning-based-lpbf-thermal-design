"""Canonical constants for the LPBF benchmark suite.

All physics thresholds, normalisation factors, and default configuration
values are defined here. Importing from this module is the single source
of truth — never hard-code these elsewhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physics constants & normalisation (must match FMConfig priors)
# ---------------------------------------------------------------------------
T_LIQUIDUS: float = 1600.0  # K — liquidus threshold for meltpool IoU / depth
T_REF: float = 2000.0  # K — normalisation range  (T_phys = T_norm * T_REF + T_AMBIENT)
T_AMBIENT: float = 300.0  # K — ambient / min temperature

# ---------------------------------------------------------------------------
# Rollout & sampling
# ---------------------------------------------------------------------------
N_EULER_STEPS: int = 25  # Euler integration steps for all models
PATCH_SIZE: int = 64  # Spatial patch edge length (voxels)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME: str = "Model-Evaluation"
MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

# ---------------------------------------------------------------------------
# Known training run IDs for system-stats fetching
# ---------------------------------------------------------------------------
TRAINING_RUN_IDS: dict[str, str] = {
    "Baseline": "30ba4110bfd048d3a269da05ce338f8d",
    "DiT v1": "c59105c3d3df489ba45cad204f00a5e6",
    "DiT v2": "2966b5be6bbd4e11b294aea8dbcad193",
    "DiT v3": "2705801ae220403aa491d62d9338aa53",
    "DiT v4": "64deb4f83825415ea2ba4b0029419968",
    "Hero Run (v5)": "99db9e4556354836a3747e7bc36fa880",
}
