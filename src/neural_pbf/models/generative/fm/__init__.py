"""Flow Matching generative surrogate for LPBF thermal prediction."""

from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import (
    compute_physics_residuum,
    fm_loss,
    interpolate,
    sample_noise,
    target_velocity,
)
from neural_pbf.models.generative.fm.velocity_net import VelocityNet

__all__ = [
    "FMConfig",
    "ConditioningEncoder",
    "VelocityNet",
    "sample_noise",
    "interpolate",
    "target_velocity",
    "fm_loss",
    "compute_physics_residuum",
]
