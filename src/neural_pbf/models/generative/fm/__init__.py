"""Flow Matching generative surrogate for LPBF thermal prediction."""

from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.dit import (
    DiTBlock,
    DiTBlockRoPE,
    VelocityDiT,
    VelocityDiTRoPE,
    apply_rope_3d,
    make_3d_sinusoidal_pos_embed,
    patch_center_coords_idx,
    sinusoidal_time_embedding,
)
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
    "DiTBlock",
    "DiTBlockRoPE",
    "VelocityDiT",
    "VelocityDiTRoPE",
    "make_3d_sinusoidal_pos_embed",
    "sinusoidal_time_embedding",
    "apply_rope_3d",
    "patch_center_coords_idx",
    "sample_noise",
    "interpolate",
    "target_velocity",
    "fm_loss",
    "compute_physics_residuum",
]
