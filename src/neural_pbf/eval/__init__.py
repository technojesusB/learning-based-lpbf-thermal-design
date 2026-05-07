"""neural_pbf.eval — surrogate evaluation framework."""
from neural_pbf.eval.data.hdf5_loader import load_trajectory, save_trajectory
from neural_pbf.eval.data.snapshot import Snapshot
from neural_pbf.eval.data.trajectory import Trajectory
from neural_pbf.eval.protocols import BaseStepper
from neural_pbf.eval.rollout.engine import RolloutEngine, RolloutResult
from neural_pbf.eval.schemas import EvalConfig, ProbeSpec, RolloutConfig

__all__ = [
    "BaseStepper",
    "EvalConfig",
    "ProbeSpec",
    "RolloutConfig",
    "RolloutEngine",
    "RolloutResult",
    "Snapshot",
    "Trajectory",
    "load_trajectory",
    "save_trajectory",
]
