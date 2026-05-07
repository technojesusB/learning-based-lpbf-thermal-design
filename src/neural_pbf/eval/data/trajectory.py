"""Trajectory — an ordered sequence of Snapshots with shared configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neural_pbf.core.config import SimulationConfig
from neural_pbf.physics.material import MaterialConfig

from .snapshot import Snapshot


@dataclass
class Trajectory:
    """An ordered sequence of simulation snapshots sharing a common config.

    Attributes:
        snapshots: Ordered list of Snapshots, first = initial state.
        sim_cfg:   Simulation domain configuration.
        mat_cfg:   Material configuration.
        metadata:  Free-form dict for provenance tags (e.g. scan speed, power).
    """

    snapshots: list[Snapshot]
    sim_cfg: SimulationConfig
    mat_cfg: MaterialConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Snapshot:
        return self.snapshots[idx]
