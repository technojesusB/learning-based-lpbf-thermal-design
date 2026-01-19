from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import torch

from lpbf.schemas.state import SnapshotState


@dataclass
class StateRecorder:
    keys: List[str] = field(default_factory=lambda: ["T", "E_acc", "t_since", "cooling_rate"])
    times: List[float] = field(default_factory=list)
    event_idxs: List[int] = field(default_factory=list)
    snaps: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self):
        for k in self.keys:
            self.snaps[k] = []

    @torch.no_grad()
    def add(self, t: float, event_idx: int, maps: Dict[str, torch.Tensor]):
        self.times.append(float(t))
        self.event_idxs.append(int(event_idx))
        for k in self.keys:
            x = maps[k]
            if x.ndim == 4:
                x = x[0]  # [1,H,W]
            self.snaps[k].append(x.detach().float().cpu())

    def to_snapshot_state(self) -> SnapshotState:
        return SnapshotState(
            t=torch.tensor(self.times, dtype=torch.float32),
            event_idx=torch.tensor(self.event_idxs, dtype=torch.int64),
            **{k: torch.stack(v, dim=0) for k, v in self.snaps.items()},
        )
