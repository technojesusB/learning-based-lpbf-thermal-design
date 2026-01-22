from typing import Any

from pydantic import BaseModel, Field


class RunMeta(BaseModel):
    """Metadata about a simulation run."""

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    seed: int
    device: str
    dtype: str
    started_at: str  # isoformat
    dx: float
    dy: float
    dz: float
    dt: float
    grid_shape: list[int]
    material_summary: dict[str, Any] = Field(default_factory=dict)
    scan_summary: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
