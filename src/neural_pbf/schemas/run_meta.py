from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field

class RunMeta(BaseModel):
    """Metadata about a simulation run."""
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    seed: int
    device: str
    dtype: str
    started_at: str  # isoformat
    dx: float
    dy: float
    dz: float
    dt: float
    grid_shape: List[int]
    material_summary: Dict[str, Union[float, str]] = Field(default_factory=dict)
    scan_summary: Dict[str, Union[float, str]] = Field(default_factory=dict)
    notes: Optional[str] = None
