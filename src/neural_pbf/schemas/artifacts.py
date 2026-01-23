from pydantic import BaseModel


class ArtifactConfig(BaseModel):
    """Configuration for artifact generation."""

    enabled: bool = True
    png_every_n_steps: int = 50
    html_every_n_steps: int = 250
    make_report: bool = True
    make_video: bool = False
    max_frames: int = 300
    downsample: int | None = None  # e.g., 2 or 4 for large grids
    vmax: float | None = None  # Max temperature for color scale consistency
    include_scan_overlay: bool = True
    save_raw: bool = False  # Save full temperature fields as .npy
