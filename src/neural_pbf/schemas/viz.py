from pydantic import BaseModel, Field


class GridConfig(BaseModel):
    enabled: bool = True
    which: str = "both"
    axis: str = "both"  # 'both', 'x', or 'y'
    linestyle: str = "--"
    alpha: float = 0.4
    color: str = "#666666"


class PlotConfig(BaseModel):
    """Base visual specifications for all research plots."""

    # Output
    dpi: int = 300
    figsize_default: tuple[int, int] = (12, 8)

    # Colors
    bg_figure: str = "#0a0a0a"
    bg_axis: str = "#0a0a0a"

    # Text
    font_size_title: int = 14
    font_size_label: int = 11
    font_size_tick: int = 9
    font_size_legend: int = 10
    font_weight_title: str = "bold"
    title_pad: int = 20

    # Grid
    grid: GridConfig = Field(default_factory=GridConfig)

    # Line styles
    line_width_main: float = 2.5
    line_width_sub: float = 1.5
    alpha_raw: float = 0.3
    alpha_stripplot: float = 0.6
    alpha_scatter: float = 0.7
    alpha_line: float = 0.8

    # Multi-Model Palette (Ordered for consistency)
    color_palette: list[str] = [
        "#e74c3c",  # 0: Red (Baseline)
        "#3498db",  # 1: Blue
        "#f1c40f",  # 2: Gold
        "#00FFC8",  # 3: Cyan (Hero)
        "#9b59b6",  # 4: Purple
        "#e67e22",  # 5: Orange
        "#2ecc71",  # 6: Green
        "#95a5a6",  # 7: Gray
    ]

    # Model Palette (Legacy mapping for named models)
    model_palette: dict[str, str] = {
        "Baseline": "#e74c3c",
        "DiT v2": "#95a5a6",
        "DiT v3": "#3498db",
        "DiT v4": "#f1c40f",
        "Hero Run (v5)": "#00FFC8",
    }

    # Component Colors
    color_train: str = "#2ecc71"
    color_val: str = "#e74c3c"
    color_fm: str = "#3498db"
    color_pde: str = "#e67e22"
    color_lambda: str = "#9b59b6"
    color_gt: str = "#ffffff"
    color_secondary: str = "#f39c12"  # Used for s/it lines etc.


class PhysicalBenchmarkConfig(PlotConfig):
    """Specifics for physical fidelity dashboards (IoU, T_max, etc)."""

    # Optimized for 2x3 layout
    font_size_title: int = 18
    font_size_label: int = 14
    font_size_tick: int = 11
    font_size_legend: int = 12

    figsize: tuple[int, int] = (25, 15)
    grid: GridConfig = Field(default_factory=lambda: GridConfig(axis="y"))

    marker_size: int = 20  # For scatter GT vs Pred
    stripplot_size: int = 4
    stripplot_jitter: float = 0.2

    line_width_whisker: float = 1.5
    line_width_median: float = 2.0


class SystemBenchmarkConfig(PlotConfig):
    """Specifics for GPU/Duration comparisons."""

    figsize: tuple[int, int] = (22, 6)
    grid: GridConfig = Field(default_factory=lambda: GridConfig(axis="y"))

    marker_size_scatter: int = 100
    marker_size_annotate: int = 8
    alpha_bar: float = 0.8


class SpatialConfig(PlotConfig):
    """Specifics for 2D/3D structure galleries (Surface/Depth slices)."""

    grid: GridConfig = Field(default_factory=lambda: GridConfig(enabled=False))
    cmap: str = "magma"
    title_pad_gallery: int = 10
    font_size_gallery: int = 10


class SpectralConfig(PlotConfig):
    """Specifics for frequency PSD analysis (aliasing/fidelity)."""

    figsize: tuple[int, int] = (16.7, 15)
    line_width_ref: float = 2.0
    alpha_aliasing: float = 0.5
    color_aliasing: str = "#f1c40f"


class ThemeRegistry(BaseModel):
    """Centralized registry for all plot themes."""

    base: PlotConfig = Field(default_factory=PlotConfig)
    physical: PhysicalBenchmarkConfig = Field(default_factory=PhysicalBenchmarkConfig)
    system: SystemBenchmarkConfig = Field(default_factory=SystemBenchmarkConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)


# Global theme instance
THEME = ThemeRegistry()
