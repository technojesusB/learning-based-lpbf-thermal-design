import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


from ..schemas.artifacts import ArtifactConfig
from .artifacts_base import ArtifactBuilder

logger = logging.getLogger(__name__)


class TemperatureArtifactBuilder(ArtifactBuilder):
    """Artifact builder for thermal simulation.

    Generates:
    - PNG heatmaps
    - Plotly interactive HTMLs (if available)
    - Report HTML linking all assets
    """

    def __init__(self, cfg: ArtifactConfig):
        super().__init__()
        self.cfg = cfg
        self._png_paths: list[Path] = []
        self._html_paths: list[Path] = []

    def _extract_temperature(self, state: Any) -> np.ndarray:
        """Extract temperature field from state, return as numpy array 2D/3D."""
        # Heuristic to find temperature in state
        # Supports dict with 'T', 'temperature', or just tensor
        T = None
        if isinstance(state, dict):
            if "T" in state:
                T = state["T"]
            elif "temperature" in state:
                T = state["temperature"]
            else:
                # Try first value?
                pass
        elif hasattr(state, "shape"):  # Tensor-like
            T = state

        if T is None:
            logger.warning("Could not extract temperature from state.")
            return np.zeros((10, 10))  # fallback

        if hasattr(T, "cpu"):
            T = T.detach().cpu().numpy()
        elif hasattr(T, "numpy"):
            T = T.numpy()

        # Handle dimensions
        T = np.squeeze(T)
        
        # Handle 3D -> 2D (take surface max or mid-slice)
        if T.ndim == 3:
            # Assuming [X, Y, Z] or [Z, Y, X]?
            # Usually simulation is [X, Y, Z]. Let's take max over Z for top-down view
            # or last slice. Let's assume standard [X, Y, Z] and Z is depth.
            # Taking max profile is safer for melt pool visualization.
            T = np.max(T, axis=-1)

        # Downsample if requested
        if self.cfg.downsample and self.cfg.downsample > 1:
            d = self.cfg.downsample
            T = T[::d, ::d]

        return T

    def on_snapshot(
        self, step_idx: int, state: Any, meta: dict[str, Any]
    ) -> list[Path]:
        generated = []

        if not self.cfg.enabled:
            return generated

        # Check cadences
        # Check cadences
        do_png = (self.cfg.png_every_n_steps > 0) and (
            step_idx % self.cfg.png_every_n_steps == 0
        )
        do_html = (self.cfg.html_every_n_steps > 0) and (
            step_idx % self.cfg.html_every_n_steps == 0
        )

        if not (do_png or do_html):
            return generated

        T = self._extract_temperature(state)

        # Scan overlay data
        scan_pos = meta.get("scan_pos")

        if do_png and plt:
            p = self.dirs["plots_png"] / f"step_{step_idx:06d}.png"
            self._save_png(T, p, step_idx, scan_pos)
            self._png_paths.append(p)
            generated.append(p)

        if do_html and go:
            p = self.dirs["plots_interactive"] / f"step_{step_idx:06d}.html"
            self._save_plotly(T, p, step_idx)
            self._html_paths.append(p)
            generated.append(p)

        return generated

    def _save_png(self, T: np.ndarray, path: Path, step: int, scan_pos: Any | None):
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            T.T, origin="lower", cmap="inferno"
        )  # .T to align with usual physics coords
        plt.colorbar(im, ax=ax, label="Temperature (K)")
        ax.set_title(f"Step {step}")

        if scan_pos is not None and self.cfg.include_scan_overlay:
            # Assumes scan_pos is iterable (x, y)
            try:
                # scale scan pos to grid if needed?
                # Assuming scan_pos is in grid units for simplicity here,
                # or real units. If real units, we need dx.
                # Let's just plot it if likely grid units.
                # Implementation dependent, keeping simple for now.
                if self.cfg.include_scan_overlay:
                    pass
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def _save_plotly(self, T: np.ndarray, path: Path, step: int):
        if go is None:
            return
        fig = go.Figure(data=go.Heatmap(z=T.T, colorscale="Inferno"))
        fig.update_layout(title=f"Temperature Field - Step {step}")
        fig.write_html(str(path), include_plotlyjs="cdn")

    def on_run_end(self, final_state: Any, meta: dict[str, Any]) -> list[Path]:
        if not self.cfg.make_report:
            return []

        report_path = self.dirs["report"] / "index.html"
        self._write_report(report_path, meta)
        return [report_path]

    def _write_report(self, path: Path, meta_dict: dict[str, Any]):
        """Generate a simple HTML report."""
        import os
        # Relative paths for links using os.path.relpath to handle sibling directories
        png_rel = [Path(os.path.relpath(p, path.parent)) for p in self._png_paths]
        html_rel = [Path(os.path.relpath(p, path.parent)) for p in self._html_paths]

        # Sort by step (filename)
        png_rel = sorted(png_rel, key=lambda p: p.name)
        html_rel = sorted(html_rel, key=lambda p: p.name)

        content = [
            "<html><head><title>Simulation Report</title>",
            "<style>body{font-family:sans-serif; max_width:1000px; "
            "margin:auto; padding:20px;}",
            "img{max_width:100%; border:1px solid #ddd; margin:5px;}",
            ".gallery{display:grid; grid-template-columns:"
            "repeat(auto-fill, minmax(300px, 1fr)); gap:10px;}",
            "</style></head><body>",
            "<h1>Simulation Report</h1>",
            "<h2>Metadata</h2>",
            "<pre>" + str(meta_dict) + "</pre>",
            "<h2>Snapshots (PNG)</h2>",
            "<div class='gallery'>",
        ]

        for p in png_rel:
            # assuming relative path works if report is in artifact/report/
            # and images in artifact/plots/
            # actually relative_to(path.parent) gives ../plots/...
            content.append(f"<div><p>{p.name}</p><img src='{p}' loading='lazy'/></div>")

        content.append("</div>")

        if html_rel:
            content.append("<h2>Interactive (Plotly)</h2><ul>")
            for p in html_rel:
                content.append(f"<li><a href='{p}'>{p.name}</a></li>")
            content.append("</ul>")

        content.append("</body></html>")

        with open(path, "w") as f:
            f.write("\n".join(content))
