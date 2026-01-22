
import logging
from pathlib import Path
from typing import Any
import glob

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    
try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None


from ..schemas.artifacts import ArtifactConfig
from .artifacts_base import ArtifactBuilder
from . import plots

logger = logging.getLogger(__name__)


class TemperatureArtifactBuilder(ArtifactBuilder):
    """Artifact builder for thermal simulation.

    Generates:
    - PNG heatmaps (Top Surface)
    - 3D Block Visualizations (GIF animated)
    - Plotly interactive HTMLs
    - Report HTML linking all assets
    """

    def __init__(self, cfg: ArtifactConfig):
        super().__init__()
        self.cfg = cfg
        self._png_paths: list[Path] = []
        self._html_paths: list[Path] = []
        self._block_paths: list[Path] = [] # Store paths for GIF
        self._comp_paths: list[Path] = [] # Store composite paths for GIF
        
        self._xt_buffer: list[np.ndarray] = []
        self._xt_times: list[float] = []
        
        self._dx: float = 1.0
        self._dy: float = 1.0
        self._dz: float = 1.0 # Need dz now
        self._length_unit: str = "m"
        
        # Determine global vmin/vmax? Hard to do online. 
        # We'll use per-frame or fixed if user provided (not in config yet).
        # Dynamic is fine for now.

    def on_run_start(self, run_meta: Any, out_dir: Path):
        super().on_run_start(run_meta, out_dir)
        if hasattr(run_meta, "dx"):
            self._dx = float(run_meta.dx)
        if hasattr(run_meta, "dy"):
            self._dy = float(run_meta.dy)
        if hasattr(run_meta, "dz"): # Ensure dz is captured
            self._dz = float(run_meta.dz)
            if self._dz == 0.0: self._dz = 1.0 # Avoid div0 or bad plots
        if hasattr(run_meta, "length_unit"):
            self._length_unit = str(run_meta.length_unit)

    def _extract_temperature(self, state: Any, reduce_3d: bool = True) -> np.ndarray:
        # (Same extraction logic as before)
        T = None
        if isinstance(state, dict):
            if "T" in state: T = state["T"]
            elif "temperature" in state: T = state["temperature"]
        elif isinstance(state, torch.Tensor):
            T = state
        elif hasattr(state, "T") and not callable(getattr(state, "T")): 
            T = state.T
        elif hasattr(state, "shape"):
            T = state

        if T is None:
            return np.zeros((10, 10))

        if hasattr(T, "cpu"):
            T = T.detach().cpu().numpy()
        elif hasattr(T, "numpy"):
            T = T.numpy()

        T = np.squeeze(T)
        
        # 3D handling logic
        if reduce_3d and T.ndim == 3:
             # Take surface max
             T = np.max(T, axis=-1)

        if self.cfg.downsample and self.cfg.downsample > 1:
            d = self.cfg.downsample
            if T.ndim == 2: T = T[::d, ::d]
            elif T.ndim == 3: T = T[::d, ::d, ::d]

        return T

    def on_snapshot(
        self, step_idx: int, state: Any, meta: dict[str, Any]
    ) -> list[Path]:
        generated: list[Path] = []

        if not self.cfg.enabled:
            return generated

        do_png = (self.cfg.png_every_n_steps > 0) and (step_idx % self.cfg.png_every_n_steps == 0)
        do_html = (self.cfg.html_every_n_steps > 0) and (step_idx % self.cfg.html_every_n_steps == 0)

        if not (do_png or do_html):
            return generated

        # Full Data
        T_full = self._extract_temperature(state, reduce_3d=False)
        T_surf = self._extract_temperature(state, reduce_3d=True)

        # 1. 3D Block Png (if 3D)
        if T_full.ndim == 3 and do_png:
             p_block = self.dirs["plots_png"] / f"step_{step_idx:06d}_block.png"
             self._save_3d_block(T_full, p_block, step_idx)
             self._block_paths.append(p_block)
             generated.append(p_block)

        # 2. Surface Png
        if do_png and plt:
            p_surf = self.dirs["plots_png"] / f"step_{step_idx:06d}_surf.png"
            self._save_surface(T_surf, p_surf, step_idx)
            self._png_paths.append(p_surf)
            generated.append(p_surf)
            
            # New: Cross-sections (if 3D)
            if T_full.ndim == 3:
                p_cross = self.dirs["plots_png"] / f"step_{step_idx:06d}_cross.png"
                self._save_cross_sections(T_full, p_cross, step_idx)
                generated.append(p_cross)
                
                # Option: Also save composite view
                p_comp = self.dirs["plots_png"] / f"step_{step_idx:06d}_composite.png"
                self._save_composite(T_full, p_comp, step_idx)
                self._comp_paths.append(p_comp)
                generated.append(p_comp)
                
                # New: Interactive Composite
                p_html_comp = self.dirs["plots_interactive"] / f"step_{step_idx:06d}_composite.html"
                self._save_plotly_composite(T_full, p_html_comp, step_idx)
                generated.append(p_html_comp)

        # 3. Interactive Plotly
        if do_html and go:
            p_html = self.dirs["plots_interactive"] / f"step_{step_idx:06d}.html"
            self._save_plotly(T_full, p_html, step_idx)
            # self._html_paths.append(p_html) # Optional: Don't flood report with htmls if many
            generated.append(p_html)
            
        # 4. XT Buffer
        if T_surf.ndim == 2:
            Ny = T_surf.shape[1]
            profile = T_surf[:, Ny // 2]
            self._xt_buffer.append(profile)
            # Time heuristic
            t_val = float(step_idx)
            if isinstance(state, (dict, object)) and not isinstance(state, torch.Tensor):
                 if hasattr(state, "t") and not callable(getattr(state, "t")): t_val = float(state.t)
            self._xt_times.append(t_val)

        return generated

    def _save_surface(self, T, path, step):
        if plt is None: return
        fig, ax = plt.subplots(figsize=(6, 5))
        plots.plot_surface_heatmap_mpl(
            ax, T, self._dx, self._dy, 
            title=f"Surface T - Step {step}", 
            unit='mm' if self._length_unit in ['m','mm'] else self._length_unit
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def save_material_overlay(self, T: np.ndarray, mask: np.ndarray, path: Path, step: int):
        """Save a composite plot of Temperature and Material Mask with discrete legend."""
        if plt is None:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Mask Plot ---
        # Phases: 0=Powder, 1=Liquid, 2=Solid
        # We need to ensure the mask reflects these values. 
        # The mask usually stores: 0=Powder, 1=Solid (Irreversible).
        # We might need to compute 'Liquid' dynamically from T > T_liquidus.
        # But 'mask' passed here is usually just the historical state.
        # Let's derive a display_mask:
        #   0: Powder (Mask=0 and T < T_liq) -- Actually Phase 0
        #   1: Liquid (T > T_liq) -- Phase 1
        #   2: Solid (Mask=1 and T < T_liq) -- Phase 2
        
        # Assuming T is already reduced to 2D
        T_screen = T.T
        M_screen = mask.T
        
        # Simple heuristic for liquidus (hardcoded or from meta? Config doesn't have it here easily)
        # We'll rely on the input 'mask' being just the irreversible state (0 or 1).
        # If we want to show Liquid, we need to know T_liquidus.
        # Ideally, we should pass phase field directly. 
        # For now, let's plot the binary mask (Powder/Solid) clearly.
        
        from matplotlib.colors import ListedColormap, BoundaryNorm
        
        # Define 3 phases just in case we can discern them, strictly sticking to inputs:
        # 0: Powder (Blue-ish or Grey)
        # 1: Solid (Orange/Red)
        cmap_mask = ListedColormap(['#cccccc', '#ff7f0e']) # Grey, Orange
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap_mask.N)
        
        im_mask = axes[0].imshow(M_screen, origin="lower", cmap=cmap_mask, norm=norm)
        axes[0].set_title(f"Material State (Step {step})")
        
        # Legend
        cbar = plt.colorbar(im_mask, ax=axes[0], ticks=[0, 1])
        cbar.ax.set_yticklabels(['Powder', 'Solid'])
        
        # --- Temperature Plot ---
        im_t = axes[1].imshow(T_screen, origin="lower", cmap="jet")
        plt.colorbar(im_t, ax=axes[1], label="T (K)")
        axes[1].set_title("Temperature Field")
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def _save_3d_block(self, T, path, step):
        if plt is None: return
        fig = plt.figure(figsize=(8, 6))
        plots.plot_3d_block_mpl_ax(
            fig.add_subplot(111, projection='3d'), 
            T, self._dx, self._dy, self._dz,
            title=f"3D Block - Step {step}",
            unit='mm'
        )
        plt.savefig(path, bbox_inches='tight', dpi=150) # Increased DPI for sharpness
        plt.close(fig)

    def _save_cross_sections(self, T, path, step):
        if plt is None: return
        fig = plt.figure(figsize=(15, 5))
        plots.plot_cross_sections(
            fig, T, self._dx, self._dy, self._dz,
            unit='mm',
            cmap='jet'
        )
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_composite(self, T, path, step):
        if plt is None: return
        fig = plt.figure(figsize=(12, 10))
        plots.plot_composite_thermal_view(
            fig, T, self._dx, self._dy, self._dz, step, unit='mm'
        )
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _save_plotly(self, T, path, step):
        if go is None: return
        if T.ndim == 3:
            fig = plots.plot_interactive_volume(T, self._dx, self._dy, self._dz, step)
        else:
            fig = plots.plot_interactive_heatmap(T, self._dx, self._dy, step)
        fig.write_html(str(path), include_plotlyjs="cdn")

    def _save_plotly_composite(self, T, path, step):
        if go is None or T.ndim != 3: return
        fig = plots.plot_interactive_composite(T, self._dx, self._dy, self._dz, step)
        fig.write_html(str(path), include_plotlyjs="cdn")
    
    def on_run_end(self, final_state: Any, meta: dict[str, Any]) -> list[Path]:
        generated = []
        
        # XT Diagram
        if self.cfg.enabled and self._xt_buffer:
             # Reuse buffer logic (could move to plots.py too but keeping here for simplicity)
             xt_path = self.dirs["plots_png"] / "xt_diagram.png"
             self._save_xt_diagram(xt_path)
             generated.append(xt_path)
        
        # Generate GIF from Block Plots
        if self._block_paths and imageio:
            gif_path = self.dirs["plots_png"] / "animation_3d.gif"
            self._create_gif(self._block_paths, gif_path)
            generated.append(gif_path)
            
        # Generate GIF from Composite Plots
        if self._comp_paths and imageio:
            gif_comp_path = self.dirs["plots_png"] / "animation_composite.gif"
            self._create_gif(self._comp_paths, gif_comp_path)
            generated.append(gif_comp_path)
            
        if self.cfg.make_report:
             report_path = self.dirs["report"] / "index.html"
             self._write_report(report_path, meta, gif_path if self._block_paths else None)
             generated.append(report_path)
             
        return generated

    def _save_xt_diagram(self, path: Path):
        # Legacy XT code logic, minimally adapted
        if plt is None or not self._xt_buffer: return
        XT = np.stack(self._xt_buffer, axis=0)
        times = np.array(self._xt_times)
        width_m = XT.shape[1] * self._dx
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use plot_api? simpler to just imshow here for XT specific extent
        # Extent logic
        scale = 1000.0 if self._length_unit == 'm' else 1.0 # assume m
        extent = [0, width_m * scale, times[0], times[-1]]
        
        im = ax.imshow(XT, origin="lower", aspect="auto", cmap="jet", extent=extent)
        plt.colorbar(im, ax=ax, label="T (K)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("t [s]")
        ax.set_title("XT Diagram")
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def _create_gif(self, image_paths: list[Path], out_path: Path):
        """Stitch images into GIF."""
        # Sort paths
        safe_paths = sorted(image_paths, key=lambda p: p.name)
        images = []
        for p in safe_paths:
            images.append(imageio.imread(p))
        
        # 10 fps default
        imageio.imwrite(out_path, images, duration=100, loop=0)
        logger.info(f"Saved GIF to {out_path}")

    def _write_report(self, path: Path, meta: dict, gif_path: Path | None):
        # Simple report
        content = ["<html><body><h1>Simulation Report</h1>"]
        if gif_path:
             content.append(f"<h2>3D Animation</h2><img src='../plots/png/{gif_path.name}' />")
        content.append("</body></html>")
        with open(path, "w") as f:
            f.write("".join(content))
