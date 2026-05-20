# LPBF Research Visualization Specifications

This document defines the "Gold Standard" for all visualization assets in the LPBF thermal surrogate project. All evaluation scripts and library functions must adhere to these specifications to ensure research-grade consistency.

## 1. Spatial Galleries (General Specs)
All multi-sample or temporal galleries (Dashboard, Evolution, Test Gallery) follow these unified design rules to ensure perfect alignment in reports:

- **Theme:** `plt.style.use('dark_background')`
- **Layout:** 2 rows (Surface/Depth) x 10 columns.
- **Figsize:** `(22, 7)`
- **DPI:** `300` (High-Resolution for publication).
- **Cmap:** `magma`, `vmin=0`, `vmax=1`.
- **Label Positioning:** 
    - Use `fig.text(0.01, ...)` for labels outside the axes.
    - Use `plt.subplots_adjust(left=0.05)` to create space for labels.
- **Labels:** 
    - 'SURFACE' (Blue: `#3498db`) on the left margin, vertical, bold, `fontsize=14`.
    - 'DEPTH' (Orange: `#e67e22`) on the left margin, vertical, bold, `fontsize=14`.
- **Axes:** `axis('off')` for all image plots.

---

## 2. Standard Loss Evolution (v1, v2, v3)
- **Reference Script:** `scratch/plot_loss_overlay.py`
- **Files:** `DiT-v1/v2/v3-losses.png`, `FM-Baseline-losses.png`
- **Specs:**
    - **Theme:** `dark_background` with custom overrides:
        - `figure.facecolor`: `#121212` (Deep black)
        - `axes.facecolor`: `#1e1e1e` (Dark gray)
        - `grid.color`: `#333333`
    - **Figsize:** `(10, 6)`
    - **Raw Lines (Background):** Alpha `0.3`, linewidth `1`.
        - Train: `#2e7d32` (Dark green) | Val: `#c62828` (Dark red)
    - **MA-5 Lines (Foreground):** Linewidth `2.5`.
        - Train: `#2ecc71` (Vibrant green) | Val: `#e74c3c` (Vibrant red)
    - **Scaling:** `yscale('log')`
    - **Grid:** `True, linestyle='--', alpha=0.2`
    - **Legend:** Facecolor `#1e1e1e`, edgecolor `#333333`
    - **DPI:** `120`

## 3. Detailed Physics Loss Analytics (v4 & v5)
- **Reference Script:** `scratch/reproduce_v4_plot.py`
- **Files:** `DiT-v4-losses-detailed.png`
- **Specs:**
    - **Figsize:** `(14, 12)` (Two vertical panels, shared X-axis)
    - **Panel 1 (Losses):**
        - **Total Train Loss (MA-5):** Yellow (`#f1c40f`), linewidth `2`, `zorder=5`
        - **FM Data Loss (MA-5):** Blue (`#3498db`), alpha `0.8`
        - **PDE Residual (MA-5):** Red (`#e74c3c`), alpha `0.8`
        - **Validation Loss (MA-5):** Light gray (`#ecf0f1`), dashed (`--`), linewidth `2`, alpha `0.9`
        - **Scaling:** `log`
    - **Panel 2 (Lambda):**
        - **Curve:** Purple (`#9b59b6`), linewidth `2.5`
        - **Fill:** Area under curve with alpha `0.2`, color `#9b59b6`
        - **Scaling:** `log`
    - **DPI:** `200`

## 4. Physical Fidelity Dashboard
- **Reference Script:** `scratch/run_final_physics_benchmark.py`
- **Files:** `physical_fidelity_benchmark_detailed.png`
- **Specs:**
    - **Palette:** Baseline (`#95a5a6`), v1 (`#e74c3c`), v2 (`#3498db`), v3 (`#2ecc71`), v4+ (`#9b59b6`)
    - **Figsize:** `(20, 15)` (4 Panels: Violin, Scatter, Box, Strip)
    - **Ideal Line (Depth Plot):** Yellow (`#f1c40f`), dashed, linewidth `2`
    - **Violin Plot:** `inner="quart", hue="Model"`
    - **Scatter:** `alpha=0.6, s=60`
    - **DPI:** `200` with `pad=4.0`

## 5. System Comparison
- **Reference Script:** `scratch/plot_system_metrics.py`
- **Files:** `system_comparison.png`
- **Specs:**
    - **Figsize:** `(15, 6)` (2 Panels)
    - **Panel 1 (Duration/Speed):**
        - **Bars (Duration):** Blue (`#3498db`), alpha `0.6`
        - **Line (Speed):** Red (`#e74c3c`), marker `o`, size `8`, linewidth `2`
    - **Panel 2 (Resource Footprint):**
        - **Scatter:** `cmap='plasma'`, edgecolors='white', linewidth `1`, size `150`
        - **X-axis:** `[0, 100] %` | **Y-axis:** `[0, 16384] MB (16GB)`
        - **Limit Line (16GB):** Red, dotted (`:`), alpha `0.5`
    - **DPI:** `120`

---

## Spatial Gallery Use Cases

### 6. Training Image Gallery (`gallery_evolution`)
- **Reference:** `src/neural_pbf/eval/viz/spatial.py` -> `gallery_evolution()`
- **Data:** 1 GT sample + 9 snapshots from different epochs.
- **Titles:** Column headers with `fontsize=10` (e.g., "Ep 100").
- **Specs:** See Section 1.

### 7. Test Gallery View (`gallery_test`) / Final Dashboard
- **Reference:** `src/neural_pbf/eval/viz/spatial.py` -> `gallery_test()` and `scratch/evaluate_v5_full_analysis.py`
- **Data:** 5 different samples (GT vs. Pred pairs).
- **Titles:** 'GT' and 'PRED' in gray above columns, `fontsize=10`, `pad=10`.
- **Specs:** See Section 1.
