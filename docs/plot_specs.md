# LPBF Research Visualization Specifications

This document describes the visualization assets in the LPBF thermal surrogate project as the code produces them. The code is authoritative: `THEME` in `src/neural_pbf/schemas/viz.py` and the parameters of each plot function. Some functions read `THEME`, others hard-code their values; each section says which.

**Saving and DPI:** A function with a `savepath` or `output_path` saves at the DPI listed in its section. `log_figure()` in `src/neural_pbf/eval/viz/logging.py` saves again at its own `dpi` (default `150`) with `bbox_inches="tight"`, logs the file to the tracker, and then deletes it. The training scripts in `experiments/` log loss panels and galleries this way, so those logged files use `150`.

## THEME Values

Shared by all themes unless overridden below (`PlotConfig`):

- **Output:** DPI `300`, `figsize_default=(12, 8)`
- **Background:** Figure and axis `#0a0a0a`
- **Fonts:** Title `14` (bold, pad `20`), label `11`, tick `9`, legend `10`
- **Grid:** Both major and minor, both axes, `--`, alpha `0.4`, `#666666`
- **Lines:** Main `2.5`, sub `1.5`
- **Alphas:** Raw `0.3`, strip plot `0.6`, scatter `0.7`, line `0.8`
- **Palette (`color_palette`):** `#e74c3c`, `#3498db`, `#f1c40f`, `#00FFC8`, `#9b59b6`, `#e67e22`, `#2ecc71`, `#95a5a6`
- **Named models (`model_palette`):** Baseline `#e74c3c`, DiT v2 `#95a5a6`, DiT v3 `#3498db`, DiT v4 `#f1c40f`, Hero Run (v5) `#00FFC8`
- **Components:** Train `#2ecc71`, Val `#e74c3c`, FM `#3498db`, PDE `#e67e22`, Lambda `#9b59b6`, GT `#ffffff`, secondary `#f39c12`

| Theme | Overrides |
|---|---|
| `THEME.base` | None |
| `THEME.physical` | Fonts `18/14/11/12`, `figsize=(25, 15)`, y-grid, `marker_size=20`, `stripplot_size=4`, `stripplot_jitter=0.2`, whisker `1.5`, median `2.0` |
| `THEME.system` | `figsize=(22, 6)` (no plot reads it, see Section 5), y-grid, `marker_size_scatter=100`, `marker_size_annotate=8`, `alpha_bar=0.8` |
| `THEME.spatial` | Grid off, `cmap="magma"`, `title_pad_gallery=10`, `font_size_gallery=10` |
| `THEME.spectral` | `figsize=(16.7, 15)`, `line_width_ref=2.0`, `alpha_aliasing=0.5`, `color_aliasing=#f1c40f` |
| `THEME.uq` | See Section 12 |

---

## 1. Spatial Galleries (General Specs)
`gallery_evolution()` and `gallery_test()` in `src/neural_pbf/eval/viz/spatial.py` share this layout:

- **Theme:** `plt.style.context('dark_background')`
- **Layout:** 2 rows (Surface/Depth). `gallery_test()` always has 10 columns; `gallery_evolution()` has 1 GT column plus up to 9 prediction columns.
- **Slices:** Surface is the last depth slice (XY). Depth is the mid-Y slice (XZ).
- **Figsize:** `(22, 7)`
- **DPI:** `dpi` parameter, default `300`, saved with `bbox_inches=None`.
- **Margins:** `plt.subplots_adjust(left=0.06, bottom=0.08, top=0.90, right=0.98)`
- **Image scale:** `vmin=0`, `vmax=1`, `origin="lower"`.
- **Labels:**
    - 'SURFACE' (Blue: `#3498db`) at `fig.text(0.03, 0.71, ...)`, vertical, bold, `fontsize=14`.
    - 'DEPTH' (Orange: `#e67e22`) at `fig.text(0.03, 0.27, ...)`, vertical, bold, `fontsize=14`.
- **Suptitle:** Optional, `y=0.96`, `fontsize=16`, bold.

---

## 2. Standard Loss Evolution
- **Reference:** `src/neural_pbf/eval/viz/losses.py` -> `loss_panel()` without `fm_losses`/`pde_losses`
- **Files:** `loss_panel.png` when logged by the training scripts in `experiments/`; otherwise the given `savepath`.
- **Specs:**
    - **Theme:** `dark_background`; figure and axis `THEME.base` (`#0a0a0a`)
    - **Figsize:** `(10, 6)`, or `(10, 8)` with a second panel when `lambda_hist` is given
    - **Raw Lines (Background):** Alpha `0.3`, linewidth `1`.
        - Train: `#2e7d32` (Dark green) | Val: `#c62828` (Dark red)
    - **MA-5 Lines (Foreground):** Linewidth `2.5`.
        - Train: `#2ecc71` | Val: `#e74c3c`
    - **Scaling:** `yscale('log')`
    - **Grid:** `which='both', linestyle='--', alpha=0.4, color='#666666'`
    - **Labels:** X `fontsize=10`; Y `fontsize=11`, bold
    - **Legend:** Upper right, facecolor `#0a0a0a`, edgecolor `#666666`
    - **Lambda Panel:** `#9b59b6`, linewidth `2`, fill alpha `0.2`, log scale
    - **DPI:** `120`, `bbox_inches="tight"`

## 3. Detailed Physics Loss Analytics
- **Reference:** `src/neural_pbf/eval/viz/losses.py` -> `loss_panel()` with both `fm_losses` and `pde_losses`
- **Files:** The given `savepath`.
- **Specs:**
    - **Figsize:** `(14, 12)` (Two vertical panels, shared X-axis)
    - **Panel 1 (Losses):**
        - **Total Train Loss (MA-5):** Yellow (`#f1c40f`), linewidth `2`, `zorder=5`
        - **FM Data Loss (MA-5):** Blue (`#3498db`), linewidth `1.5`, alpha `0.8`
        - **PDE Residual (MA-5):** Orange (`#e67e22`), linewidth `1.5`, alpha `0.8`
        - **Validation Loss (MA-5):** Light gray (`#ecf0f1`), dashed (`--`), linewidth `2`, alpha `0.9`
        - **Scaling:** `log`
    - **Panel 2 (Lambda):**
        - **Curve:** Purple (`#9b59b6`), linewidth `2`
        - **Fill:** Area under curve with alpha `0.2`, color `#9b59b6`
        - **Scaling:** `log`
    - **DPI:** `200`, `bbox_inches="tight"`

## 4. Physical Fidelity Dashboard
- **Reference:** `src/neural_pbf/eval/viz/physical_dashboard.py` -> `plot_physical_dashboard()`
- **Files:** `benchmark_dashboard_physics.png` (from `experiments/benchmark_suite.py`)
- **Specs:** Reads `THEME.physical`.
    - **Palette:** `color_map` argument if given, otherwise `color_palette` in sorted model order
    - **Figsize:** `(25, 15)` (2 rows x 3 columns)
    - **Panels:** IoU, Depth Error, Depth GT vs Pred; Hotspot Offset, T_max Error, T_max GT vs Pred
    - **Box:** `whis=100`, no fliers, box alpha `0.3`; whiskers, caps, and median white at alpha `0.3` (whisker `1.5`, median `2.0`)
    - **Strip:** Size `4`, jitter `0.2`, alpha `0.6`
    - **Scatter:** `alpha=0.7, s=20`, white edge, linewidth `0.5`
    - **Ideal Line (Scatter Panels):** White, dashed, linewidth `1.5`
    - **Layout:** `tight_layout(pad=5.0)`, then `top=0.90, hspace=0.4, wspace=0.3`
    - **DPI:** `300`, `bbox_inches="tight"`

## 5. System Comparison
- **Reference:** `src/neural_pbf/eval/viz/system_dashboard.py` -> `plot_system_dashboard()`
- **Files:** `benchmark_system_comparison.png` (from `experiments/benchmark_suite.py`)
- **Specs:** Reads `THEME.system`, except the figsize.
    - **Figsize:** `(16.7, 15)`, hard-coded (2 rows x 2 columns)
    - **Panel [0,0] (Training Duration):**
        - **Bars:** Per-model color, alpha `0.8`
        - **Line (s/sample, twin axis):** `#f39c12`, marker `o`, linewidth `1.5`
    - **Panels [0,1] and [1,1] (GPU Resources, Training and Inference):**
        - **Scatter:** Per-model color, size `100`, alpha `0.7`
        - **X-axis:** `[0, 105] %`, ticks every `10` up to `100`
        - **Y-axis:** `[0, total VRAM + 1000] MB`; VRAM from `pynvml`, falling back to `16384`
        - **Legend:** Upper left, `framealpha=0.3`
    - **Panel [1,0] (Inference Throughput):** Bars, per-model color, alpha `0.8`
    - **DPI:** `300`, `bbox_inches="tight"`

---

## Spatial Gallery Use Cases

### 6. Training Image Gallery (`gallery_evolution`)
- **Reference:** `src/neural_pbf/eval/viz/spatial.py` -> `gallery_evolution()`
- **Data:** 1 GT sample + up to 9 snapshots from different epochs. Longer histories are reduced to 9 evenly spaced snapshots, endpoints included.
- **Titles:** 'GT Surface', 'GT Depth', and epoch labels (e.g., "Ep 100"), `fontsize=10`.
- **Style:** Hard-coded `cmap="magma"`, `axis('off')` on all images, no `THEME`.
- **Specs:** See Section 1.

### 7. Test Gallery View (`gallery_test`)
- **Reference:** `src/neural_pbf/eval/viz/spatial.py` -> `gallery_test()`
- **Data:** 1 to 5 samples (GT vs. Pred pairs). Unused columns use `axis('off')`.
- **Titles:** `GT {label}` and `Pred {label}`, white (`#ffffff`), bold, `fontsize=10`, `pad=10`.
- **Style:** Reads `THEME.spatial` (`cmap`, figure background, titles). Ticks hidden with `tick_params`.
- **Specs:** See Section 1.

---

## Other Evaluation Figures

### 8. Validation Grid (`val_grid_2x2`)
- **Reference:** `src/neural_pbf/eval/viz/spatial.py` -> `val_grid_2x2()`
- **Specs:** `dark_background`, figsize `(12, 10)`, hard-coded `cmap="magma"`, `vmin=0`, `vmax=1`, `axis('off')`, DPI `120`, `bbox_inches="tight"`. No `THEME`.

### 9. Spatial Comparisons
- **Reference:** `src/neural_pbf/eval/viz/spatial.py`
- **Specs:** Default Matplotlib style, no `THEME`, DPI `100`, `bbox_inches="tight"`.
    - **`triple_view()`:** Figsize `(15, 4)`; GT and Pred `inferno`, error `hot`
    - **`profile_slices()`:** Figsize `(12, 4)`, GT solid, Pred dashed
    - **`isotherm_overlay()`:** Figsize `(7, 5)`; GT `inferno` at alpha `0.4`, GT contour blue, Pred contour red dashed, linewidth `2`

### 10. Temporal Plots
- **Reference:** `src/neural_pbf/eval/viz/temporal.py` -> `peak_tracking_plot()`, `probe_history_plot()`, `error_evolution_plot()`
- **Specs:** `dark_background`, figsize `(10, 4)`, `THEME.base` backgrounds and grid, DPI `100`, `bbox_inches="tight"`.
    - **Peak Tracking:** GT `#ffffff`, Pred `#2ecc71` dashed
    - **Probe History:** GT `#ffffff`, Pred dashed in the default color cycle
    - **Error Evolution:** `#3498db`, marker `o`, size `3`

### 11. Structural and Spectral Benchmarks
- **Structural Reference:** `src/neural_pbf/eval/viz/structural.py` -> `plot_tv_pbd_comparison()`
- **Structural Files:** `benchmark_structural_metrics.png` (from `experiments/benchmark_suite.py`)
- **Structural Specs:** Figsize `(16, 7)`, hard-coded (1 row x 2 columns). Reads `THEME.spectral`; bars use `THEME.system.alpha_bar` (`0.8`). PBD ideal line at `1.0`: white, dashed, linewidth `1.5`, alpha `0.6`. DPI `300`, `bbox_inches="tight"`.
- **Spectral Reference:** `src/neural_pbf/eval/metrics/spectral.py` -> `plot_spectral_full_analysis()`
- **Spectral Files:** `benchmark_spectral_analysis.png` (from `experiments/benchmark_suite.py`)
- **Spectral Specs:** Reads `THEME.spectral`. Figsize `(16.7, 15)` (2 rows x 2 columns), reference line white, linewidth `2.0`, model lines alpha `0.8`, aliasing lines `#f1c40f` dotted at alpha `0.5`, DPI `300`.

---

## 12. UQ Figures
- **Reference:** `src/neural_pbf/eval/viz/uq.py`. `render_uq_figures()` renders the full set; S6 in `src/neural_pbf/uq/stages.py` calls it.
- **Input:** Saved tables only, no model, GPU, or test-set access: `coverage_rows.csv`, `sample_points.csv`, `sensitivity_rows.csv`, `point_errors.csv` in the pipeline root.
- **Re-render:** `scripts/run_uq_pipeline.sh --render-figures` (add `--smoke` for the smoke run). Refused while a pipeline driver runs; warns on uncommitted pipeline code.
- **Logging:** S6 renders a second pass that logs each figure to MLflow through `log_figure()`, under `figures/`, using a temporary copy. That pass does not touch the saved PNGs.

**Shared specs (all from `THEME.uq`):**

- **Theme:** `plt.style.context('dark_background')`; figure and axis `#0a0a0a`
- **Fonts:** Title and suptitle `18` (bold, title pad `20`), label `14`, tick `11`, legend `12`
- **Grid:** `--`, alpha `0.4`, `#666666`; y-axis by default, both axes on line and scatter panels
- **Ideal Line:** White (`#ffffff`), dashed, linewidth `1.5`
- **Method Colors and Markers:**
    - Absolute: `#3498db`, `o`
    - Normalized: `#e67e22`, `s`
    - Constant (single draw): `#2ecc71`, `^`
    - Raw ensemble: `#9b59b6`, `D`
- **Line Markers:** Default `o`, size `8`, method lines linewidth `2.5`
- **Coverage Y-Axis:** `[-0.05, 1.05]`, except coverage vs nominal, which scales to the data
- **Suptitle:** Names the model, plus the method and alpha where a figure shows only one
- **Error Bars:** Linewidth `1.5`, capsize `3`
- **DPI:** `300`, `bbox_inches="tight"`

**Figures:**

| Function | Files | Figsize | Content |
|---|---|---|---|
| `plot_coverage_vs_nominal()` | `coverage_nominal_vs_empirical.png`, `<model>_nominal.png` | `(22, 6)` | 1x3. Empirical against nominal coverage per method, ideal line on the diagonal. Hollow markers show alphas with no run-level support. |
| `plot_spread_vs_error()` | `spread_vs_error.png`, `<model>_spread.png` | `(22, 6)` | 1x3. Ensemble spread against absolute error of the ensemble mean. `s=20`, alpha `0.7`, `#3498db`, white edge `0.5`. |
| `plot_coverage_per_material()` | `coverage_per_material.png`, `<model>_material.png` | `(22, 6)` | 1x3. Primary method and alpha, run-level bootstrap 95% intervals, ideal line at `1 - alpha`. |
| `plot_run_coverage_spread()` | `coverage_per_run.png` | `(12, 8)` | Primary model, method, and alpha. Box (`whis=100`, no fliers, alpha `0.3`) and strip (size `4`, jitter `0.2`, alpha `0.6`) per quantity, ideal line at `1 - alpha`. |
| `plot_sensitivity_vs_k()` | `sensitivity_vs_k.png` | `(25, 15)` | 2x3. Coverage (top) and finite mean width (bottom) against K, per method, at the primary alpha. Primary model only. |
| `plot_leakage_demo()` | `leakage_demo.png` | `(25, 15)` | 2x3. Coverage with bootstrap intervals (top) and MAE bars at alpha `0.8` (bottom). Models in saved order, colored by `color_palette`, numbered, and named with their run-disjoint label in the legend. The order is not a ranking. |

The figures without `<model>` show the model with role `primary` in `coverage_rows.csv`. Each `<model>` file exists once per model in `coverage_rows.csv`.

**Failures:** A figure that fails to render does not fail S6. Its old PNG is removed, `figure_failures.json` in the pipeline root records it, and the report and S7 summary name it together with the `--render-figures` command. A successful re-render removes the notice. An MLflow failure also does not fail S6: `mlflow_failures.json` records it and the S7 summary states it.
