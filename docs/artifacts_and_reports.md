# Artifacts & Reports

## Visualization
The pipeline generates visual artifacts to aid in debugging and analysis.

### PNG Snapshots
Static heatmaps generated using Matplotlib.
- Location: `plots/png/`
- Cadence: Controlled by `png_every_n_steps` (default 50).

### Interactive Plots
Interactive 3D/Heatmap visualizations using Plotly.
- Location: `plots/interactive/`
- Cadence: Controlled by `html_every_n_steps` (default 250).
- **Note**: Requires `plotly` installed. If missing, this feature degrades gracefully.

## Reports
A summary HTML report is generated at the end of the run.
- Location: `report/index.html`
- Contains:
    - Run Metadata (Git hash, config, material properties).
    - Gallery of PNG snapshots.
    - Links to interactive plots.

## Configuration
Configure artifacts via `ArtifactConfig`:
```python
artifact_cfg = ArtifactConfig(
    enabled=True,
    png_every_n_steps=50,
    html_every_n_steps=250,
    make_report=True,
    downsample=2  # Downsample 2x for smaller files
)
```
