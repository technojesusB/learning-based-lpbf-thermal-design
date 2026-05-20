# LPBF Thermal Surrogate: Metrics Guide

This guide explains the advanced metrics used to evaluate the structural and spectral fidelity of our 3D Diffusion Transformers, focusing on high-frequency artifacts (aliasing) and physical consistency.

---

## 1. Power Spectral Density (PSD)
**The "Audio EQ" for Physics.**

### What it is:
PSD measures the intensity of "waves" (frequencies) in the 3D temperature field. Just like an audio equalizer shows bass, mids, and treble, PSD shows how much energy is in large structures (low frequency) versus tiny details (high frequency).

### Why we use it:
Patch-based models (DiTs) often create artificial "steps" at patch boundaries. These steps show up as "treble noise" (high-frequency spikes) in the PSD. By comparing the model's PSD to the Ground Truth (Solver), we can see if the model accurately captures the physics across all scales.

### ELI5 version:
> Imagine you're looking at a photo of a beach. The large dunes are low frequencies. The tiny grains of sand are high frequencies. If your camera (the model) adds weird digital noise that looks like a grid, the PSD will show a big spike in the "sand" department where it shouldn't be.

---

## 2. Total Variation (TV)
**The "Graininess" Index.**

### What it is:
Total Variation is the sum of the absolute differences between neighboring voxels. 
$$TV(T) = \sum |T_{i+1} - T_i|$$
It quantifies the "roughness" of the field.

### Why we use it:
Unregularized DiTs (like v1/v2) often produce "jittery" or "noisy" temperature fields. A high TV relative to the Ground Truth means the model is outputting high-frequency noise that isn't physically there.

### ELI5 version:
> Imagine a smooth slide. If you rub your hand over it, it's easy (Low TV). Now imagine a slide made of LEGO bricks. Your hand will jump up and down at every brick edge (High TV). We want our thermal field to be as smooth as the slide, not jumpy like the LEGOs.

---

## 3. Patchiness Boundary Discontinuity (PBD)
**The "Seam" Meter.**

### What it is:
PBD is the ratio of gradients at the patch boundaries versus the gradients inside the patches.
$$PBD = \frac{\text{Mean}(\nabla T_{boundary})}{\text{Mean}(\nabla T_{interior})}$$

### Interpretation:
- **1.0**: Perfect. The transition between patches is as smooth as the rest of the field.
- **> 1.0**: Patchy. There are visible "seams" or "jumps" where the tokens meet.
- **< 1.0**: Unnaturally smooth boundaries (rare).

### Why we use it:
This is our primary metric for "Aliasing". It tells us exactly how much the model is struggling with its $4^3$ or $8^3$ tokenization grid.

### ELI5 version:
> Imagine you're sewing a quilt. If you're a pro, the seams between the fabric patches are invisible (PBD = 1.0). If you're a beginner, the seams are thick and bumpy (PBD > 1.0). We want our AI to be a pro tailor.

---

## Summary Table

| Metric | Ideal (GT) | Meaning of High Value |
| :--- | :--- | :--- |
| **PSD Curve** | Matches GT | High-frequency noise or "checkerboard" artifacts. |
| **TV Error** | 0% | Grainy, "shivering" temperature field. |
| **PBD** | 1.0 | Visible grid/tile boundaries in the 3D volume. |
