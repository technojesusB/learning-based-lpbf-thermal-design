# Materials Gallery

This section provides a detailed overview of the thermophysical properties for materials relevant to Laser Powder Bed Fusion (LPBF). Each material includes a description, granular simulation parameters, and representative plots with extrapolation logic.

## Standardized Documentation Structure
To ensure transparency and simulation-readiness, each material page follows a strict template:
- **Key Physical Constants**: Melting points ($T_{sol}, T_{liq}$), latent heat of fusion ($L_f$), and reference density.
- **Simulation Assumptions**: Explicit documentation of extrapolation logic (e.g., constant properties in liquid phase).
- **Thermophysical Tables**: High-fidelity discrete values for $k, C_p, \rho,$ and calculated diffusivity $\alpha$.
- **Refined Visualizations**: Triple-plots covering $k(T)$, $C_p(T)$, and $\alpha(T)$ with extrapolated regions.
- **BibTeX Citations**: Standardized source attribution in code blocks.

---

## 1. Material Presets (Built-in)
The following materials are available as pre-configured presets in the solver (`MaterialConfig`).

- [**Stainless Steel 316L**](ss316l): Widespread austenitic steel with excellent corrosion resistance.
- [**Ti-6Al-4V (Grade 5)**](ti64): High strength-to-weight ratio, critical for aerospace.

---

## 2. Additional Material Library

### Stainless Steels
- [**Stainless Steel 304**](ss304): Baseline chromium-nickel alloy.
- [**17-4PH Stainless Steel**](17_4ph): Martensitic precipitation-hardening steel.
- [**Maraging Steel 1.2709**](maraging): High-strength tool steel.

### Aluminum Alloys
- [**AlSi10Mg**](alsi10mg): Common casting alloy with high conductivity.
- [**Aluminum 6061**](al6061): High-performance alloy requiring careful thermal management.

### Nickel Superalloys
- [**Inconel 718**](in718): Standard high-temperature superalloy.
- [**Inconel 625**](in625): High-strength, corrosion-resistant superalloy.

### Titanium & Tool Steels
- [**CP-Ti (Grade 2)**](cp_ti): Commercially pure titanium for biomedical/chemical use.
- [**H13 Tool Steel**](h13): Chromium-molybdenum hot-work steel.

---

## 3. Scientific Context

### Material States in LPBF Simulations
Distinguishing between the physical states is critical for accurate thermal modeling:

1.  **Powder Bed**: Un-melted particles. Extremely low conductivity ($k \approx 0.1 - 1.0$ W/mK) due to gas-filled pores and minimal contact area.
2.  **As-Built Solid (LPBF)**: Material immediately after solidification. Often contains supersaturated phases and fine cellular microstructures due to extreme cooling rates ($10^5 - 10^7$ K/s). Properties can differ significantly from wrought counterparts (e.g., lower conductivity).
3.  **Bulk / Wrought**: Conventionally processed fully-dense material. Used as a first-order approximation for liquid and solidified regions after thermal equilibrium/heat treatment.

### Powder-Bed Effective Conductivity
Typical modeling approaches for $k_{eff, powder}$:
- **Zehner-Schlünder (ZBS)**: Geometric model for conduction through particles and gas.
- **Sih-Barlow**: Extends ZBS with radiation terms, essential for high-temperature gradients.

> [!NOTE]
> **Detailed Powder Parameter Grids**: Experimental research on temperature-dependent powder conductivity for these alloys is ongoing. Extended tables for specific powder-bed configurations will be added as high-fidelity data becomes available.

### BibTeX & Citations
Poynting to transparent source attribution provided on each individual material page.
