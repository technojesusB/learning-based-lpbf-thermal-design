# Aluminum Alloy AlSi10Mg

Lightweight casting alloy with high thermal conductivity. Ideal for thermal management applications in LPBF.

## Key Physical Constants
| Property | Value | Source |
| :--- | :--- | :--- |
| **Solidus Temperature ($T_{sol}$)** | 570 °C (843 K) | [2] |
| **Liquidus Temperature ($T_{liq}$)** | 610 °C (883 K) | [2] |
| **Latent Heat of Fusion ($L_f$)** | $\approx 380$ J/g | [Est.] |
| **Reference Density ($\rho_{RT}$)**| 2680 kg/m³ | [1] |

## Simulation Assumptions & Extrapolation
*   **Data Limit**: Akwaboa (2023) data is measured up to 673K (400 °C).
*   **Solid Extension**: Linear extrapolation is used from 673K up to $T_{sol}$ (843K).
*   **Liquid Phase**: Properties assumed constant above $T_{liq}$.

## Thermophysical Table
Reference: Akwaboa et al. [1].

| T [°C] | T [K] | $k$ [W/mK] | $C_p$ [J/kgK] | $\alpha$ [$mm²/s$] | State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 100 | 373 | 138.4 | 800 | 64.5 | As-Built |
| 200 | 473 | 153.9 | 835 | 68.7 | As-Built |
| 300 | 573 | 169.8 | 862 | 73.5 | As-Built |
| 400 | 673 | 155.2 | 758 | 76.4 | As-Built |

## Visualization
![AlSi10Mg Properties](/docs/assets/materials/alsi10mg_refined.png)

---

## References & Citations

```bibtex
@article{Akwaboa2023LPBF,
  author  = {Akwaboa, Stephen and others},
  title   = {Thermophysical Properties of Laser Powder Bed Fused Ti-6Al-4V and AlSi10Mg Alloys...},
  journal = {Materials},
  volume  = {16},
  year    = {2023}
}
```
