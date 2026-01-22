# Commercially Pure Titanium (CP-Ti Grade 2)

Pure titanium for biocompatible and corrosion-resistant LPBF parts.

## Key Physical Constants
| Property | Value | Source |
| :--- | :--- | :--- |
| **Melting Temperature ($T_{m}$)** | 1668 °C (1941 K) | [1] |
| **Latent Heat of Fusion ($L_f$)** | 440 J/g | [1] |
| **Reference Density ($\rho_{RT}$)**| 4510 kg/m³ | [1] |

## Simulation Assumptions & Extrapolation
*   **Phase Change**: Allotropic $\alpha \to \beta$ transition at 885 °C involves a small heat of transformation, simplified as increased $C_p$ in the function.
*   **Liquid State**: Properties assumed constant above $T_m$.

## Thermophysical Table
Reference: NIST / Carpenter Technology.

| T [K] | T [°C] | $k$ [W/mK] | $C_p$ [J/kgK] | $\alpha$ [$mm²/s$] |
| :--- | :--- | :--- | :--- | :--- |
| 298 | 25 | 21.6 | 522 | 9.17 |
| 700 | 427 | 28.0 | 700 | 8.87 |
| 1158 | 885 | 30.0 | 900 | 7.37 |
| 1700 | 1427| 32.0 | 1200 | 5.91 |
| 1941 | 1668| 32.0 | 1200 | 5.91 |

## Visualization
![CP-Ti Properties](/docs/assets/materials/cp_ti_refined.png)

---

## References & Citations

```bibtex
@online{NISTTitanium,
  title = {Thermophysical Properties of Pure Titanium},
  organization = {NIST WebBook},
  url   = {https://webbook.nist.gov/chemistry/}
}
```
