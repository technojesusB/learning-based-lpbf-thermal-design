# Maraging Steel 1.2709 (M300)

Ultra-high-strength tool steel. Famous for its toughness and ease of LPBF processing.

## Key Physical Constants
| Property | Value | Source |
| :--- | :--- | :--- |
| **Solidus Temperature ($T_{sol}$)** | 1430 °C (1703 K) | [3] |
| **Liquidus Temperature ($T_{liq}$)** | 1480 °C (1753 K) | [3] |
| **Latent Heat of Fusion ($L_f$)** | $\approx 270$ J/g | [3] |
| **Reference Density ($\rho_{RT}$)**| 8100 kg/m³ | [1] |

## Simulation Assumptions & Extrapolation
*   **Solid Extension**: Linear extension of room temperature conductivity up to 1300 °C.
*   **Isothermal Phase**: Latent heat distributed linearly over the solidification interval (50 K).

## Thermophysical Table
Compiled from Renishaw and academic literature.

| T [°C] | T [K] | $k$ [W/mK] | $C_p$ [J/kgK] | $\alpha$ [$mm²/s$] |
| :--- | :--- | :--- | :--- | :--- |
| 20 | 293 | 14.1 | 450 | 3.87 |
| 600 | 873 | 21.0 | 450 | 5.76 |
| 1300 | 1573 | 29.0 | 450 | 7.96 |
| 1430 | 1703 | 31.0 | 450 | 8.50 |

## Visualization
![Maraging Properties](/docs/assets/materials/maraging_refined.png)

---

## References & Citations

```bibtex
@manual{RenishawMaragingRef,
  title  = {Maraging Steel 1.2709 (M300) Material Data Sheet},
  organization = {Renishaw plc},
  year   = {2021}
}
```
