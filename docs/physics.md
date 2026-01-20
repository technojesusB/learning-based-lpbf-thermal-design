# Physics Model & Numerical Methods

## Governing Equation
We solve the transient Heat Conduction equation with a temperature-dependent source term and effective properties to model phase change:

$$ \rho C_{eff}(T) \frac{\partial T}{\partial t} = \nabla \cdot (k(T) \nabla T) + Q_{source} + Q_{loss} $$

Where:
- $\rho$: Density [kg/m^3] (assumed constant)
- $C_{eff}(T)$: Effective Specific Heat Capacity [J/(kg K)] including Latent Heat.
- $k(T)$: Thermal Conductivity [W/(m K)].
- $Q_{source}$: Volumetric Heat Source [W/m^3].
- $Q_{loss}$: Boundary heat loss [W/m^3] (approximate).

## Discretization

### Spatial
- **Method**: 2nd-order Central Finite Differences on a structured Cartesian grid.
- **Fluxes**: Thermal conductivity $k$ at cell faces is computed using the **Harmonic Mean** to ensure flux continuity and correct limits for high-contrast materials.
- **Boundaries**: Homogeneous Neumann (Zero Flux) conditions are enforced via ghost-cell padding (`mode='replicate'`).

### Temporal
- **Method**: Explicit Euler.
- **Stability**: The timestep $dt$ is constrained by the CFL condition for diffusion: $dt < \frac{dx^2}{2 \alpha_{max}}$.
- **Phase Change**: Modeling via the **Apparent Heat Capacity Method**. $C_{eff}$ includes a term proportional to $L \frac{d\phi}{dT}$, where $\phi(T)$ is the smooth liquid fraction function (Sigmoid).

## 2D Mode Assumptions
When running in 2D (`Lz` is None):
- The domain represents a thin slice or layer of material.
- **Source Normalization**: Input surface flux (e.g., Laser $W/m^2$) is converted to volumetric heat generation $W/m^3$ by dividing by an assumed layer thickness `default_dz`.
- **Consistency**: Users must specify `default_dz` in `SimulationConfig` to ensure energy conservation is calculated over a physical volume $V = L_x L_y dz$.

## Key Parameter Choices & Defaults

| Parameter | Default Value | Rationale |
| :--- | :--- | :--- |
| **`default_dz`** | `0.05 mm` | Represents a typical LPBF powder layer thickness (30-60 $\mu m$). This is used to normalize surface flux ($W/m^2$) into volumetric heat ($W/m^3$) in 2D simulations. Correct normalization is critical for physical temperatures. |
| **`loss_h`** | `0.0` or user | Heat loss coefficient. While physically units should be $[W/m^3\cdot K]$ in our volumetric source formulation, we currently treat it as a linear decay/convection term. Default is 0 (Adiabatic) for strict energy conservation checks. |
| **`transition_sharpness`** | `5.0` | Controls the width of the sigmoid phase transition. A value of 5.0 ensures the transition occurs smoothly over the defined $T_{liq} - T_{sol}$ range without being so sharp that it causes gradients to explode in the explicit solver. |
| **Test `atol`** | `1e-3` | Unit tests use `atol=1e-3` for energy checks. This loose tolerance accounts for the accumulation of float32 errors over time steps and the inherent truncation error of the 1st-order Explicit Euler scheme. |

