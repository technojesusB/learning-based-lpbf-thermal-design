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
