# Physics Formulation

## Governing Equation
The simulator solves the transient heat conduction equation with temperature-dependent properties and phase change:

$$ \rho(T) c_p(T) \frac{\partial T}{\partial t} = \nabla \cdot (k(T) \nabla T) + Q_{source} - Q_{loss} $$

Where:
- $T$: Temperature [K]
- $\rho$: Density [kg/m³]
- $c_p$: Specific heat capacity [J/(kg·K)]
- $k$: Thermal conductivity [W/(m·K)]
- $Q_{source}$: Volumetric heat source [W/m³]
- $Q_{loss}$: Boundary or volumetric heat loss (convection/radiation) [W/m³]

## Phase Change
Phase change (melting/solidification) is handled using the **Apparent Heat Capacity** method. The latent heat $L$ is incorporated into an effective heat capacity $c_p^{eff}$:

$$ c_p^{eff}(T) = c_p^{base} + L \frac{d\phi}{dT} $$

Where $\phi(T)$ is the liquid fraction ($0 \le \phi \le 1$). We approximate $\frac{d\phi}{dT}$ using a Gaussian function centered at the melting range:

$$ \frac{d\phi}{dT} \approx \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(T-T_m)^2}{2\sigma^2}} $$

This ensures that $\int c_p^{eff} dT \approx \Delta H_{sensible} + L$.

## Discretization
We use the **Finite Difference Method (FDM)** on a regular grid (2D or 3D).
Time integration is performed using **Explicit Euler** stepping:

$$ T^{n+1} = T^n + \Delta t \frac{1}{\rho c_p} \left( \nabla \cdot (k \nabla T) + Q \right) $$

### Stability
Explicit Euler is conditionally stable. The timestep $\Delta t$ must satisfy:

$$ \Delta t \le \frac{\min(dx, dy, dz)^2}{2 \cdot D \cdot \alpha_{max}} $$

Where $\alpha = k / (\rho c_p)$ is thermal diffusivity and $D$ is dimensionality (2 or 3). The simulator includes an **Adaptive Time Stepper** (`stepper.step_adaptive`) that automatically sub-steps if the requested $\Delta t$ violates this condition.

## Boundary Conditions
- **Domain Boundaries (`ops.py`):** Neumann (Zero Flux) condition is applied by default via replicate padding. This simulates an insulated domain or symmetry.
- **Surface**: For 2D simulation, surface flux can be modeled as a volumetric source distributed over the thickness, or explicitly added as a sink term.

## Cooling Rate Calculation
Cooling rate $R = |\partial T / \partial t|$ is strictly defined as the instantaneous cooling rate at the moment the temperature crosses the **Solidification Temperature** ($T_{solidus}$) downwards.

The simulator captures this event at every pixel:
1. Store $T_{prev}$ from previous step.
2. Check if $T_{prev} > T_{sol}$ and $T_{new} \le T_{sol}$.
3. If true, record $R = (T_{prev} - T_{new}) / \Delta t$.
