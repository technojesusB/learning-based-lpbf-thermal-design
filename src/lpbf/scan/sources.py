# src/lpbf/scan/sources.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import torch
from pydantic import BaseModel, Field, ConfigDict

class HeatSourceConfig(BaseModel):
    """
    Base configuration for any heat source.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    power: float = Field(..., ge=0.0, description="Power in Watts [W]")
    eta: float = Field(1.0, ge=0.0, le=1.0, description="Absorption efficiency")

class GaussianSourceConfig(HeatSourceConfig):
    sigma: float = Field(..., gt=0.0, description="Gaussian beam radius [m] (1/e^2 or similar definition)")
    depth: float | None = Field(None, gt=0.0, description="Optical penetration depth [m] for volumetric source. If None, surface flux.")

class HeatSource(ABC):
    """
    Abstract Base Class for Heat Sources.
    """
    def __init__(self, config: HeatSourceConfig):
        self.config = config

    @abstractmethod
    def intensity(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor | None, 
                  x0: float, y0: float, z0: float | None = None) -> torch.Tensor:
        """
        Compute spatial intensity distribution centered at (x0, y0, z0).
        Returns Q [W/m^3] or q [W/m^2] depending on source type.
        """
        pass

class GaussianBeam(HeatSource):
    def __init__(self, config: GaussianSourceConfig):
        super().__init__(config)
        self.config: GaussianSourceConfig = config

    def intensity(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor | None, 
                  x0: float, y0: float, z0: float | None = None) -> torch.Tensor:
        """
        Gaussian profile:
        I(r) = (2P / (pi*sigma^2)) * exp(-2*r^2/sigma^2)  <-- Standard laser definition?
        
        Let's use the provided 'sigma' as standard deviation (simpler math):
        Gaussian(r) ~ exp(-r^2 / (2*sigma^2))
        Normalization must ensure Integral(I) dA = Power * eta.
        
        Surface Flux (2D or 3D Surface):
          Integral(A * exp(-(x^2+y^2)/(2s^2))) = A * 2*pi*s^2
          So A = (P*eta) / (2*pi*s^2)
          
        Volumetric (3D with depth):
          Beer-Lambert decay in Z? or Gaussian in Z?
          Let's assume Gaussian in Z as well for simplicity, or exponential decay.
          Usually: I(z) = I0 * exp(-z/depth)
        """
        # Distances squared
        r2 = (X - x0)**2 + (Y - y0)**2
        
        # Base Gaussian (2D)
        # using 1/(2*pi*sigma^2) normalization makes integral = 1
        norm_2d = 1.0 / (2.0 * 3.1415926535 * self.config.sigma**2)
        shape_2d = torch.exp(-r2 / (2.0 * self.config.sigma**2))
        
        flux = (self.config.power * self.config.eta) * norm_2d * shape_2d
        
        if Z is not None and self.config.depth is not None:
            # Volumetric source
            # Multiply by normalized Z profile
            # Exponential decay into material (assume surface at z=0 or z=z0?)
            # Let's assume z0 is the surface z coordinate.
            # I(z) ~ exp(-(z_depth)/d)
            # Integral_0^inf exp(-z/d) dz = d
            
            # If Z coordinate system: Z decreases into material? Or Increases?
            # Standard: Z is height. Material <= Z_surface.
            # Depth d means z in [Z_surface - d, Z_surface]? or exp decay?
            
            # Let's assume Exponential decay from Z_surface (z0) downwards.
            # Z < z0. depth = z0 - Z.
            
            d = self.config.depth
            dz = (z0 - Z) if z0 is not None else -Z # assuming z0=0 if None
            
            # Mask for above surface (no heat)
            mask = (dz >= 0.0) 
            
            # Normalized z-profile: (1/d) * exp(-z/d)
            z_profile = (1.0 / d) * torch.exp(-dz / d)
            
            Q_vol = flux * z_profile
            return torch.where(mask, Q_vol, torch.zeros_like(Q_vol))
        
        elif Z is not None and self.config.depth is None:
            # Surface flux applied to top layer of 3D grid?
            # This is tricky in FDM. Usually we apply specific Surface BC.
            # Or we return a volumetric source that is only non-zero at the surface layer?
            # For now, let's assume the caller expects Volume Source for 3D and "Surface Source" (implicit thickness) for 2D.
            # If 3D and depth is None -> Error or Delta function?
            # We'll return 0 in volume and assume Boundary Condition handles it?
            # No, 'source term' Q is usually volumetric. 
            # If strictly surface, it should be added to the boundary condition, NOT Q.
            raise NotImplementedError("Surface flux in 3D volume requires implementation via Boundary Conditions, not Source Term Q.")
            
        else:
            # 2D case (X, Y only).
            # "Volumetric" in 2D means [W/m^3] assuming unit thickness?
            # Or [W/m^2] if T is integrated?
            # Our PDE: rho*cp*dT/dt = ... + Q
            # If dimensions of T are K, then term is K/s.
            # Q term must be [W/m^3] / [J/m^3 K] * [K] -> [W/m^3].
            # So Q must be W/m^3.
            # If 2D sim, we assume some thickness 'dz_2d'?
            # Usually we treat it as "Average Q over thickness dz".
            # Q_2d = Flux_surf / dz_eff?
            # Or just Q is W/m^3.
            
            # Let's assume the user provides Q as Volumetric Energy Density Source 
            # OR we assume the "Surface Flux" is absorbed in the first layer or distributed?
            
            # For 2D "Plate", Q is usually W/m^2 (source per unit area) / thickness.
            # Let's assume we return Flux [W/m^2]. The integrator must divide by thickness if needed.
            return flux 

