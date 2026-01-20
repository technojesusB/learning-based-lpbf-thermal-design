# src/lpbf/scan/sources.py
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, ConfigDict, Field


class HeatSourceConfig(BaseModel):
    """
    Base configuration for any heat source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    power: float = Field(..., ge=0.0, description="Laser Power in Watts [W]")
    eta: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Absorption efficiency (0.0 to 1.0). Absorbed Power = Power * eta.",
    )


class GaussianSourceConfig(HeatSourceConfig):
    """
    Configuration for a Gaussian Beam Heat Source.
    """

    sigma: float = Field(
        ...,
        gt=0.0,
        description="Gaussian standard deviation [m]. Related to D4sigma diameter by D4s = 4 * sigma.",
    )
    depth: float | None = Field(
        None,
        gt=0.0,
        description="Optical penetration depth [m] for volumetric source. If None, acts as a surface flux [W/m^2].",
    )


class HeatSource(ABC):
    """
    Abstract Base Class for Heat Sources.
    """

    def __init__(self, config: HeatSourceConfig):
        self.config = config

    @abstractmethod
    def intensity(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor | None,
        x0: float,
        y0: float,
        z0: float | None = None,
    ) -> torch.Tensor:
        """
        Compute the spatial intensity distribution of the heat source centered at (x0, y0, z0).

        Args:
            X (torch.Tensor): X-coordinates of the evaluation grid (shape: [..., H, W] or broadcastable).
            Y (torch.Tensor): Y-coordinates of the evaluation grid.
            Z (torch.Tensor | None): Z-coordinates of the evaluation grid. Required if the source is volumetric.
            x0 (float): Current X-position of the beam center [m].
            y0 (float): Current Y-position of the beam center [m].
            z0 (float | None): Current Z-position of the beam center (surface Z) [m]. Defaults to 0 or None.

        Returns:
            torch.Tensor: Heat source intensity field.
                          - If surface source: Flux [W/m^2].
                          - If volumetric source: Volumetric Heat Generation [W/m^3].
        """
        pass


class GaussianBeam(HeatSource):
    """
    Implementation of a Gaussian Laser Beam Heat Source.

    Supports both 2D (Surface Flux) and 3D (Volumetric) modes.

    Physical Model (Surface):
        I(r) = A * exp(-r^2 / (2 * sigma^2))
        Normalization: Integral(I) dA = P_absorbed
        A = P_absorbed / (2 * pi * sigma^2)

    Physical Model (Volumetric):
        Q(r, z) = I(r) * f(z)
        where f(z) is an exponential decay into the material (Beer-Lambert law approximation).
        f(z) = (1/d) * exp(-z_depth / d)
        Normalization: Integral(f(z)) dz = 1
    """

    def __init__(self, config: GaussianSourceConfig):
        super().__init__(config)
        self.config: GaussianSourceConfig = config

    def intensity(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor | None,
        x0: float,
        y0: float,
        z0: float | None = None,
    ) -> torch.Tensor:
        """
        Compute the Gaussian beam intensity.

        Args:
            X (torch.Tensor): Grid X coordinates [m].
            Y (torch.Tensor): Grid Y coordinates [m].
            Z (torch.Tensor | None): Grid Z coordinates [m].
            x0 (float): Beam center X [m].
            y0 (float): Beam center Y [m].
            z0 (float | None): Beam center Z (Surface) [m]. Unused for purely 2D surface flux.

        Returns:
            torch.Tensor: Source term field. [W/m^2] (2D) or [W/m^3] (3D).

        Raises:
            NotImplementedError: If Z coordinates are provided (3D context) but the source is configured without a depth (Surface Flux),
                                 as implementing surface flux in a 3D FDM volume requires setting Boundary Conditions rather than a volumetric source term.
        """
        # Distances squared from center
        r2 = (X - x0) ** 2 + (Y - y0) ** 2

        # Normalization factor for 2D Gaussian
        # 1/(2*pi*sigma^2) ensures Integral of exp(...) is 2*pi*sigma^2, canceling to 1.
        norm_2d = 1.0 / (2.0 * 3.1415926535 * self.config.sigma**2)

        # Radial profile
        shape_2d = torch.exp(-r2 / (2.0 * self.config.sigma**2))

        # Base Flux [W/m^2]
        flux = (self.config.power * self.config.eta) * norm_2d * shape_2d

        if Z is not None and self.config.depth is not None:
            # Volumetric source with exponential decay in Depth
            # z0 is the surface z-coordinate.
            # Depth d is positive into the material.

            # Assuming Z coordinate system:
            # If Z is height (increasing upwards), and material is at Z <= z0:
            # depth_into_mat = z0 - Z

            d = self.config.depth
            dz = (z0 - Z) if z0 is not None else -Z

            # Mask for strictly below surface (dz >= 0)
            mask = dz >= 0.0

            # Normalized z-profile: (1/d) * exp(-z_depth / d)
            # Integral from 0 to inf is 1.
            z_profile = (1.0 / d) * torch.exp(-dz / d)

            Q_vol = flux * z_profile

            # Zero out any contribution above the surface
            return torch.where(mask, Q_vol, torch.zeros_like(Q_vol))

        elif Z is not None and self.config.depth is None:
            # 3D Grid but Surface Flux requested.
            # This cannot be represented as a volumetric source term Q [W/m^3] easily without numerical delta functions.
            # It should be handled by the Boundary Condition logic.
            raise NotImplementedError(
                "Surface flux in 3D volume requires implementation via Boundary Conditions, not Source Term Q."
            )

        else:
            # 2D case (X, Y only).
            # Returns Flux [W/m^2].
            # Note: The PDE integrator must interpret this correctly.
            # For a 2D simulation representing a thin slice or surface, this flux is usually applied as a source.
            return flux
