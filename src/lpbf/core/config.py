# src/lpbf/config.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import torch
from lpbf.utils.units import LengthUnit, to_meters

class SimulationConfig(BaseModel):
    """
    Global configuration for the LPBF thermal simulation.
    
    This class handles the definition of the simulation domain size, grid resolution,
    and base time stepping parameters. It enforces strict SI units (meters, seconds, Kelvin)
    for all internal calculations, while allowing users to define the domain geometry
    in more convenient units (e.g., millimeters) via the `length_unit` field.

    Attributes:
        Lx (float): Length of the domain in the X-direction (in `length_unit`).
        Ly (float): Length of the domain in the Y-direction (in `length_unit`).
        Lz (float | None): Length of the domain in the Z-direction (in `length_unit`).
                           If None, the simulation runs in 2D mode.
        Nx (int): Number of grid points in the X-direction. Must be > 1.
        Ny (int): Number of grid points in the Y-direction. Must be > 1.
        Nz (int): Number of grid points in the Z-direction. Must be > 0. Defaults to 1.
        length_unit (LengthUnit): The unit system used for input lengths (Lx, Ly, Lz).
                                  Defaults to MILLIMETERS.
        dt_base (float): The base timestep for the integrator in seconds [s].
        T_ambient (float): The ambient (initial) temperature of the domain in Kelvin [K].
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Domain size (user units)
    Lx: float = Field(..., gt=0.0, description="Length in X (user units)")
    Ly: float = Field(..., gt=0.0, description="Length in Y (user units)")
    Lz: float | None = Field(None, gt=0.0, description="Length in Z (user units). If None, 2D mode.")
    
    # Grid resolution
    Nx: int = Field(..., gt=1)
    Ny: int = Field(..., gt=1)
    Nz: int = Field(1, gt=0)

    # Unit system for INPUTS
    length_unit: LengthUnit = Field(default=LengthUnit.MILLIMETERS)

    # Time settings
    dt_base: float = Field(default=1e-5, gt=0.0, description="Base timestep [s]")
    T_ambient: float = Field(default=293.15, gt=0.0, description="Ambient temperature [K]")
    loss_h: float = Field(default=0.0, ge=0.0, description="Linear cooling loss coefficient [1/s]")

    @property
    def is_3d(self) -> bool:
        """
        Check if the simulation is configured for 3D.

        Returns:
            bool: True if Lz is provided and Nz > 1, False otherwise.
        """
        return self.Lz is not None and self.Nz > 1

    @property
    def Lx_m(self) -> float:
        """
        Get the X-domain length in meters.

        Returns:
            float: Lx converted to [m].
        """
        return to_meters(self.Lx, self.length_unit)

    @property
    def Ly_m(self) -> float:
        """
        Get the Y-domain length in meters.

        Returns:
            float: Ly converted to [m].
        """
        return to_meters(self.Ly, self.length_unit)

    @property
    def Lz_m(self) -> float:
        """
        Get the Z-domain length in meters.

        Returns:
            float: Lz converted to [m]. Returns 0.0 if 2D.
        """
        if self.Lz is None: return 0.0
        return to_meters(self.Lz, self.length_unit)

    @property
    def dx(self) -> float:
        """
        Grid spacing in the X-direction in meters.

        Returns:
            float: dx = Lx_m / (Nx - 1) [m].
        """
        return self.Lx_m / (self.Nx - 1) if self.Nx > 1 else self.Lx_m

    @property
    def dy(self) -> float:
        """
        Grid spacing in the Y-direction in meters.

        Returns:
            float: dy = Ly_m / (Ny - 1) [m].
        """
        return self.Ly_m / (self.Ny - 1) if self.Ny > 1 else self.Ly_m
    
    @property
    def dz(self) -> float:
        """
        Grid spacing in the Z-direction in meters.

        Returns:
            float: dz = Lz_m / (Nz - 1) [m] if 3D, else 1.0 (arbitrary unit thickness).
        """
        if not self.is_3d: return 1.0 # arbitrary for 2D
        return self.Lz_m / (self.Nz - 1) if self.Nz > 1 else self.Lz_m
