# src/lpbf/config.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import torch
from .units import LengthUnit, to_meters

class SimulationConfig(BaseModel):
    """
    Global simulation configuration.
    Enforces SI units (meters) internally.
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

    @property
    def is_3d(self) -> bool:
        return self.Lz is not None and self.Nz > 1

    @property
    def Lx_m(self) -> float:
        return to_meters(self.Lx, self.length_unit)

    @property
    def Ly_m(self) -> float:
        return to_meters(self.Ly, self.length_unit)

    @property
    def Lz_m(self) -> float:
        if self.Lz is None: return 0.0
        return to_meters(self.Lz, self.length_unit)

    @property
    def dx(self) -> float:
        return self.Lx_m / (self.Nx - 1) if self.Nx > 1 else self.Lx_m

    @property
    def dy(self) -> float:
        return self.Ly_m / (self.Ny - 1) if self.Ny > 1 else self.Ly_m
    
    @property
    def dz(self) -> float:
        if not self.is_3d: return 1.0 # arbitrary for 2D
        return self.Lz_m / (self.Nz - 1) if self.Nz > 1 else self.Lz_m

