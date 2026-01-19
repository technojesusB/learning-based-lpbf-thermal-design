# src/lpbf/units.py
from enum import Enum

class LengthUnit(str, Enum):
    METERS = "m"
    MILLIMETERS = "mm"

def to_meters(value: float, unit: LengthUnit) -> float:
    """Convert length value to meters."""
    if unit == LengthUnit.METERS:
        return value
    elif unit == LengthUnit.MILLIMETERS:
        return value * 1e-3
    else:
        raise ValueError(f"Unknown unit: {unit}")

def from_meters(value: float, unit: LengthUnit) -> float:
    """Convert value in meters to target unit."""
    if unit == LengthUnit.METERS:
        return value
    elif unit == LengthUnit.MILLIMETERS:
        return value * 1e3
    else:
        raise ValueError(f"Unknown unit: {unit}")
