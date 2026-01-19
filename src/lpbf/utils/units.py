# src/lpbf/units.py
from enum import Enum

class LengthUnit(str, Enum):
    """
    Enumeration of supported length units for simulation input.
    Internally, the simulation always operates in SI units (meters).
    """
    METERS = "m"
    MILLIMETERS = "mm"

def to_meters(value: float, unit: LengthUnit) -> float:
    """
    Convert a length value from the specified unit to meters.

    Args:
        value (float): The length value in `unit`.
        unit (LengthUnit): The source unit (e.g., MILLIMETERS).

    Returns:
        float: The equivalent length in meters [m].

    Raises:
        ValueError: If the provided unit is not supported.
    """
    if unit == LengthUnit.METERS:
        return value
    elif unit == LengthUnit.MILLIMETERS:
        return value * 1e-3
    else:
        raise ValueError(f"Unknown unit: {unit}")

def from_meters(value: float, unit: LengthUnit) -> float:
    """
    Convert a length value from meters to the specified target unit.

    Args:
        value (float): The length value in meters [m].
        unit (LengthUnit): The target unit (e.g., MILLIMETERS).

    Returns:
        float: The equivalent length in `unit`.

    Raises:
        ValueError: If the provided unit is not supported.
    """
    if unit == LengthUnit.METERS:
        return value
    elif unit == LengthUnit.MILLIMETERS:
        return value * 1e3
    else:
        raise ValueError(f"Unknown unit: {unit}")
