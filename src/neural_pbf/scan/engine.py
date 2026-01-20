# src/lpbf/scan/engine.py
from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ScanPattern(str, Enum):
    """
    Enumeration of supported scan patterns.
    """

    LINE = "line"
    HATCH = "hatch"
    POINT = "point"


class ScanEvent(BaseModel):
    """
    Represents a single atomic event in the laser scan path.
    Can be a continuous vector scan (Line) or a spot dwell (Point).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Coordinates in meters [m]
    x_start: float = Field(..., description="Start X coordinate [m]")
    y_start: float = Field(..., description="Start Y coordinate [m]")
    x_end: float = Field(..., description="End X coordinate [m]")
    y_end: float = Field(..., description="End Y coordinate [m]")

    power: float = Field(..., description="Laser power during this event [W]")
    speed: float = Field(..., description="Scan speed [m/s]. If 0, treated as dwell.")

    # For spot dwell
    dwell_time: float = Field(0.0, description="Dwell time [s] if speed is 0.")

    # Laser status
    laser_on: bool = Field(
        True, description="Whether the laser is active emitting power."
    )

    @property
    def is_point(self) -> bool:
        """Check if the event is a point dwell (zero distance)."""
        dist = math.sqrt(
            (self.x_end - self.x_start) ** 2 + (self.y_end - self.y_start) ** 2
        )
        return dist < 1e-9

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the event [s].
        Returns:
            float: dwell_time if point, else distance / speed.
        """
        if self.is_point:
            return self.dwell_time
        elif self.speed > 0:
            dist = math.sqrt(
                (self.x_end - self.x_start) ** 2 + (self.y_end - self.y_start) ** 2
            )
            return dist / self.speed
        else:
            return 0.0


class ScanPathGenerator:
    """
    Utility class to generate sequences of ScanEvents (scan paths) from high-level
    parameters.
    """

    @staticmethod
    def line(
        start: tuple[float, float], end: tuple[float, float], power: float, speed: float
    ) -> list[ScanEvent]:
        """
        Generate a single linear scan vector.

        Args:
            start (tuple[float, float]): (x, y) start coordinates [m].
            end (tuple[float, float]): (x, y) end coordinates [m].
            power (float): Laser power [W].
            speed (float): Scan speed [m/s].

        Returns:
            List[ScanEvent]: A list containing the single scan event.
        """
        return [
            ScanEvent(
                x_start=start[0],
                y_start=start[1],
                x_end=end[0],
                y_end=end[1],
                power=power,
                speed=speed,
                laser_on=True,
            )
        ]

    @staticmethod
    def hatch(
        corner_start: tuple[float, float],
        width: float,
        height: float,
        spacing: float,
        power: float,
        speed: float,
        angle_deg: float = 0.0,
        skywriting: bool = False,
    ) -> list[ScanEvent]:
        """
        Generate a serpentine hatch pattern covering a rectangular area.
        Includes "travel" (laser off) events between hatch lines.

        Args:
            corner_start (tuple[float, float]): Bottom-left corner (x, y) [m].
            width (float): Width of rect in X [m].
            height (float): Height of rect in Y [m].
            spacing (float): Hatch spacing (hatch distance) [m].
            power (float): Laser power [W].
            speed (float): Scan speed [m/s].
            angle_deg (float): Rotation angle in degrees (Not yet implemented).
            skywriting (bool): Whether to add skywriting maneuvers (Not yet
                implemented).

        Returns:
            List[ScanEvent]: Sequence of laser-on and laser-off events.
        """
        # Simplification: Only 0 degree (horizontal) for now
        # TODO: Implement Rotation matrix for angle_deg

        events = []

        y_current = corner_start[1]
        y_max = corner_start[1] + height
        x_min = corner_start[0]
        x_max = corner_start[0] + width

        direction = 1  # 1 = right, -1 = left

        while y_current <= y_max:
            if direction == 1:
                start = (x_min, y_current)
                end = (x_max, y_current)
            else:
                start = (x_max, y_current)
                end = (x_min, y_current)

            # Scan line
            events.append(
                ScanEvent(
                    x_start=start[0],
                    y_start=start[1],
                    x_end=end[0],
                    y_end=end[1],
                    power=power,
                    speed=speed,
                    laser_on=True,
                )
            )

            # Jump to next line (Laser OFF)
            next_y = y_current + spacing
            if next_y <= y_max:
                # Travel event
                # Assume high travel speed (e.g. 5x scan speed)
                travel_speed = speed * 5.0
                events.append(
                    ScanEvent(
                        x_start=end[0],
                        y_start=end[1],
                        x_end=end[0] if direction == 1 else x_min,
                        y_end=next_y,
                        power=0.0,
                        speed=travel_speed,
                        laser_on=False,
                    )
                )

            y_current = next_y
            direction *= -1

        return events
