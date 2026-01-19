# src/lpbf/scan/engine.py
from __future__ import annotations
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ConfigDict
import torch
import math

class ScanPattern(str, Enum):
    LINE = "line"
    HATCH = "hatch"
    POINT = "point"

class ScanEvent(BaseModel):
    """
    A single continuous laser vector or point dwell.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    # Start and End positions (normalized 0..1 or meters?)
    # Let's enforce meters to match the rest of the new strict SI design.
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    
    power: float
    speed: float # [m/s]. If 0, it's a spot dwell.
    
    # For spot dwell
    dwell_time: float = 0.0
    
    # Laser status
    laser_on: bool = True
    
    @property
    def is_point(self) -> bool:
        dist = math.sqrt((self.x_end - self.x_start)**2 + (self.y_end - self.y_start)**2)
        return dist < 1e-9

    @property
    def duration(self) -> float:
        if self.is_point:
            return self.dwell_time
        elif self.speed > 0:
            dist = math.sqrt((self.x_end - self.x_start)**2 + (self.y_end - self.y_start)**2)
            return dist / self.speed
        else:
            return 0.0

class ScanPathGenerator:
    """
    Helper to generate lists of ScanEvents from high-level parameters.
    """
    @staticmethod
    def line(start: tuple[float, float], end: tuple[float, float], power: float, speed: float) -> List[ScanEvent]:
        return [ScanEvent(
            x_start=start[0], y_start=start[1],
            x_end=end[0], y_end=end[1],
            power=power, speed=speed, laser_on=True
        )]
        
    @staticmethod
    def hatch(
        corner_start: tuple[float, float], 
        width: float, 
        height: float, 
        spacing: float, 
        power: float, 
        speed: float,
        angle_deg: float = 0.0,
        skywriting: bool = False # TODO: Implement skywriting returns
    ) -> List[ScanEvent]:
        """
        Generate a simple serpentine hatch pattern.
        """
        # Simplification: Only 0 degree (horizontal) for now
        # TODO: Implement Rotation matrix for angle_deg
        
        events = []
        
        y_current = corner_start[1]
        y_max = corner_start[1] + height
        x_min = corner_start[0]
        x_max = corner_start[0] + width
        
        direction = 1 # 1 = right, -1 = left
        
        while y_current <= y_max:
            if direction == 1:
                start = (x_min, y_current)
                end = (x_max, y_current)
            else:
                start = (x_max, y_current)
                end = (x_min, y_current)
            
            # Scan line
            events.append(ScanEvent(
                x_start=start[0], y_start=start[1],
                x_end=end[0], y_end=end[1],
                power=power, speed=speed, laser_on=True
            ))
            
            # Jump to next line (Laser OFF)
            next_y = y_current + spacing
            if next_y <= y_max:
                # Travel event
                # Assume separate travel speed? Or same speed?
                # Usually travel speed is high. Let's use same speed for now or add param.
                travel_speed = speed * 5.0 
                events.append(ScanEvent(
                    x_start=end[0], y_start=end[1],
                    x_end=end[0] if direction == 1 else x_min, # pure vertical step?
                    # Meander: end of line 1 -> start of line 2
                    # Line 1: x_min -> x_max. End at x_max.
                    # Line 2: x_max -> x_min. Start at x_max.
                    # Vertical jump: (x_max, y) -> (x_max, y+dy)
                    y_end=next_y,
                    power=0.0,
                    speed=travel_speed,
                    laser_on=False
                ))
            
            y_current = next_y
            direction *= -1
            
        return events
