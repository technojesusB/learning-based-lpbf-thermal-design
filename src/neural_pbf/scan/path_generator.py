from typing import Literal

import numpy as np


def generate_pulsed_path(
    pattern: Literal["zigzag", "raster", "island"],
    Lx: float,
    Ly: float,
    point_distance: float,
    hatch_spacing: float,
    angle_deg: float = 0.0,
    island_size: float | None = 2.0e-3,
) -> np.ndarray:
    """
    Generates a sequence of (x, y) coordinates for a pulsed laser scan path.

    Args:
        pattern: Scanning strategy ('zigzag', 'raster', 'island').
        Lx: Physical domain length in X [m].
        Ly: Physical domain length in Y [m].
        point_distance: Distance between discrete laser exposures [m].
        hatch_spacing: Distance between adjacent hatch lines [m].
        angle_deg: Rotation of the scan lines relative to the X-axis [degrees].
        island_size: Size of square islands [m]. Only used if pattern == 'island'.

    Returns:
        np.ndarray: Array of shape (N, 2) containing the (x, y) coordinates.
    """
    theta = np.radians(angle_deg)

    # Calculate the bounding box needed in the rotated coordinate frame
    diag = np.sqrt(Lx**2 + Ly**2)
    Lx_rot = diag * 1.5  # Overscan to ensure coverage
    Ly_rot = diag * 1.5

    x_min, x_max = -Lx_rot / 2, Lx_rot / 2
    y_min, y_max = -Ly_rot / 2, Ly_rot / 2

    points = []

    def _generate_block(bx_min, bx_max, by_min, by_max, zigzag=True):
        """Helper to generate hatches inside a specific block."""
        block_points = []
        y_lines = np.arange(by_min, by_max, hatch_spacing)
        for i, y in enumerate(y_lines):
            x_pts = np.arange(bx_min, bx_max, point_distance)
            if len(x_pts) == 0:
                continue

            y_pts = np.full_like(x_pts, y)

            # If zigzag and it's an odd line, reverse direction
            if zigzag and i % 2 == 1:
                x_pts = x_pts[::-1]
                y_pts = y_pts[::-1]

            pts = np.stack([x_pts, y_pts], axis=1)
            block_points.append(pts)

        if block_points:
            return np.concatenate(block_points, axis=0)
        return np.empty((0, 2))

    if pattern in ["zigzag", "raster"]:
        pts = _generate_block(x_min, x_max, y_min, y_max, zigzag=(pattern == "zigzag"))
        points.append(pts)

    elif pattern == "island":
        if island_size is None or island_size <= 0:
            raise ValueError("island_size must be > 0 for island pattern.")

        x_islands = np.arange(x_min, x_max, island_size)
        y_islands = np.arange(y_min, y_max, island_size)

        for yi in y_islands:
            for xi in x_islands:
                b_x_min, b_x_max = xi, min(xi + island_size, x_max)
                b_y_min, b_y_max = yi, min(yi + island_size, y_max)

                island_pts = _generate_block(
                    b_x_min, b_x_max, b_y_min, b_y_max, zigzag=True
                )
                if island_pts.shape[0] > 0:
                    points.append(island_pts)
    else:
        raise ValueError(f"Unknown pattern {pattern}")

    if not points:
        return np.empty((0, 2))

    path = np.concatenate(points, axis=0)

    # Apply rotation
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_path = path @ R.T

    # Translate back
    shifted_path = rotated_path + np.array([Lx / 2, Ly / 2])

    # Filter points
    eps = 1e-9
    mask = (
        (shifted_path[:, 0] >= -eps)
        & (shifted_path[:, 0] <= Lx + eps)
        & (shifted_path[:, 1] >= -eps)
        & (shifted_path[:, 1] <= Ly + eps)
    )

    return shifted_path[mask]
