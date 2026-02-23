"""Minimal 2D visual world with bug trajectories and 1D retinal projection.

A schematic arena where a circular "bug" moves along trajectories.
The visual scene is projected onto a 1D retinal strip (horizontal slice
through the arena center), producing a (n_pixels, n_steps) luminance
array suitable for driving a retinal circuit.

The bug is rendered as a Gaussian luminance bump — bright spot on
dark background — mimicking a small object in the frog's visual field.

Reference:
    Lettvin JY, Maturana HR, McCulloch WS, Pitts WH.
    "What the Frog's Eye Tells the Frog's Brain."
    Proc IRE 1959, 47(11):1940-1951.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Arena:
    """A 2D visual arena projected onto a 1D retinal strip.

    Parameters
    ----------
    width : float
        Arena width (arbitrary units).
    n_pixels : int
        Number of pixels in the 1D retinal strip.
    background : float
        Background luminance level.
    """
    width: float = 100.0
    n_pixels: int = 50
    background: float = 0.0


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def straight_trajectory(start, velocity, n_steps, dt=0.1):
    """Bug moves in a straight line.

    Parameters
    ----------
    start : tuple of float
        (x, y) starting position.
    velocity : tuple of float
        (vx, vy) in units/ms.
    n_steps : int
        Number of timesteps.
    dt : float
        Timestep (ms).

    Returns
    -------
    np.ndarray, shape (n_steps, 2)
        Position at each timestep.
    """
    t = np.arange(n_steps) * dt
    x = start[0] + velocity[0] * t
    y = start[1] + velocity[1] * t
    return np.column_stack([x, y])


def circular_trajectory(center, radius, speed, n_steps, dt=0.1):
    """Bug moves in a circle.

    Parameters
    ----------
    center : tuple of float
        (cx, cy) center of circular path.
    radius : float
        Radius of circular path.
    speed : float
        Angular speed (radians/ms).
    n_steps : int
        Number of timesteps.
    dt : float
        Timestep (ms).

    Returns
    -------
    np.ndarray, shape (n_steps, 2)
    """
    t = np.arange(n_steps) * dt
    theta = speed * t
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])


def random_walk_trajectory(start, step_size, n_steps, dt=0.1, seed=None):
    """Bug performs a 2D random walk.

    Parameters
    ----------
    start : tuple of float
        (x, y) starting position.
    step_size : float
        RMS step size per timestep.
    n_steps : int
        Number of timesteps.
    dt : float
        Timestep (ms).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_steps, 2)
    """
    rng = np.random.RandomState(seed)
    steps = rng.randn(n_steps, 2) * step_size * np.sqrt(dt)
    positions = np.cumsum(steps, axis=0)
    positions[:, 0] += start[0]
    positions[:, 1] += start[1]
    return positions


def static_trajectory(position, n_steps):
    """Bug sits still at a fixed position.

    Parameters
    ----------
    position : tuple of float
        (x, y) position.
    n_steps : int
        Number of timesteps.

    Returns
    -------
    np.ndarray, shape (n_steps, 2)
    """
    return np.tile(position, (n_steps, 1))


def onset_trajectory(position, onset_step, n_steps):
    """Bug appears at onset_step, absent before.

    Returns positions with NaN before onset (no bug) and fixed
    position after onset.

    Parameters
    ----------
    position : tuple of float
        (x, y) where the bug appears.
    onset_step : int
        Step at which the bug appears.
    n_steps : int
        Total number of timesteps.

    Returns
    -------
    np.ndarray, shape (n_steps, 2)
        NaN rows for steps before onset.
    """
    positions = np.full((n_steps, 2), np.nan)
    positions[onset_step:] = position
    return positions


# ---------------------------------------------------------------------------
# Retinal projection
# ---------------------------------------------------------------------------

def render_retina(arena, bug_positions, bug_radius=2.0):
    """Project bug onto 1D retinal strip.

    The retinal strip is a horizontal line through the center of
    the arena (y = arena.width/2). Each pixel covers an equal
    fraction of the arena width. The bug is rendered as a Gaussian
    luminance bump centered at the bug's x-coordinate.

    Parameters
    ----------
    arena : Arena
        The visual arena.
    bug_positions : np.ndarray, shape (n_steps, 2)
        Bug (x, y) at each timestep. NaN rows → no bug (background only).
    bug_radius : float
        Spatial width (sigma) of the Gaussian bug profile, in arena units.

    Returns
    -------
    np.ndarray, shape (n_pixels, n_steps)
        Luminance at each retinal pixel at each timestep.
    """
    n_steps = bug_positions.shape[0]
    n_pix = arena.n_pixels

    # Pixel centers in arena coordinates
    pixel_x = np.linspace(0, arena.width, n_pix, endpoint=False)
    pixel_x += arena.width / (2 * n_pix)  # center of each bin

    image = np.full((n_pix, n_steps), arena.background)

    for t in range(n_steps):
        bx = bug_positions[t, 0]
        if np.isnan(bx):
            continue  # no bug this frame
        # Gaussian luminance profile centered at bug x-position
        image[:, t] = arena.background + np.exp(
            -0.5 * ((pixel_x - bx) / bug_radius) ** 2
        )

    return image
