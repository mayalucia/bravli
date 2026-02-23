"""Frog retinal circuit: predictive coding without plasticity.

A schematic 3-population retinal circuit inspired by Lettvin et al. (1959)
and the predictive coding reinterpretation by Srinivasan, Laughlin & Dubs
(1982). The circuit demonstrates that temporal filtering (slow vs fast
time constants) and spatial center-surround connectivity implement
prediction-error computation without any learning.

Three population types, replicated across N spatial locations:

    P (Photoreceptors): Fast (τ=2ms), relay luminance input.
    S (Sustained):      Slow (τ=50ms), center-surround input from P.
                        Tracks the temporal mean = the prediction.
    T (Transient):      Fast (τ=5ms), P excitation minus S inhibition.
                        Fires when stimulus deviates from prediction = error.

The key insight: S's slow time constant makes it a temporal low-pass
filter, automatically tracking the running mean of its input. No
plasticity is needed because the visual statistics are stationary.
This is the simplest possible predictive coding circuit.

Population layout in the RateCircuit:
    indices [0..N-1]     = P (photoreceptors)
    indices [N..2N-1]    = S (sustained)
    indices [2N..3N-1]   = T (transient / bug detectors)

References:
    Lettvin JY et al. Proc IRE 1959, 47(11):1940-1951.
    Srinivasan MV, Laughlin SB, Dubs A. Proc R Soc Lond B 1982, 216:427-459.
"""

import numpy as np

from bravli.simulation.rate_engine import RateCircuit, phi, simulate_rate
from bravli.simulation.visual_world import (
    Arena, render_retina,
    straight_trajectory, circular_trajectory, random_walk_trajectory,
    onset_trajectory,
)


# Population type offsets (multiply by n_pixels to get base index)
P_TYPE = 0  # photoreceptors
S_TYPE = 1  # sustained
T_TYPE = 2  # transient / bug detectors

POPULATION_TYPES = ["P", "S", "T"]

DEFAULT_PARAMS = {
    # Time constants (ms)
    "tau_P": 2.0,       # photoreceptors: fast relay
    "tau_S": 50.0,      # sustained: slow → temporal mean
    "tau_T": 5.0,       # transient: fast → responds to change

    # Connectivity strengths
    "w_PS": 1.0,        # P → S excitatory (center weight)
    "w_PT": 1.0,        # P → T excitatory (local)
    "w_ST": 1.0,        # S → T inhibitory (prediction subtraction)

    # Center-surround (Difference of Gaussians)
    "sigma_center": 1.5,   # center Gaussian width (pixels)
    "sigma_surround": 5.0, # surround Gaussian width (pixels)
    "surround_weight": 0.7, # relative strength of surround vs center
}


def _dog_weights(n_pixels, sigma_c, sigma_s, w_surround):
    """Build Difference-of-Gaussians connectivity matrix.

    Returns a (n_pixels, n_pixels) matrix where entry [i, j] is the
    DoG weight from pixel j to pixel i:

        DoG(d) = exp(-d²/2σ_c²) - w_surround * exp(-d²/2σ_s²)

    Rows are normalized so that each row sums to ~1 (center-dominated).

    Parameters
    ----------
    n_pixels : int
    sigma_c : float
        Center Gaussian width (pixels).
    sigma_s : float
        Surround Gaussian width (pixels).
    w_surround : float
        Relative strength of surround inhibition.

    Returns
    -------
    np.ndarray, shape (n_pixels, n_pixels)
    """
    idx = np.arange(n_pixels)
    d = idx[:, None] - idx[None, :]  # distance matrix
    center = np.exp(-0.5 * (d / sigma_c) ** 2)
    surround = np.exp(-0.5 * (d / sigma_s) ** 2)
    dog = center - w_surround * surround
    # Normalize so that the center peak (diagonal) = 1.
    # This means a delta-function stimulus at pixel j gives S_j ≈ 1.
    # Uniform illumination gives S ≈ (1 - w_surround) * σ_s/σ_c ≈ small,
    # which is exactly the spatial prediction-error property we want.
    # No row-sum normalization — we want the raw DoG shape.
    return dog


def build_frog_retina(n_pixels=50, params=None):
    """Build a frog retinal circuit.

    Creates a RateCircuit with 3*n_pixels populations:
    P[0..N-1], S[N..2N-1], T[2N..3N-1].

    Parameters
    ----------
    n_pixels : int
        Number of spatial locations in the 1D retina.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    RateCircuit
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    N = n_pixels
    n_pop = 3 * N

    # Labels
    labels = (
        [f"P{i}" for i in range(N)]
        + [f"S{i}" for i in range(N)]
        + [f"T{i}" for i in range(N)]
    )

    # Time constants
    tau = np.zeros(n_pop)
    tau[P_TYPE * N:(P_TYPE + 1) * N] = p["tau_P"]
    tau[S_TYPE * N:(S_TYPE + 1) * N] = p["tau_S"]
    tau[T_TYPE * N:(T_TYPE + 1) * N] = p["tau_T"]

    # Transfer functions (all rectified linear)
    transfer_fn = [phi] * n_pop

    # Weight matrix
    W = np.zeros((n_pop, n_pop))

    # P → S: center-surround (DoG) connectivity
    dog = _dog_weights(N, p["sigma_center"], p["sigma_surround"],
                       p["surround_weight"])
    W[S_TYPE * N:(S_TYPE + 1) * N,
      P_TYPE * N:(P_TYPE + 1) * N] = p["w_PS"] * dog

    # P → T: local excitatory (identity / delta connectivity)
    for i in range(N):
        W[T_TYPE * N + i, P_TYPE * N + i] = p["w_PT"]

    # S → T: local inhibitory (prediction subtraction)
    for i in range(N):
        W[T_TYPE * N + i, S_TYPE * N + i] = -p["w_ST"]

    # Bias (none)
    bias = np.zeros(n_pop)

    return RateCircuit(
        n_populations=n_pop,
        labels=labels,
        tau=tau,
        transfer_fn=transfer_fn,
        W=W,
        bias=bias,
    )


def retinal_stimulus(retinal_image, n_pixels):
    """Convert a retinal image to a stimulus array for the circuit.

    The retinal image drives only the photoreceptor populations (P).
    S and T populations receive no direct external input.

    Parameters
    ----------
    retinal_image : np.ndarray, shape (n_pixels, n_steps)
        Luminance at each retinal pixel at each timestep.
    n_pixels : int
        Number of spatial locations.

    Returns
    -------
    np.ndarray, shape (3 * n_pixels, n_steps)
    """
    n_steps = retinal_image.shape[1]
    stim = np.zeros((3 * n_pixels, n_steps))
    stim[:n_pixels, :] = retinal_image  # only P populations receive input
    return stim


def run_frog_experiment(trajectory_type="onset", n_pixels=30,
                        params=None, dt=0.1, duration=500.0,
                        arena_width=100.0, bug_radius=3.0,
                        **trajectory_kwargs):
    """Run a complete frog retina experiment.

    Parameters
    ----------
    trajectory_type : str
        One of "onset", "straight", "circular", "random_walk".
    n_pixels : int
        Retinal resolution.
    params : dict, optional
        Circuit parameter overrides.
    dt : float
        Timestep (ms).
    duration : float
        Simulation duration (ms).
    arena_width : float
        Arena width.
    bug_radius : float
        Bug Gaussian width in arena units.
    **trajectory_kwargs
        Passed to the trajectory generator.

    Returns
    -------
    dict with keys:
        "result": RateResult
        "circuit": RateCircuit
        "retinal_image": np.ndarray (n_pixels, n_steps)
        "bug_positions": np.ndarray (n_steps, 2)
        "arena": Arena
        "n_pixels": int
    """
    n_steps = int(duration / dt)
    arena = Arena(width=arena_width, n_pixels=n_pixels)
    center = arena_width / 2

    # Generate trajectory
    if trajectory_type == "onset":
        onset_step = trajectory_kwargs.get("onset_step", int(200.0 / dt))
        position = trajectory_kwargs.get("position", (center, center))
        bug_positions = onset_trajectory(position, onset_step, n_steps)
    elif trajectory_type == "straight":
        start = trajectory_kwargs.get("start", (10.0, center))
        velocity = trajectory_kwargs.get("velocity", (0.05, 0.0))
        bug_positions = straight_trajectory(start, velocity, n_steps, dt)
    elif trajectory_type == "circular":
        circ_center = trajectory_kwargs.get("center", (center, center))
        radius = trajectory_kwargs.get("radius", 20.0)
        speed = trajectory_kwargs.get("speed", 0.005)
        bug_positions = circular_trajectory(circ_center, radius, speed,
                                            n_steps, dt)
    elif trajectory_type == "random_walk":
        start = trajectory_kwargs.get("start", (center, center))
        step_size = trajectory_kwargs.get("step_size", 0.5)
        seed = trajectory_kwargs.get("seed", None)
        bug_positions = random_walk_trajectory(start, step_size, n_steps,
                                               dt, seed)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Render retinal image
    retinal_image = render_retina(arena, bug_positions, bug_radius)

    # Build circuit and stimulus
    circuit = build_frog_retina(n_pixels, params)
    stim = retinal_stimulus(retinal_image, n_pixels)

    # Simulate
    result = simulate_rate(circuit, duration=duration, dt=dt, stimulus=stim)

    return {
        "result": result,
        "circuit": circuit,
        "retinal_image": retinal_image,
        "bug_positions": bug_positions,
        "arena": arena,
        "n_pixels": n_pixels,
    }


def extract_population(result, pop_type, n_pixels):
    """Extract rate traces for one population type.

    Parameters
    ----------
    result : RateResult
    pop_type : int
        P_TYPE, S_TYPE, or T_TYPE.
    n_pixels : int

    Returns
    -------
    np.ndarray, shape (n_pixels, n_steps)
    """
    start = pop_type * n_pixels
    end = start + n_pixels
    return result.rates[start:end, :]
