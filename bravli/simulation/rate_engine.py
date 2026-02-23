"""Rate-based dynamics engine for population-level neural circuits.

Complements the spiking LIF engine (engine.py) with a continuous
rate-dynamics simulator. Populations are the unit of simulation,
not individual neurons. Each population has a transfer function,
time constant, and connectivity to other populations.

The key difference from spiking models: rates are continuous
variables, there are no spikes, and connectivity can include
divisive normalization (as in PV → pyramidal gain control).

Equation for population i:
    τ_i dr_i/dt = -r_i + φ_i(Σ_j W_ij r_j + stimulus_i + bias_i)

With optional divisive normalization on target i from source k:
    τ_i dr_i/dt = -r_i + φ_i(I_num / (I_0 + w_div * r_k))

Reference:
    Wilson HR & Cowan JD (1972). Biophys J 12:1-24.
    Wilmes KA et al. (2025). eLife 14:e95127.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from bravli.utils import get_logger

LOG = get_logger("simulation.rate_engine")


# ---------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------

def phi(x):
    """Rectified linear activation, clipped at 20.

    φ(x) = max(0, min(x, 20))

    Used for excitatory and SST populations.
    """
    return np.clip(x, 0.0, 20.0)


def phi_pv(x):
    """Quadratic rectified activation, clipped at 20.

    φ_PV(x) = max(0, min(x², 20))

    Critical for variance learning: the squared nonlinearity makes
    PV firing rate proportional to the second moment of input,
    enabling it to track stimulus variance. Replacing this with
    a linear function breaks variance learning.

    Reference: Wilmes & Senn (2025), Eq. 11.
    """
    return np.clip(np.where(x > 0, x ** 2, 0.0), 0.0, 20.0)


def phi_power(x, k=2.0):
    """Power-law rectified activation, clipped at 20.

    φ_k(x) = max(0, min(x^k, 20))

    Generalization used for dendritic nonlinearity in E+/E- neurons.
    """
    return np.clip(np.where(x > 0, x ** k, 0.0), 0.0, 20.0)


# ---------------------------------------------------------------------------
# Circuit dataclass
# ---------------------------------------------------------------------------

@dataclass
class RateCircuit:
    """A circuit for rate-based simulation.

    Unlike Circuit (spiking), populations are the unit, not individual
    neurons. Each population has a transfer function, time constant,
    and additive connectivity to other populations.

    Parameters
    ----------
    n_populations : int
        Number of populations.
    labels : list of str
        Human-readable population names (e.g., ["E+", "E-", "SST+", "PV+", "R"]).
    tau : np.ndarray
        Time constants (ms), shape (n_pop,).
    transfer_fn : list of Callable
        Activation function per population. Each takes and returns a scalar or array.
    W : np.ndarray
        Additive weight matrix, shape (n_pop, n_pop). W[i, j] is the weight
        from population j to population i.
    bias : np.ndarray
        Constant input per population, shape (n_pop,).
    divisive : dict, optional
        Divisive normalization connections. Maps target population index to
        (source_index, w_divisive, I_0). The target's input becomes:
            φ(I_additive / (I_0 + w_div * r_source))
        instead of φ(I_additive).
    """
    n_populations: int
    labels: List[str]
    tau: np.ndarray
    transfer_fn: List[Callable]
    W: np.ndarray
    bias: np.ndarray
    divisive: Dict[int, Tuple[int, float, float]] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.labels) == self.n_populations
        assert self.tau.shape == (self.n_populations,)
        assert len(self.transfer_fn) == self.n_populations
        assert self.W.shape == (self.n_populations, self.n_populations)
        assert self.bias.shape == (self.n_populations,)

    def population_index(self, label):
        """Get population index by label."""
        return self.labels.index(label)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RateResult:
    """Result of a rate simulation.

    Analogous to SimulationResult for spiking models, but with
    continuous rate traces instead of spike times.

    Parameters
    ----------
    rates : np.ndarray
        Rate traces, shape (n_populations, n_steps).
    time : np.ndarray
        Time points (ms), shape (n_steps,).
    dt : float
        Timestep (ms).
    duration : float
        Total duration (ms).
    labels : list of str
        Population labels.
    """
    rates: np.ndarray
    time: np.ndarray
    dt: float
    duration: float
    labels: List[str]

    @property
    def n_populations(self):
        return self.rates.shape[0]

    @property
    def n_steps(self):
        return self.rates.shape[1]

    def rate(self, label):
        """Get rate trace for a population by label."""
        idx = self.labels.index(label)
        return self.rates[idx]

    def mean_rate(self, label, start=None, end=None):
        """Mean rate of a population over a time window."""
        trace = self.rate(label)
        t = self.time
        mask = np.ones(len(t), dtype=bool)
        if start is not None:
            mask &= t >= start
        if end is not None:
            mask &= t <= end
        return float(np.mean(trace[mask]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_rate(circuit, duration=1000.0, dt=0.1,
                  stimulus=None, plasticity_fn=None):
    """Euler integration of rate dynamics.

    For each population i, at each timestep:
        I_add_i = Σ_j W[i,j] * r_j + stimulus_i + bias_i

    If population i has divisive normalization from source k:
        input_i = I_add_i / (I_0 + w_div * r_k)
    else:
        input_i = I_add_i

    Then:
        τ_i * dr_i/dt = -r_i + φ_i(input_i)

    Parameters
    ----------
    circuit : RateCircuit
        The population-level circuit.
    duration : float
        Simulation duration (ms).
    dt : float
        Timestep (ms). Should be << min(tau) for stability.
    stimulus : np.ndarray, optional
        External stimulus, shape (n_populations, n_steps).
    plasticity_fn : callable, optional
        Called each step as plasticity_fn(step, t, dt, rates, circuit).
        May mutate circuit.W in-place.

    Returns
    -------
    RateResult
    """
    n = circuit.n_populations
    n_steps = int(duration / dt)

    # State
    r = np.zeros(n, dtype=np.float64)
    rates = np.zeros((n, n_steps), dtype=np.float64)
    time = np.arange(n_steps) * dt

    LOG.info("Starting rate simulation: %d populations, %.0f ms, dt=%.2f ms",
             n, duration, dt)

    for step in range(n_steps):
        # Additive input: W @ r + bias + stimulus
        I_add = circuit.W @ r + circuit.bias
        if stimulus is not None:
            I_add += stimulus[:, step]

        # Compute activation for each population
        activated = np.zeros(n)
        for i in range(n):
            if i in circuit.divisive:
                src, w_div, I_0 = circuit.divisive[i]
                denom = I_0 + w_div * r[src]
                activated[i] = circuit.transfer_fn[i](I_add[i] / denom)
            else:
                activated[i] = circuit.transfer_fn[i](I_add[i])

        # Euler update: tau * dr/dt = -r + activated
        dr = (dt / circuit.tau) * (-r + activated)
        r += dr

        # Enforce non-negativity
        r = np.maximum(r, 0.0)

        # Record
        rates[:, step] = r

        # Plasticity
        if plasticity_fn is not None:
            plasticity_fn(step, step * dt, dt, r, circuit)

    LOG.info("Rate simulation complete. Final rates: %s",
             {label: f"{rates[i, -1]:.3f}"
              for i, label in enumerate(circuit.labels)})

    return RateResult(
        rates=rates,
        time=time,
        dt=dt,
        duration=duration,
        labels=circuit.labels,
    )
