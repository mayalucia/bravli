"""Stimulus protocols for LIF simulation.

Each function returns a stimulus array of shape (n_neurons, n_steps)
representing external current injection in mV at each timestep.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class StimulusProtocol:
    """Description of a stimulus for provenance tracking.

    Attributes
    ----------
    name : str
        Protocol name.
    target_indices : np.ndarray
        Neuron indices receiving stimulus.
    params : dict
        Protocol parameters.
    """
    name: str
    target_indices: np.ndarray
    params: dict


def poisson_stimulus(n_neurons, n_steps, target_indices, rate_hz=100.0,
                     weight=68.75, dt=0.1, seed=None):
    """Generate Poisson spike train input.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons in the circuit.
    n_steps : int
        Number of simulation timesteps.
    target_indices : array-like
        Indices of neurons receiving Poisson input.
    rate_hz : float
        Poisson firing rate (Hz) per target neuron.
    weight : float
        Weight of each Poisson spike (mV). Shiu uses 250 * 0.275 = 68.75 mV.
    dt : float
        Timestep (ms).
    seed : int, optional
        Random seed.

    Returns
    -------
    stimulus : np.ndarray
        Shape (n_neurons, n_steps).
    protocol : StimulusProtocol
        Description for provenance.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    target_indices = np.asarray(target_indices)
    stimulus = np.zeros((n_neurons, n_steps), dtype=np.float64)

    # Probability of spike per timestep
    p_spike = rate_hz * (dt / 1000.0)

    # Generate spikes
    spikes = rng.random((len(target_indices), n_steps)) < p_spike
    stimulus[target_indices] = spikes * weight

    protocol = StimulusProtocol(
        name="poisson",
        target_indices=target_indices,
        params={"rate_hz": rate_hz, "weight": weight, "dt": dt, "seed": seed},
    )

    return stimulus, protocol


def step_stimulus(n_neurons, n_steps, target_indices, amplitude=5.0,
                  start_ms=100.0, end_ms=500.0, dt=0.1):
    """Generate a step current injection.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons.
    n_steps : int
        Number of timesteps.
    target_indices : array-like
        Neurons receiving current.
    amplitude : float
        Current amplitude (mV equivalent).
    start_ms, end_ms : float
        Start and end time of the step (ms).
    dt : float
        Timestep (ms).

    Returns
    -------
    stimulus : np.ndarray
        Shape (n_neurons, n_steps).
    protocol : StimulusProtocol
    """
    target_indices = np.asarray(target_indices)
    stimulus = np.zeros((n_neurons, n_steps), dtype=np.float64)

    start_step = int(start_ms / dt)
    end_step = min(int(end_ms / dt), n_steps)

    stimulus[np.ix_(target_indices, range(start_step, end_step))] = amplitude

    protocol = StimulusProtocol(
        name="step",
        target_indices=target_indices,
        params={"amplitude": amplitude, "start_ms": start_ms, "end_ms": end_ms},
    )

    return stimulus, protocol


def pulse_stimulus(n_neurons, n_steps, target_indices, amplitude=10.0,
                   pulse_ms=1.0, time_ms=100.0, dt=0.1):
    """Generate a brief current pulse.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons.
    n_steps : int
        Number of timesteps.
    target_indices : array-like
        Neurons receiving pulse.
    amplitude : float
        Pulse amplitude (mV equivalent).
    pulse_ms : float
        Pulse duration (ms).
    time_ms : float
        Time of pulse onset (ms).
    dt : float
        Timestep (ms).

    Returns
    -------
    stimulus : np.ndarray
        Shape (n_neurons, n_steps).
    protocol : StimulusProtocol
    """
    target_indices = np.asarray(target_indices)
    stimulus = np.zeros((n_neurons, n_steps), dtype=np.float64)

    start_step = int(time_ms / dt)
    end_step = min(start_step + int(pulse_ms / dt), n_steps)

    if start_step < n_steps:
        stimulus[np.ix_(target_indices, range(start_step, end_step))] = amplitude

    protocol = StimulusProtocol(
        name="pulse",
        target_indices=target_indices,
        params={"amplitude": amplitude, "pulse_ms": pulse_ms, "time_ms": time_ms},
    )

    return stimulus, protocol


def combine_stimuli(*stimuli):
    """Sum multiple stimulus arrays.

    Parameters
    ----------
    *stimuli : np.ndarray
        Stimulus arrays of the same shape.

    Returns
    -------
    np.ndarray
        Element-wise sum.
    """
    result = stimuli[0].copy()
    for s in stimuli[1:]:
        result += s
    return result
