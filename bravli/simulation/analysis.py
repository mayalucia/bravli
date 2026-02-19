"""Post-simulation analysis tools.

Functions for computing firing rates, spike rasters, E/I balance,
and activity statistics from SimulationResult objects.
"""

import numpy as np
import pandas as pd


def firing_rates(result, time_window=None):
    """Compute per-neuron firing rates.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    time_window : tuple of float, optional
        (start_ms, end_ms) to restrict rate computation.

    Returns
    -------
    np.ndarray
        Firing rate per neuron (Hz).
    """
    if time_window is not None:
        t0, t1 = time_window
        duration_s = (t1 - t0) / 1000.0
        rates = np.array([
            np.sum((st >= t0) & (st < t1)) / duration_s
            for st in result.spike_times
        ])
    else:
        duration_s = result.duration / 1000.0
        rates = np.array([len(st) / duration_s for st in result.spike_times])

    return rates


def spike_raster(result, neuron_indices=None, time_window=None):
    """Extract spike raster data for plotting.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    neuron_indices : array-like, optional
        Subset of neurons. If None, all neurons.
    time_window : tuple of float, optional
        (start_ms, end_ms) to restrict.

    Returns
    -------
    times : np.ndarray
        Spike times (ms).
    neurons : np.ndarray
        Neuron indices for each spike.
    """
    if neuron_indices is None:
        neuron_indices = range(result.n_neurons)

    times = []
    neurons = []
    for i in neuron_indices:
        st = result.spike_times[i]
        if time_window is not None:
            t0, t1 = time_window
            st = st[(st >= t0) & (st < t1)]
        times.append(st)
        neurons.append(np.full(len(st), i))

    if times:
        return np.concatenate(times), np.concatenate(neurons)
    return np.array([]), np.array([])


def ei_balance(result, circuit):
    """Compute excitatory/inhibitory balance per neuron.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    circuit : Circuit
        The circuit that was simulated.

    Returns
    -------
    pd.DataFrame
        Per-neuron: total_excitatory_input, total_inhibitory_input,
        ei_ratio, firing_rate_hz.
    """
    rates = firing_rates(result)

    # Compute total signed input per postsynaptic neuron
    exc_input = np.zeros(circuit.n_neurons)
    inh_input = np.zeros(circuit.n_neurons)

    pre_rates = rates[circuit.pre_idx]
    weighted = pre_rates * circuit.weights

    exc_mask = circuit.weights > 0
    inh_mask = circuit.weights < 0

    np.add.at(exc_input, circuit.post_idx[exc_mask], weighted[exc_mask])
    np.add.at(inh_input, circuit.post_idx[inh_mask], np.abs(weighted[inh_mask]))

    with np.errstate(divide="ignore", invalid="ignore"):
        ei_ratio = np.where(inh_input > 0, exc_input / inh_input, np.inf)

    return pd.DataFrame({
        "exc_input": exc_input,
        "inh_input": inh_input,
        "ei_ratio": ei_ratio,
        "firing_rate_hz": rates,
    })


def active_fraction(result, threshold_hz=1.0, time_window=None):
    """Fraction of neurons firing above a threshold rate.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    threshold_hz : float
        Minimum rate to count as "active".
    time_window : tuple of float, optional
        Restrict to time window.

    Returns
    -------
    float
        Fraction of neurons active.
    """
    rates = firing_rates(result, time_window=time_window)
    return np.mean(rates > threshold_hz)


def population_rate(result, bin_ms=10.0):
    """Compute population-averaged firing rate over time.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    bin_ms : float
        Time bin width (ms).

    Returns
    -------
    times : np.ndarray
        Bin centers (ms).
    rates : np.ndarray
        Population rate (Hz) per bin.
    """
    n_bins = int(result.duration / bin_ms)
    counts = np.zeros(n_bins)

    for st in result.spike_times:
        if len(st) > 0:
            bins = np.clip((st / bin_ms).astype(int), 0, n_bins - 1)
            np.add.at(counts, bins, 1)

    bin_s = bin_ms / 1000.0
    rates = counts / (result.n_neurons * bin_s)
    times = np.arange(n_bins) * bin_ms + bin_ms / 2

    return times, rates
