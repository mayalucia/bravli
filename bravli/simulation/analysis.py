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


def population_sparseness(rates):
    """Compute Treves-Rolls population sparseness.

    S = (mean(r))^2 / mean(r^2)

    S = 1 means all neurons fire at the same rate (dense).
    S -> 1/N means exactly one neuron fires (maximally sparse).

    Parameters
    ----------
    rates : np.ndarray
        Per-neuron firing rates (Hz). Shape (n_neurons,).

    Returns
    -------
    float
        Sparseness in [0, 1]. Lower = sparser.
    """
    rates = np.asarray(rates, dtype=np.float64)
    if len(rates) == 0:
        return 0.0
    mean_r = np.mean(rates)
    mean_r2 = np.mean(rates ** 2)
    if mean_r2 == 0:
        return 0.0
    return (mean_r ** 2) / mean_r2


def lifetime_sparseness(result, neuron_indices=None, bin_ms=50.0):
    """Per-neuron lifetime sparseness across time bins.

    For each neuron, computes Treves-Rolls sparseness over its
    binned spike count vector. A neuron that fires uniformly in
    all bins has S=1; one that fires in a single bin has S~1/N_bins.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    neuron_indices : array-like, optional
        Subset of neurons. If None, all neurons.
    bin_ms : float
        Time bin width (ms).

    Returns
    -------
    np.ndarray
        Per-neuron lifetime sparseness.
    """
    if neuron_indices is None:
        neuron_indices = range(result.n_neurons)
    neuron_indices = np.asarray(neuron_indices)

    n_bins = max(1, int(result.duration / bin_ms))
    sparsenesses = np.zeros(len(neuron_indices))

    for j, i in enumerate(neuron_indices):
        st = result.spike_times[i]
        if len(st) == 0:
            sparsenesses[j] = 0.0
            continue
        bins = np.clip((st / bin_ms).astype(int), 0, n_bins - 1)
        counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
        mean_c = np.mean(counts)
        mean_c2 = np.mean(counts ** 2)
        if mean_c2 == 0:
            sparsenesses[j] = 0.0
        else:
            sparsenesses[j] = (mean_c ** 2) / mean_c2

    return sparsenesses


def active_fraction_by_group(result, groups, threshold_hz=1.0):
    """Fraction of neurons active per named group.

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    groups : dict of str -> array-like
        Mapping from group name to neuron indices.
    threshold_hz : float
        Minimum rate to count as "active".

    Returns
    -------
    dict
        Group name -> (n_active, n_total, fraction).
    """
    rates = firing_rates(result)
    out = {}
    for name, indices in groups.items():
        indices = np.asarray(indices)
        if len(indices) == 0:
            out[name] = (0, 0, 0.0)
            continue
        group_rates = rates[indices]
        n_active = int(np.sum(group_rates > threshold_hz))
        out[name] = (n_active, len(indices), n_active / len(indices))
    return out


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


def weight_evolution(snapshots, times):
    """Summarize weight trajectory across plasticity snapshots.

    Parameters
    ----------
    snapshots : list of np.ndarray
        Weight vectors at each snapshot (from ThreeFactorSTDP).
    times : list of float
        Snapshot times (ms).

    Returns
    -------
    pd.DataFrame
        Columns: time_ms, mean_weight, std_weight, min_weight, max_weight,
        frac_depressed (fraction of synapses below initial value).
    """
    if not snapshots:
        return pd.DataFrame(columns=[
            "time_ms", "mean_weight", "std_weight", "min_weight",
            "max_weight", "frac_depressed",
        ])

    initial = snapshots[0]
    rows = []
    for t, w in zip(times, snapshots):
        rows.append({
            "time_ms": t,
            "mean_weight": float(np.mean(w)),
            "std_weight": float(np.std(w)),
            "min_weight": float(np.min(w)),
            "max_weight": float(np.max(w)),
            "frac_depressed": float(np.mean(w < initial - 1e-10)),
        })
    return pd.DataFrame(rows)


def mbon_response_change(pre_rates, post_rates, mbon_indices):
    """Learning index per MBON: (pre - post) / (pre + post).

    Positive values indicate depression (post < pre = learned avoidance).

    Parameters
    ----------
    pre_rates : np.ndarray
        Per-neuron firing rates before training.
    post_rates : np.ndarray
        Per-neuron firing rates after training.
    mbon_indices : array-like
        Indices of MBON neurons.

    Returns
    -------
    np.ndarray
        Learning index per MBON. Range [-1, 1].
    """
    mbon_indices = np.asarray(mbon_indices)
    pre = pre_rates[mbon_indices]
    post = post_rates[mbon_indices]
    denom = pre + post
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (pre - post) / denom, 0.0)


def performance_index(cs_plus_rates, cs_minus_rates, mbon_indices):
    """Differential MBON response: CS+ vs CS-.

    PI = (response_CS- - response_CS+) / (response_CS- + response_CS+)

    Positive PI means CS+ drives less MBON activity than CS-,
    indicating learned aversion to CS+.

    Parameters
    ----------
    cs_plus_rates : np.ndarray
        Per-neuron rates during CS+ presentation (post-training).
    cs_minus_rates : np.ndarray
        Per-neuron rates during CS- presentation.
    mbon_indices : array-like
        MBON neuron indices.

    Returns
    -------
    np.ndarray
        Performance index per MBON.
    """
    mbon_indices = np.asarray(mbon_indices)
    plus = cs_plus_rates[mbon_indices]
    minus = cs_minus_rates[mbon_indices]
    denom = minus + plus
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (minus - plus) / denom, 0.0)
