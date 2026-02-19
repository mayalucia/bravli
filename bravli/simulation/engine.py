"""Pure-numpy LIF simulation engine.

Implements the Shiu et al. (2024) LIF model with Euler integration.
Supports heterogeneous neuron parameters and synaptic delays.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bravli.utils import get_logger

LOG = get_logger("simulation.engine")


@dataclass
class SimulationResult:
    """Results from a simulation run.

    Attributes
    ----------
    spike_times : list of np.ndarray
        spike_times[i] is an array of spike times (ms) for neuron i.
    v_trace : Optional[np.ndarray]
        Membrane potential trace, shape (n_recorded, n_steps).
        Only populated if record_v=True.
    g_trace : Optional[np.ndarray]
        Synaptic conductance trace, shape (n_recorded, n_steps).
    recorded_idx : np.ndarray
        Indices of neurons whose traces were recorded.
    dt : float
        Timestep used (ms).
    duration : float
        Total simulation time (ms).
    n_neurons : int
        Number of neurons.
    """
    spike_times: list
    v_trace: Optional[np.ndarray] = None
    g_trace: Optional[np.ndarray] = None
    recorded_idx: np.ndarray = None
    dt: float = 0.1
    duration: float = 1000.0
    n_neurons: int = 0

    @property
    def n_spikes(self):
        """Total number of spikes across all neurons."""
        return sum(len(st) for st in self.spike_times)

    def mean_rate(self):
        """Mean firing rate across all neurons (Hz)."""
        duration_s = self.duration / 1000.0
        return self.n_spikes / (self.n_neurons * duration_s) if self.n_neurons > 0 else 0.0

    def neuron_rates(self):
        """Per-neuron firing rates (Hz)."""
        duration_s = self.duration / 1000.0
        return np.array([len(st) / duration_s for st in self.spike_times])


def simulate(circuit, duration=1000.0, dt=0.1, stimulus=None,
             record_v=False, record_idx=None, seed=None,
             plasticity_fn=None):
    """Run a LIF simulation.

    Parameters
    ----------
    circuit : Circuit
        The neural circuit to simulate.
    duration : float
        Simulation duration (ms).
    dt : float
        Timestep (ms).
    stimulus : np.ndarray, optional
        External current injection, shape (n_neurons, n_steps) in mV.
        If None, no external input.
    record_v : bool
        If True, record membrane potential traces for selected neurons.
    record_idx : array-like, optional
        Indices of neurons to record. If None and record_v=True,
        records the first 100 neurons.
    seed : int, optional
        Random seed (for reproducibility of Poisson stimuli).
    plasticity_fn : callable, optional
        Called after spike detection each timestep:
        plasticity_fn(step, t, dt, spiked, v, g, circuit)
        May mutate circuit.weights in-place for online learning.
        If None, no plasticity (default — all existing behavior unchanged).

    Returns
    -------
    SimulationResult
    """
    if seed is not None:
        np.random.seed(seed)

    n = circuit.n_neurons
    n_steps = int(duration / dt)

    # --- State arrays ---
    v = circuit.v_rest.copy()
    g = np.zeros(n, dtype=np.float64)
    refractory_until = np.full(n, -1.0, dtype=np.float64)

    # --- Precompute integration constants ---
    dt_over_tau_m = dt / circuit.tau_m  # per-neuron (array or scalar)

    # Handle tau_syn: scalar or per-synapse
    tau_syn = circuit.tau_syn
    if np.isscalar(tau_syn):
        decay_g = 1.0 - dt / tau_syn
    else:
        # Per-synapse tau_syn: use default 5.0 for the g decay
        decay_g = 1.0 - dt / 5.0

    # --- Delay buffer ---
    delay = circuit.delay_steps if np.isscalar(circuit.delay_steps) else int(np.max(circuit.delay_steps))
    spike_buffer = np.zeros((delay + 1, n), dtype=bool)

    # --- Spike recording ---
    spike_times = [[] for _ in range(n)]

    # --- Voltage recording ---
    if record_v:
        if record_idx is None:
            record_idx = np.arange(min(100, n))
        else:
            record_idx = np.asarray(record_idx)
        v_trace = np.zeros((len(record_idx), n_steps), dtype=np.float32)
        g_trace = np.zeros((len(record_idx), n_steps), dtype=np.float32)
    else:
        record_idx = np.array([], dtype=int)
        v_trace = None
        g_trace = None

    # --- Connectivity arrays ---
    pre_idx = circuit.pre_idx
    post_idx = circuit.post_idx
    weights = circuit.weights

    LOG.info("Starting simulation: %d neurons, %d synapses, %.0f ms, dt=%.1f ms",
             n, circuit.n_synapses, duration, dt)

    # --- Main loop ---
    for step in range(n_steps):
        t = step * dt

        # 1. Synaptic decay
        g *= decay_g

        # 2. Deliver spikes from delay buffer
        buf_slot = step % (delay + 1)
        delayed_spikes = spike_buffer[buf_slot]
        if np.any(delayed_spikes):
            firing_mask = delayed_spikes[pre_idx]
            if np.any(firing_mask):
                np.add.at(g, post_idx[firing_mask], weights[firing_mask])
        spike_buffer[buf_slot] = False  # clear after reading

        # 3. External stimulus
        if stimulus is not None:
            g += stimulus[:, step]

        # 4. Voltage update (Euler)
        not_refractory = (t >= refractory_until)
        dv = dt_over_tau_m * (-(v - circuit.v_rest) + g)
        v += dv * not_refractory

        # 5. Spike detection
        spiked = (v >= circuit.v_thresh) & not_refractory
        if np.any(spiked):
            spike_indices = np.where(spiked)[0]
            for idx in spike_indices:
                spike_times[idx].append(t)

            # Reset
            v[spiked] = circuit.v_reset[spiked]
            g[spiked] = 0.0

            # Refractory
            refractory_until[spiked] = t + circuit.t_ref[spiked]

            # Insert into delay buffer
            future_step = (step + delay) % (delay + 1)
            spike_buffer[future_step, spike_indices] = True

        # 6. Plasticity update (optional)
        if plasticity_fn is not None:
            plasticity_fn(step, t, dt, spiked, v, g, circuit)

        # 7. Record traces
        if record_v and len(record_idx) > 0:
            v_trace[:, step] = v[record_idx]
            g_trace[:, step] = g[record_idx]

    # Convert spike times to arrays
    spike_times = [np.array(st) for st in spike_times]

    total_spikes = sum(len(st) for st in spike_times)
    LOG.info("Simulation complete: %d spikes, mean rate %.2f Hz",
             total_spikes,
             total_spikes / (n * duration / 1000.0) if n > 0 else 0.0)

    return SimulationResult(
        spike_times=spike_times,
        v_trace=v_trace,
        g_trace=g_trace,
        recorded_idx=record_idx,
        dt=dt,
        duration=duration,
        n_neurons=n,
    )
