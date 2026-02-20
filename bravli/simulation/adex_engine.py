"""Adaptive exponential integrate-and-fire (AdEx) engine.

The AdEx model extends LIF with two features:
1. Exponential spike initiation: voltage grows exponentially near threshold
2. Spike-frequency adaptation: an adaptation current w accumulates with
   each spike and decays slowly, reducing excitability over time

This produces richer firing patterns than LIF: regular spiking, bursting,
adaptation, initial bursting, delayed spiking — depending on parameters.

The equations (Brette & Gerstner 2005):
    C_m * dV/dt = -g_L * (V - V_rest) + g_L * delta_T * exp((V - V_T)/delta_T)
                  - w + g + I_ext
    tau_w * dw/dt = a * (V - V_rest) - w

On spike (V > V_cutoff):
    V -> V_reset
    w -> w + b

Parameters:
    delta_T : sharpness of exponential spike initiation (mV)
    a : subthreshold adaptation conductance (nS)
    b : spike-triggered adaptation increment (pA)
    tau_w : adaptation time constant (ms)
    V_T : threshold for exponential term (mV), distinct from spike cutoff

References:
    Brette R, Gerstner W (2005). J Comp Neurosci 19(2):175-197.
    Naud R et al. (2008). Biol Cybern 99(4-5):335-347.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bravli.simulation.engine import SimulationResult
from bravli.utils import get_logger

LOG = get_logger("simulation.adex")


@dataclass(frozen=True)
class AdExParams:
    """Parameters for AdEx neurons.

    These augment the LIF parameters with adaptation and exponential terms.
    Can be per-neuron (arrays) or global (scalars).
    """
    delta_t: float = 2.0       # Exponential slope factor (mV)
    a: float = 0.0             # Subthreshold adaptation (nS)
    b: float = 0.0             # Spike-triggered adaptation (pA -> mV in our units)
    tau_w: float = 100.0       # Adaptation time constant (ms)
    v_cutoff: float = 0.0      # Spike cutoff potential (mV), above V_thresh


# Predefined AdEx parameter sets for Drosophila cell types
ADEX_PRESETS = {
    "regular_spiking": AdExParams(
        delta_t=2.0, a=0.0, b=0.5, tau_w=100.0,
    ),
    "adapting": AdExParams(
        delta_t=2.0, a=0.5, b=1.0, tau_w=100.0,
    ),
    "bursting": AdExParams(
        delta_t=2.0, a=0.0, b=2.0, tau_w=20.0,
    ),
    "fast_spiking": AdExParams(
        delta_t=1.0, a=0.0, b=0.0, tau_w=50.0,
    ),
}


def simulate_adex(circuit, adex_params=None, duration=1000.0, dt=0.1,
                  stimulus=None, record_v=False, record_idx=None,
                  seed=None, noise_sigma=0.0, release_prob=1.0):
    """Run an AdEx simulation on a circuit.

    Uses the same Circuit dataclass as the LIF engine. The AdEx parameters
    augment the circuit's LIF parameters with adaptation and exponential
    spike initiation.

    Parameters
    ----------
    circuit : Circuit
        Neural circuit (same as LIF engine).
    adex_params : AdExParams or dict, optional
        AdEx-specific parameters. If dict, maps neuron index to AdExParams.
        If None, uses default regular_spiking preset.
    duration : float
        Simulation duration (ms).
    dt : float
        Timestep (ms).
    stimulus : np.ndarray, optional
        External stimulus, shape (n_neurons, n_steps).
    record_v : bool
        Record voltage traces.
    record_idx : array-like, optional
        Neurons to record.
    seed : int, optional
        Random seed.
    noise_sigma : float
        Intrinsic noise (mV/sqrt(ms)).
    release_prob : float or np.ndarray
        Synaptic release probability.

    Returns
    -------
    SimulationResult
        Same result format as LIF engine (compatible with all analysis tools).
    """
    rng = np.random.RandomState(seed)

    n = circuit.n_neurons
    n_steps = int(duration / dt)

    # Parse AdEx parameters
    if adex_params is None:
        adex_params = ADEX_PRESETS["regular_spiking"]

    if isinstance(adex_params, AdExParams):
        delta_t = np.full(n, adex_params.delta_t)
        a_adapt = np.full(n, adex_params.a)
        b_adapt = np.full(n, adex_params.b)
        tau_w = np.full(n, adex_params.tau_w)
        v_cutoff = np.full(n, adex_params.v_cutoff)
    elif isinstance(adex_params, dict):
        # Per-neuron or per-group params
        default = ADEX_PRESETS["regular_spiking"]
        delta_t = np.full(n, default.delta_t)
        a_adapt = np.full(n, default.a)
        b_adapt = np.full(n, default.b)
        tau_w = np.full(n, default.tau_w)
        v_cutoff = np.full(n, default.v_cutoff)
        for idx, params in adex_params.items():
            if isinstance(idx, (int, np.integer)):
                delta_t[idx] = params.delta_t
                a_adapt[idx] = params.a
                b_adapt[idx] = params.b
                tau_w[idx] = params.tau_w
                v_cutoff[idx] = params.v_cutoff
    else:
        raise ValueError(f"adex_params must be AdExParams or dict, got {type(adex_params)}")

    # Stochastic parameters
    use_noise = noise_sigma > 0.0
    noise_scale = noise_sigma * np.sqrt(dt) if use_noise else 0.0
    use_failure = not (np.isscalar(release_prob) and release_prob >= 1.0)
    release_prob_arr = np.asarray(release_prob) if use_failure else None

    # State arrays
    v = circuit.v_rest.copy()
    g = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)  # adaptation current
    refractory_until = np.full(n, -1.0, dtype=np.float64)

    # Integration constants
    tau_m = circuit.tau_m
    tau_syn = circuit.tau_syn
    if np.isscalar(tau_syn):
        decay_g = 1.0 - dt / tau_syn
    else:
        decay_g = 1.0 - dt / 5.0

    # Delay buffer
    delay = circuit.delay_steps if np.isscalar(circuit.delay_steps) else int(np.max(circuit.delay_steps))
    spike_buffer = np.zeros((delay + 1, n), dtype=bool)

    # Spike recording
    spike_times = [[] for _ in range(n)]

    # Voltage recording
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

    # Connectivity
    pre_idx = circuit.pre_idx
    post_idx = circuit.post_idx
    weights = circuit.weights

    LOG.info("Starting AdEx simulation: %d neurons, %d synapses, %.0f ms",
             n, circuit.n_synapses, duration)

    # Main loop
    for step in range(n_steps):
        t = step * dt

        # 1. Synaptic decay
        g *= decay_g

        # 2. Deliver spikes (with optional release failure)
        buf_slot = step % (delay + 1)
        delayed_spikes = spike_buffer[buf_slot]
        if np.any(delayed_spikes):
            firing_mask = delayed_spikes[pre_idx]
            if np.any(firing_mask):
                if use_failure:
                    firing_indices = np.where(firing_mask)[0]
                    if np.isscalar(release_prob):
                        released = rng.random(len(firing_indices)) < release_prob
                    else:
                        released = rng.random(len(firing_indices)) < release_prob_arr[firing_indices]
                    transmitted = firing_indices[released]
                    if len(transmitted) > 0:
                        np.add.at(g, post_idx[transmitted], weights[transmitted])
                else:
                    np.add.at(g, post_idx[firing_mask], weights[firing_mask])
        spike_buffer[buf_slot] = False

        # 3. External stimulus
        if stimulus is not None:
            g += stimulus[:, step]

        # 3.5. Intrinsic noise
        if use_noise:
            g += noise_scale * rng.randn(n)

        # 4. Voltage update (AdEx Euler)
        not_refractory = (t >= refractory_until)

        # Exponential term: delta_T * exp((V - V_T) / delta_T)
        # V_T is the threshold for the exponential (use circuit.v_thresh)
        v_diff = v - circuit.v_thresh
        # Clamp to prevent overflow in exp
        exp_term = delta_t * np.exp(np.minimum(v_diff / delta_t, 20.0))

        # dV/dt = (-(V - V_rest) + exp_term + g - w) / tau_m
        dv = (dt / tau_m) * (-(v - circuit.v_rest) + exp_term + g - w)
        v += dv * not_refractory

        # Adaptation update: dw/dt = (a * (V - V_rest) - w) / tau_w
        dw = (dt / tau_w) * (a_adapt * (v - circuit.v_rest) - w)
        w += dw

        # 5. Spike detection (at cutoff, which is above V_thresh)
        spike_thresh = circuit.v_thresh + v_cutoff
        spiked = (v >= spike_thresh) & not_refractory
        if np.any(spiked):
            spike_indices = np.where(spiked)[0]
            for idx in spike_indices:
                spike_times[idx].append(t)

            # Reset
            v[spiked] = circuit.v_reset[spiked]
            g[spiked] = 0.0
            w[spiked] += b_adapt[spiked]  # adaptation increment

            # Refractory
            refractory_until[spiked] = t + circuit.t_ref[spiked]

            # Delay buffer
            future_step = (step + delay) % (delay + 1)
            spike_buffer[future_step, spike_indices] = True

        # 6. Record traces
        if record_v and len(record_idx) > 0:
            v_trace[:, step] = v[record_idx]
            g_trace[:, step] = g[record_idx]

    # Convert spike times
    spike_times = [np.array(st) for st in spike_times]

    total_spikes = sum(len(st) for st in spike_times)
    LOG.info("AdEx simulation complete: %d spikes, mean rate %.2f Hz",
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
