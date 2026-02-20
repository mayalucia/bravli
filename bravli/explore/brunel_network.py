"""Brunel phase diagram: network regime classification.

Implements the Brunel (2000) random LIF network to:
1. Reproduce the four dynamical regimes (SR, SI, AR, AI)
2. Sweep the (g, eta) phase diagram
3. Classify the FlyWire connectome's dynamical regime

The four regimes:
- SR (synchronous regular): neurons fire in lockstep
- SI (synchronous irregular): population oscillations, individual irregularity
- AR (asynchronous regular): steady global rate, regular individual firing
- AI (asynchronous irregular): steady global rate, irregular individual firing

The AI regime corresponds to the balanced E/I state (van Vreeswijk & Sompolinsky 1996).

References:
    Brunel N (2000). J Comp Neurosci 8(3):183-208.
    van Vreeswijk C, Sompolinsky H (1996). Science 274:1724-1726.
"""

import numpy as np
import pandas as pd

from bravli.simulation.circuit import Circuit
from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import poisson_stimulus
from bravli.simulation.analysis import firing_rates, population_rate

from bravli.utils import get_logger

LOG = get_logger("explore.brunel")


# ---------------------------------------------------------------------------
# Brunel random network construction
# ---------------------------------------------------------------------------

def build_brunel_network(n_excitatory=10000, gamma=0.25, epsilon=0.1,
                         g=4.0, eta=2.0, j=0.1, tau_m=20.0, t_ref=2.0,
                         v_rest=0.0, v_thresh=20.0, v_reset=10.0,
                         delay_ms=1.5, dt=0.1, seed=42):
    """Build a Brunel-style random LIF network.

    Parameters
    ----------
    n_excitatory : int
        Number of excitatory neurons.
    gamma : float
        Ratio N_I / N_E. Default 0.25 (80/20 E/I split).
    epsilon : float
        Connection probability.
    g : float
        Relative inhibitory strength: w_I = -g * J.
    eta : float
        External drive relative to threshold rate: nu_ext = eta * nu_thr.
    j : float
        Excitatory synaptic weight (mV).
    tau_m : float
        Membrane time constant (ms).
    t_ref : float
        Refractory period (ms).
    v_rest : float
        Resting potential (mV). Brunel uses 0.
    v_thresh : float
        Threshold (mV). Brunel uses 20.
    v_reset : float
        Reset potential (mV). Brunel uses 10.
    delay_ms : float
        Synaptic delay (ms).
    dt : float
        Simulation timestep (ms).
    seed : int
        Random seed for network generation.

    Returns
    -------
    circuit : Circuit
        The random network.
    params : dict
        Network parameters including nu_thr, nu_ext, C_E, C_I.
    """
    rng = np.random.RandomState(seed)

    n_inhibitory = int(n_excitatory * gamma)
    n_total = n_excitatory + n_inhibitory
    c_e = int(epsilon * n_excitatory)  # excitatory connections per neuron
    c_i = int(epsilon * n_inhibitory)  # inhibitory connections per neuron

    # Threshold rate: external rate that alone brings neuron to threshold
    # For our engine with exponential synaptic filter (tau_syn), the
    # steady-state voltage from Poisson input at rate nu is:
    #   V_ss = v_rest + J * C_E * nu * tau_syn
    # (because g_ss = J * C_E * nu * tau_syn and V_ss = v_rest + g_ss)
    # Setting V_ss = V_thresh:
    #   nu_thr = (V_thresh - V_rest) / (J * C_E * tau_syn)
    # Our engine uses exponential synaptic filtering: spikes add J to g,
    # g decays with tau_syn, then dV/dt = (-(V-V_rest) + g) / tau_m.
    # For Brunel's delta synapses (spike adds J directly to V), the
    # effective PSP area is J * tau_m. In our engine, the PSP area is
    # J * tau_syn * tau_m / (tau_m - tau_syn) ≈ J * tau_syn (if tau_syn << tau_m).
    #
    # We use tau_syn=0.5ms (fast, preserves spike timing fluctuations)
    # and scale J_eff = J * tau_m / tau_syn to match Brunel's effective PSP.
    tau_syn = 0.5
    j_eff = j * tau_m / tau_syn  # scale weights to compensate for fast tau_syn
    # Threshold rate: nu_thr = (V_thresh - V_rest) / (J_eff * C_E * tau_syn)
    #                        = (V_thresh - V_rest) / (J * tau_m / tau_syn * C_E * tau_syn)
    #                        = (V_thresh - V_rest) / (J * C_E * tau_m)
    nu_thr = (v_thresh - v_rest) / (j * c_e * tau_m)

    # External input rate
    nu_ext = eta * nu_thr

    LOG.info("Brunel network: N_E=%d, N_I=%d, C_E=%d, C_I=%d, g=%.1f, eta=%.1f",
             n_excitatory, n_inhibitory, c_e, c_i, g, eta)
    LOG.info("  nu_thr=%.4f kHz, nu_ext=%.4f kHz", nu_thr, nu_ext)

    # Build random connectivity
    pre_list = []
    post_list = []
    weight_list = []

    for i in range(n_total):
        # Excitatory inputs
        if c_e > 0:
            exc_sources = rng.choice(n_excitatory, size=c_e, replace=False)
            pre_list.append(exc_sources)
            post_list.append(np.full(c_e, i, dtype=np.int32))
            weight_list.append(np.full(c_e, j_eff))

        # Inhibitory inputs
        if c_i > 0:
            inh_sources = rng.choice(
                np.arange(n_excitatory, n_total), size=c_i, replace=False
            )
            pre_list.append(inh_sources)
            post_list.append(np.full(c_i, i, dtype=np.int32))
            weight_list.append(np.full(c_i, -g * j_eff))

    pre_idx = np.concatenate(pre_list).astype(np.int32)
    post_idx = np.concatenate(post_list).astype(np.int32)
    weights = np.concatenate(weight_list).astype(np.float64)

    delay_steps = max(1, int(round(delay_ms / dt)))

    circuit = Circuit(
        n_neurons=n_total,
        v_rest=np.full(n_total, v_rest),
        v_thresh=np.full(n_total, v_thresh),
        v_reset=np.full(n_total, v_reset),
        tau_m=np.full(n_total, tau_m),
        t_ref=np.full(n_total, t_ref),
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        tau_syn=tau_syn,
        delay_steps=delay_steps,
    )

    params = {
        "n_excitatory": n_excitatory,
        "n_inhibitory": n_inhibitory,
        "n_total": n_total,
        "gamma": gamma,
        "epsilon": epsilon,
        "c_e": c_e,
        "c_i": c_i,
        "g": g,
        "eta": eta,
        "j": j,
        "tau_m": tau_m,
        "t_ref": t_ref,
        "v_rest": v_rest,
        "v_thresh": v_thresh,
        "v_reset": v_reset,
        "delay_ms": delay_ms,
        "tau_syn": tau_syn,
        "j_eff": j_eff,
        "nu_thr": nu_thr,
        "nu_ext": nu_ext,
    }

    return circuit, params


def build_brunel_stimulus(circuit, params, duration_ms=1000.0, dt=0.1, seed=42):
    """Build external Poisson drive for a Brunel network.

    Each neuron receives C_E independent Poisson inputs at rate nu_ext.
    The expected number of spikes per dt is: C_E * nu_ext * dt.
    Each spike contributes J mV. We draw the spike count from a Poisson
    distribution and add count * J to the stimulus.

    Parameters
    ----------
    circuit : Circuit
        Brunel network.
    params : dict
        Network parameters from build_brunel_network.
    duration_ms : float
        Stimulus duration.
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    stim : np.ndarray
        Shape (n_neurons, n_steps).
    """
    n = circuit.n_neurons
    n_steps = int(duration_ms / dt)
    nu_ext = params["nu_ext"]  # in kHz (spikes per ms)
    c_e = params["c_e"]
    j_eff = params["j_eff"]  # scaled weight matching engine's tau_syn

    # Expected number of external spikes per neuron per dt
    lam = c_e * nu_ext * dt  # Poisson lambda

    rng = np.random.RandomState(seed)
    stim = np.zeros((n, n_steps), dtype=np.float64)

    for step in range(n_steps):
        # Draw number of Poisson spikes arriving this timestep
        n_spikes = rng.poisson(lam, n)
        stim[:, step] = n_spikes * j_eff

    return stim


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(result, excitatory_indices=None, dt=0.1,
                    analysis_window=None):
    """Classify a simulation result into Brunel's four regimes.

    Uses two metrics:
    1. CV of ISI (coefficient of variation of interspike intervals)
       - CV < 0.5: regular firing
       - CV > 0.8: irregular firing
    2. Synchrony index (normalized variance of population rate)
       - Low: asynchronous
       - High: synchronous

    Parameters
    ----------
    result : SimulationResult
        Simulation output.
    excitatory_indices : array-like, optional
        Indices of excitatory neurons. If None, uses first 80%.
    dt : float
        Timestep (ms).
    analysis_window : tuple, optional
        (start_ms, end_ms) for analysis. Default: last 80% of simulation.

    Returns
    -------
    dict
        regime : str ("SR", "SI", "AR", "AI")
        cv_isi : float (mean CV of ISI)
        synchrony : float (synchrony index)
        mean_rate : float (Hz)
        fano_factor : float
        details : dict
    """
    if analysis_window is None:
        t_start = result.duration * 0.2  # skip transient
        t_end = result.duration
    else:
        t_start, t_end = analysis_window

    if excitatory_indices is None:
        n_exc = int(result.n_neurons * 0.8)
        excitatory_indices = np.arange(n_exc)

    # 1. Compute CV of ISI for active neurons
    cv_values = []
    for i in excitatory_indices:
        st = result.spike_times[i]
        st = st[(st >= t_start) & (st < t_end)]
        if len(st) >= 3:
            isi = np.diff(st)
            if np.mean(isi) > 0:
                cv_values.append(np.std(isi) / np.mean(isi))

    mean_cv = np.mean(cv_values) if cv_values else 0.0

    # 2. Compute synchrony index from population rate
    bin_ms = 1.0  # 1 ms bins for synchrony detection
    times, rates = population_rate(result, bin_ms=bin_ms)
    # Restrict to analysis window
    mask = (times >= t_start) & (times < t_end)
    rates_window = rates[mask]

    if len(rates_window) > 0 and np.mean(rates_window) > 0:
        synchrony = np.var(rates_window) / np.mean(rates_window)
    else:
        synchrony = 0.0

    # 3. Mean firing rate
    rates_full = firing_rates(result, time_window=(t_start, t_end))
    mean_rate = float(np.mean(rates_full[excitatory_indices]))

    # 4. Fano factor (spike count variance / mean in bins)
    bin_ms_fano = 50.0
    n_bins = max(1, int((t_end - t_start) / bin_ms_fano))
    spike_counts = np.zeros(n_bins)
    for i in excitatory_indices[:min(1000, len(excitatory_indices))]:
        st = result.spike_times[i]
        st = st[(st >= t_start) & (st < t_end)]
        for t in st:
            b = min(int((t - t_start) / bin_ms_fano), n_bins - 1)
            spike_counts[b] += 1
    fano = np.var(spike_counts) / np.mean(spike_counts) if np.mean(spike_counts) > 0 else 0.0

    # Classify using continuous metrics
    # CV > cv_threshold: irregular firing
    # synchrony > sync_threshold: population oscillations
    cv_threshold = 0.5
    sync_threshold = 10.0

    is_irregular = mean_cv > cv_threshold
    is_synchronous = synchrony > sync_threshold

    if mean_rate < 0.5:
        regime = "quiescent"
    elif is_synchronous and is_irregular:
        regime = "SI"
    elif is_synchronous and not is_irregular:
        regime = "SR"
    elif not is_synchronous and is_irregular:
        regime = "AI"
    else:
        regime = "AR"

    LOG.info("Regime: %s (CV=%.2f, sync=%.1f, rate=%.1f Hz)",
             regime, mean_cv, synchrony, mean_rate)

    return {
        "regime": regime,
        "cv_isi": mean_cv,
        "synchrony": synchrony,
        "mean_rate": mean_rate,
        "fano_factor": fano,
        "n_active": len(cv_values),
        "details": {
            "is_irregular": is_irregular,
            "is_synchronous": is_synchronous,
            "cv_threshold": cv_threshold,
            "sync_threshold": sync_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Phase diagram sweep
# ---------------------------------------------------------------------------

def brunel_phase_sweep(g_values=None, eta_values=None,
                       n_excitatory=2000, duration_ms=500.0,
                       dt=0.1, seed=42, **kwargs):
    """Sweep the (g, eta) phase diagram.

    Parameters
    ----------
    g_values : list of float
        Inhibitory strength ratios to test.
    eta_values : list of float
        External drive ratios to test.
    n_excitatory : int
        Network size (smaller for speed).
    duration_ms : float
        Simulation duration per point.
    dt : float
        Timestep.
    seed : int
        Random seed.
    **kwargs
        Passed to build_brunel_network.

    Returns
    -------
    pd.DataFrame
        One row per (g, eta) point with regime classification.
    """
    if g_values is None:
        g_values = [3.0, 4.0, 4.5, 5.0, 6.0]
    if eta_values is None:
        eta_values = [0.9, 1.5, 2.0, 3.0, 4.0]

    results = []
    total = len(g_values) * len(eta_values)
    count = 0

    for g in g_values:
        for eta in eta_values:
            count += 1
            LOG.info("Phase sweep %d/%d: g=%.1f, eta=%.1f", count, total, g, eta)

            circuit, params = build_brunel_network(
                n_excitatory=n_excitatory, g=g, eta=eta,
                dt=dt, seed=seed, **kwargs
            )

            stim = build_brunel_stimulus(
                circuit, params, duration_ms=duration_ms, dt=dt,
                seed=seed + count,
            )

            result = simulate(
                circuit, duration=duration_ms, dt=dt,
                stimulus=stim, seed=seed + count,
            )

            classification = classify_regime(result, dt=dt)

            results.append({
                "g": g,
                "eta": eta,
                "regime": classification["regime"],
                "cv_isi": classification["cv_isi"],
                "synchrony": classification["synchrony"],
                "mean_rate": classification["mean_rate"],
                "fano_factor": classification["fano_factor"],
                "n_active": classification["n_active"],
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# FlyWire regime classification
# ---------------------------------------------------------------------------

def classify_flywire_regime(circuit, mb_neurons=None, duration_ms=500.0,
                            pn_rate_hz=30.0, pn_weight=68.75,
                            odor_fraction=0.1, seed=42):
    """Classify the FlyWire MB circuit's dynamical regime.

    Drives the circuit with Poisson PN input and classifies using the
    same metrics as the Brunel phase diagram.

    Parameters
    ----------
    circuit : Circuit
        FlyWire MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame, optional
        MB neuron annotations. If provided, uses PN indices for stimulus.
    duration_ms : float
        Simulation duration.
    pn_rate_hz : float
        PN firing rate.
    pn_weight : float
        PN spike weight.
    odor_fraction : float
        Fraction of PNs to drive.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Regime classification + comparison to Brunel diagram.
    """
    from bravli.explore.mushroom_body import neuron_groups
    from bravli.explore.isn_experiment import identify_ei_groups

    n_steps = int(duration_ms / 0.1)
    rng = np.random.RandomState(seed)

    # Build stimulus
    if mb_neurons is not None:
        groups = neuron_groups(circuit, mb_neurons)
        pn_indices = groups.get("PN", np.array([], dtype=np.int32))
        ei_groups = identify_ei_groups(circuit, mb_neurons)
        exc_indices = ei_groups["E"]
    else:
        # Assume first 80% excitatory
        pn_indices = np.arange(int(circuit.n_neurons * 0.1))
        exc_indices = np.arange(int(circuit.n_neurons * 0.8))

    n_active = max(1, int(odor_fraction * len(pn_indices)))
    active_pns = rng.choice(pn_indices, size=n_active, replace=False)

    stim, _ = poisson_stimulus(
        circuit.n_neurons, n_steps, active_pns,
        rate_hz=pn_rate_hz, weight=pn_weight, seed=seed,
    )

    # Simulate
    result = simulate(
        circuit, duration=duration_ms, dt=0.1,
        stimulus=stim, seed=seed,
    )

    # Classify
    classification = classify_regime(result, excitatory_indices=exc_indices)

    # Compute effective g and eta for comparison
    if circuit.n_synapses > 0:
        exc_weights = circuit.weights[circuit.weights > 0]
        inh_weights = circuit.weights[circuit.weights < 0]
        mean_exc = np.mean(exc_weights) if len(exc_weights) > 0 else 1.0
        mean_inh = np.mean(np.abs(inh_weights)) if len(inh_weights) > 0 else 0.0
        effective_g = mean_inh / mean_exc if mean_exc > 0 else 0.0
    else:
        effective_g = 0.0

    classification["effective_g"] = effective_g
    classification["circuit_type"] = "flywire_mb"
    classification["n_neurons"] = circuit.n_neurons
    classification["n_synapses"] = circuit.n_synapses

    LOG.info("FlyWire regime: %s (effective_g=%.2f, rate=%.1f Hz)",
             classification["regime"], effective_g, classification["mean_rate"])

    return classification


def brunel_report(phase_df, flywire_result=None):
    """Print a structured report for the Brunel phase diagram analysis.

    Parameters
    ----------
    phase_df : pd.DataFrame
        Output of brunel_phase_sweep.
    flywire_result : dict, optional
        Output of classify_flywire_regime.

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "=" * 60,
        "BRUNEL PHASE DIAGRAM — Network Regime Analysis",
        "=" * 60,
        "",
        "--- Phase diagram ---",
        "",
    ]

    # Pivot table: g x eta -> regime
    if len(phase_df) > 0:
        pivot = phase_df.pivot(index="g", columns="eta", values="regime")
        lines.append(pivot.to_string())
        lines.append("")

        # Regime counts
        counts = phase_df["regime"].value_counts()
        lines.append("Regime distribution:")
        for regime, n in counts.items():
            lines.append(f"  {regime}: {n}")
        lines.append("")

        # Summary statistics per regime
        lines.append("--- Per-regime statistics ---")
        for regime in ["AI", "SI", "SR", "AR"]:
            subset = phase_df[phase_df["regime"] == regime]
            if len(subset) > 0:
                lines.append(f"  {regime}: mean_rate={subset['mean_rate'].mean():.1f} Hz, "
                             f"CV={subset['cv_isi'].mean():.2f}, "
                             f"sync={subset['synchrony'].mean():.1f}")

    # FlyWire result
    if flywire_result is not None:
        lines.append("")
        lines.append("--- FlyWire MB circuit ---")
        lines.append(f"  Regime:       {flywire_result['regime']}")
        lines.append(f"  CV ISI:       {flywire_result['cv_isi']:.2f}")
        lines.append(f"  Synchrony:    {flywire_result['synchrony']:.1f}")
        lines.append(f"  Mean rate:    {flywire_result['mean_rate']:.1f} Hz")
        lines.append(f"  Effective g:  {flywire_result['effective_g']:.2f}")
        lines.append(f"  Neurons:      {flywire_result['n_neurons']:,}")
        lines.append(f"  Synapses:     {flywire_result['n_synapses']:,}")

    # Interpretation
    lines.append("")
    lines.append("--- Interpretation ---")
    if flywire_result is not None:
        regime = flywire_result["regime"]
        if regime == "AI":
            lines.append("  The FlyWire MB operates in the ASYNCHRONOUS IRREGULAR regime.")
            lines.append("  This is the balanced E/I state (van Vreeswijk & Sompolinsky 1996).")
            lines.append("  Individual neurons fire irregularly at low rates, while the")
            lines.append("  population rate is stable. This is the most common regime in cortex.")
        elif regime == "SI":
            lines.append("  The FlyWire MB operates in the SYNCHRONOUS IRREGULAR regime.")
            lines.append("  Population oscillations coexist with irregular individual firing.")
            lines.append("  This suggests the circuit is near the boundary of balanced dynamics.")
        elif regime == "SR":
            lines.append("  The FlyWire MB operates in the SYNCHRONOUS REGULAR regime.")
            lines.append("  Neurons fire nearly in lockstep. This is unusual for cortical")
            lines.append("  circuits and suggests excitation dominates inhibition.")
        elif regime == "AR":
            lines.append("  The FlyWire MB operates in the ASYNCHRONOUS REGULAR regime.")
            lines.append("  Individual neurons fire regularly but without population synchrony.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
