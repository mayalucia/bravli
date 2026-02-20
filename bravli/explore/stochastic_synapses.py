"""Stochastic synapses: noise, failure, and stochastic resonance.

Central synapses fail 50-90% of the time (Allen & Stevens 1994). This is not
sloppiness — it is actively maintained. This module investigates how synaptic
unreliability and intrinsic noise affect network dynamics, and tests whether
stochastic resonance enhances weak signal detection.

Three noise mechanisms:
1. Synaptic release failure: per-synapse Bernoulli trial (release_prob < 1)
2. Intrinsic noise current: Gaussian white noise added to membrane potential
3. Combined: both mechanisms together

References:
    Allen C, Stevens CF (1994). PNAS 91(22):10380-10383.
    Faisal AA, Selen LP, Wolpert DM (2008). Nature Rev Neurosci 9:292-303.
    Longtin A (1993). J Stat Phys 70(1):309-327.
    Gammaitoni L et al. (1998). Rev Mod Phys 70(1):223-287.
"""

import numpy as np
import pandas as pd

from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import poisson_stimulus, step_stimulus
from bravli.simulation.analysis import (
    firing_rates, population_rate, active_fraction_by_group,
)
from bravli.explore.mushroom_body import neuron_groups

from bravli.utils import get_logger

LOG = get_logger("explore.stochastic")


# ---------------------------------------------------------------------------
# Noise sweep experiments
# ---------------------------------------------------------------------------

def noise_sweep(circuit, stimulus=None, noise_sigmas=None,
                duration_ms=500.0, dt=0.1, seed=42):
    """Measure network dynamics across a range of intrinsic noise levels.

    Parameters
    ----------
    circuit : Circuit
        Neural circuit to simulate.
    stimulus : np.ndarray, optional
        External stimulus. If None, no external input.
    noise_sigmas : list of float, optional
        Noise levels to test. Default: [0, 0.5, 1, 2, 5, 10, 20].
    duration_ms : float
        Simulation duration.
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: noise_sigma, mean_rate, n_spikes, active_fraction,
        rate_cv (CV of per-neuron rates).
    """
    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    rows = []
    for sigma in noise_sigmas:
        LOG.info("Noise sweep: sigma=%.1f", sigma)
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stimulus, noise_sigma=sigma, seed=seed,
        )
        rates = firing_rates(result)
        active = np.mean(rates > 1.0)
        rate_cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0.0

        rows.append({
            "noise_sigma": sigma,
            "mean_rate": float(np.mean(rates)),
            "n_spikes": result.n_spikes,
            "active_fraction": float(active),
            "rate_cv": float(rate_cv),
        })

    return pd.DataFrame(rows)


def release_prob_sweep(circuit, stimulus=None, release_probs=None,
                       duration_ms=500.0, dt=0.1, seed=42):
    """Measure network dynamics across a range of release probabilities.

    Parameters
    ----------
    circuit : Circuit
        Neural circuit.
    stimulus : np.ndarray, optional
        External stimulus.
    release_probs : list of float, optional
        Release probabilities to test. Default: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0].
    duration_ms : float
        Simulation duration.
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: release_prob, mean_rate, n_spikes, active_fraction, rate_cv.
    """
    if release_probs is None:
        release_probs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    rows = []
    for p in release_probs:
        LOG.info("Release prob sweep: p=%.2f", p)
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stimulus, release_prob=p, seed=seed,
        )
        rates = firing_rates(result)
        active = np.mean(rates > 1.0)
        rate_cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0.0

        rows.append({
            "release_prob": p,
            "mean_rate": float(np.mean(rates)),
            "n_spikes": result.n_spikes,
            "active_fraction": float(active),
            "rate_cv": float(rate_cv),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stochastic resonance test
# ---------------------------------------------------------------------------

def stochastic_resonance_test(circuit, signal_indices, signal_amplitude,
                              noise_sigmas=None, signal_frequency_hz=5.0,
                              duration_ms=2000.0, dt=0.1, seed=42):
    """Test for stochastic resonance: does optimal noise enhance signal detection?

    Protocol:
    1. Create a weak subthreshold periodic signal (sinusoidal current injection
       to a subset of neurons)
    2. For each noise level, simulate and measure how well the output tracks
       the input signal (correlation between stimulus periodicity and spike
       periodicity)
    3. If stochastic resonance is present, there exists an optimal noise level
       where signal detection peaks

    The signal-to-noise ratio (SNR) is measured via the power spectral density
    of the population rate at the signal frequency.

    Parameters
    ----------
    circuit : Circuit
        Neural circuit.
    signal_indices : array-like
        Indices of neurons receiving the periodic signal.
    signal_amplitude : float
        Amplitude of the sinusoidal signal (mV). Should be subthreshold.
    noise_sigmas : list of float, optional
        Noise levels to test.
    signal_frequency_hz : float
        Frequency of the periodic signal.
    duration_ms : float
        Simulation duration (needs to be long enough for frequency analysis).
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: noise_sigma, snr, mean_rate, signal_power, noise_power.
    dict
        Additional info: signal_frequency_hz, signal_indices, best_sigma.
    """
    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    signal_indices = np.asarray(signal_indices)
    n_steps = int(duration_ms / dt)
    signal_freq_khz = signal_frequency_hz / 1000.0  # convert to per-ms

    # Build periodic stimulus
    t_array = np.arange(n_steps) * dt
    signal_wave = signal_amplitude * np.sin(2 * np.pi * signal_freq_khz * t_array)

    stim = np.zeros((circuit.n_neurons, n_steps), dtype=np.float64)
    for idx in signal_indices:
        stim[idx, :] = signal_wave

    rows = []
    for sigma in noise_sigmas:
        LOG.info("Stochastic resonance: sigma=%.1f", sigma)
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stim, noise_sigma=sigma, seed=seed,
        )

        # Compute population rate in fine bins
        bin_ms = 1.0
        times, pop_rate = population_rate(result, bin_ms=bin_ms)

        if len(pop_rate) < 2:
            rows.append({
                "noise_sigma": sigma,
                "snr": 0.0,
                "mean_rate": 0.0,
                "signal_power": 0.0,
                "noise_power": 0.0,
            })
            continue

        # FFT of population rate
        fft_vals = np.fft.rfft(pop_rate - np.mean(pop_rate))
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(pop_rate), d=bin_ms / 1000.0)  # Hz

        # Find signal power at the target frequency
        freq_idx = np.argmin(np.abs(freqs - signal_frequency_hz))
        signal_power = float(power[freq_idx])

        # Noise power: mean of surrounding frequencies (exclude DC and signal)
        noise_mask = np.ones(len(power), dtype=bool)
        noise_mask[0] = False  # exclude DC
        # Exclude signal frequency and its immediate neighbors
        for offset in range(-2, 3):
            idx = freq_idx + offset
            if 0 <= idx < len(noise_mask):
                noise_mask[idx] = False
        noise_power = float(np.mean(power[noise_mask])) if np.any(noise_mask) else 1e-10

        snr = signal_power / noise_power if noise_power > 0 else 0.0

        rates = firing_rates(result)
        rows.append({
            "noise_sigma": sigma,
            "snr": float(snr),
            "mean_rate": float(np.mean(rates)),
            "signal_power": signal_power,
            "noise_power": noise_power,
        })

    df = pd.DataFrame(rows)

    # Find best noise level
    if len(df) > 0 and df["snr"].max() > 0:
        best_idx = df["snr"].idxmax()
        best_sigma = float(df.loc[best_idx, "noise_sigma"])
    else:
        best_sigma = 0.0

    info = {
        "signal_frequency_hz": signal_frequency_hz,
        "signal_amplitude": signal_amplitude,
        "signal_indices": signal_indices,
        "best_sigma": best_sigma,
        "has_resonance": best_sigma > 0.0,
    }

    return df, info


# ---------------------------------------------------------------------------
# MB-specific stochastic experiment
# ---------------------------------------------------------------------------

def mb_stochastic_experiment(circuit, mb_neurons,
                             noise_sigmas=None, release_probs=None,
                             pn_rate_hz=50.0, pn_weight=68.75,
                             odor_fraction=0.1,
                             duration_ms=500.0, seed=42):
    """Run stochastic synapse experiments on the MB circuit.

    Tests both noise and release probability effects on odor coding:
    - How does noise affect KC sparseness?
    - How does release failure affect MBON response fidelity?
    - Is there an optimal noise level for odor discrimination?

    Parameters
    ----------
    circuit : Circuit
        MB circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    noise_sigmas : list of float, optional
        Noise levels to test.
    release_probs : list of float, optional
        Release probabilities to test.
    pn_rate_hz : float
        PN firing rate for odor.
    pn_weight : float
        PN spike weight.
    odor_fraction : float
        Fraction of PNs driven.
    duration_ms : float
        Simulation duration.
    seed : int
        Random seed.

    Returns
    -------
    dict
        noise_sweep : pd.DataFrame
        release_sweep : pd.DataFrame
        per_group : dict of noise_sigma -> group activity fractions
    """
    if noise_sigmas is None:
        noise_sigmas = [0.0, 1.0, 3.0, 5.0, 10.0]
    if release_probs is None:
        release_probs = [0.1, 0.3, 0.5, 0.7, 1.0]

    dt = 0.1
    rng = np.random.RandomState(seed)
    groups = neuron_groups(circuit, mb_neurons)
    pn_indices = groups.get("PN", np.array([], dtype=np.int32))

    if len(pn_indices) == 0:
        return {"error": "No PN neurons"}

    # Build odor stimulus
    n_active = max(1, int(odor_fraction * len(pn_indices)))
    active_pns = rng.choice(pn_indices, size=n_active, replace=False)
    n_steps = int(duration_ms / dt)
    stim, _ = poisson_stimulus(
        circuit.n_neurons, n_steps, active_pns,
        rate_hz=pn_rate_hz, weight=pn_weight, seed=seed,
    )

    # Noise sweep
    noise_rows = []
    per_group_noise = {}
    for sigma in noise_sigmas:
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stim, noise_sigma=sigma, seed=seed,
        )
        rates = firing_rates(result)
        active = active_fraction_by_group(result, groups)
        per_group_noise[sigma] = active

        kc_rates = rates[groups.get("KC", [])] if "KC" in groups else np.array([])
        noise_rows.append({
            "noise_sigma": sigma,
            "mean_rate": float(np.mean(rates)),
            "kc_mean_rate": float(np.mean(kc_rates)) if len(kc_rates) > 0 else 0.0,
            "kc_active_frac": active.get("KC", (0, 0, 0.0))[2],
            "mbon_active_frac": active.get("MBON", (0, 0, 0.0))[2],
        })

    # Release probability sweep
    release_rows = []
    for p in release_probs:
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stim, release_prob=p, seed=seed,
        )
        rates = firing_rates(result)
        active = active_fraction_by_group(result, groups)

        kc_rates = rates[groups.get("KC", [])] if "KC" in groups else np.array([])
        release_rows.append({
            "release_prob": p,
            "mean_rate": float(np.mean(rates)),
            "kc_mean_rate": float(np.mean(kc_rates)) if len(kc_rates) > 0 else 0.0,
            "kc_active_frac": active.get("KC", (0, 0, 0.0))[2],
            "mbon_active_frac": active.get("MBON", (0, 0, 0.0))[2],
        })

    return {
        "noise_sweep": pd.DataFrame(noise_rows),
        "release_sweep": pd.DataFrame(release_rows),
        "per_group_noise": per_group_noise,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def stochastic_report(noise_df=None, release_df=None,
                      sr_df=None, sr_info=None,
                      mb_results=None):
    """Print a structured report on stochastic synapse experiments.

    Parameters
    ----------
    noise_df : pd.DataFrame, optional
        From noise_sweep().
    release_df : pd.DataFrame, optional
        From release_prob_sweep().
    sr_df : pd.DataFrame, optional
        From stochastic_resonance_test().
    sr_info : dict, optional
        Additional info from stochastic_resonance_test().
    mb_results : dict, optional
        From mb_stochastic_experiment().

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "=" * 60,
        "STOCHASTIC SYNAPSES — Noise, Failure, and Resonance",
        "=" * 60,
        "",
    ]

    if noise_df is not None and len(noise_df) > 0:
        lines.append("--- Intrinsic noise sweep ---")
        lines.append(f"{'sigma':>8s} {'rate (Hz)':>10s} {'active':>8s} {'CV':>8s}")
        lines.append("-" * 36)
        for _, row in noise_df.iterrows():
            lines.append(
                f"{row['noise_sigma']:8.1f} {row['mean_rate']:10.1f} "
                f"{row['active_fraction']:8.2f} {row['rate_cv']:8.2f}"
            )
        lines.append("")

    if release_df is not None and len(release_df) > 0:
        lines.append("--- Release probability sweep ---")
        lines.append(f"{'p_rel':>8s} {'rate (Hz)':>10s} {'active':>8s} {'CV':>8s}")
        lines.append("-" * 36)
        for _, row in release_df.iterrows():
            lines.append(
                f"{row['release_prob']:8.2f} {row['mean_rate']:10.1f} "
                f"{row['active_fraction']:8.2f} {row['rate_cv']:8.2f}"
            )
        lines.append("")

    if sr_df is not None and len(sr_df) > 0:
        lines.append("--- Stochastic resonance test ---")
        if sr_info:
            lines.append(f"  Signal: {sr_info['signal_frequency_hz']:.1f} Hz, "
                         f"amplitude {sr_info['signal_amplitude']:.1f} mV")
        lines.append(f"{'sigma':>8s} {'SNR':>10s} {'rate (Hz)':>10s}")
        lines.append("-" * 30)
        for _, row in sr_df.iterrows():
            lines.append(
                f"{row['noise_sigma']:8.1f} {row['snr']:10.1f} "
                f"{row['mean_rate']:10.1f}"
            )
        if sr_info and sr_info.get("has_resonance"):
            lines.append(f"\n  STOCHASTIC RESONANCE DETECTED: optimal sigma = "
                         f"{sr_info['best_sigma']:.1f}")
        else:
            lines.append("\n  No clear stochastic resonance peak.")
        lines.append("")

    if mb_results and "noise_sweep" in mb_results:
        lines.append("--- MB circuit: noise effects on odor coding ---")
        mb_noise = mb_results["noise_sweep"]
        lines.append(f"{'sigma':>8s} {'KC rate':>10s} {'KC active':>10s} {'MBON active':>12s}")
        lines.append("-" * 42)
        for _, row in mb_noise.iterrows():
            lines.append(
                f"{row['noise_sigma']:8.1f} {row['kc_mean_rate']:10.1f} "
                f"{row['kc_active_frac']:10.2f} {row['mbon_active_frac']:12.2f}"
            )
        lines.append("")

    if mb_results and "release_sweep" in mb_results:
        lines.append("--- MB circuit: release failure effects ---")
        mb_rel = mb_results["release_sweep"]
        lines.append(f"{'p_rel':>8s} {'KC rate':>10s} {'KC active':>10s} {'MBON active':>12s}")
        lines.append("-" * 42)
        for _, row in mb_rel.iterrows():
            lines.append(
                f"{row['release_prob']:8.2f} {row['kc_mean_rate']:10.1f} "
                f"{row['kc_active_frac']:10.2f} {row['mbon_active_frac']:12.2f}"
            )
        lines.append("")

    # Interpretation
    lines.append("--- Interpretation ---")
    if noise_df is not None and len(noise_df) > 1:
        zero_rate = noise_df.loc[noise_df["noise_sigma"] == 0.0, "mean_rate"]
        max_sigma_rate = noise_df.iloc[-1]["mean_rate"]
        if len(zero_rate) > 0 and max_sigma_rate > zero_rate.values[0]:
            lines.append("  Noise INCREASES firing rate — noise-driven activity.")
        elif len(zero_rate) > 0:
            lines.append("  Noise has minimal effect on mean rate — robust dynamics.")

    if release_df is not None and len(release_df) > 1:
        p1_rate = release_df.loc[release_df["release_prob"] == 1.0, "mean_rate"]
        plow_rate = release_df.iloc[0]["mean_rate"]
        if len(p1_rate) > 0:
            ratio = plow_rate / p1_rate.values[0] if p1_rate.values[0] > 0 else 0
            if ratio < 0.5:
                lines.append("  Synaptic failure STRONGLY reduces activity — "
                             "circuit is failure-sensitive.")
            else:
                lines.append("  Circuit is ROBUST to synaptic failure — "
                             "activity degrades gracefully.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
