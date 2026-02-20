"""LIF vs AdEx comparison on the MB circuit.

Tests the Zhang et al. (2024) hypothesis: does network topology dominate
single-neuron model details? We simulate the same MB circuit with both
LIF and AdEx models, using the same stimulus, and compare:

1. Mean firing rates
2. Active neuron fractions (sparseness)
3. Per-group rate correlations
4. Spike timing precision
5. MBON response patterns

If topology dominates, both models should produce qualitatively similar
results. Divergence indicates that single-neuron dynamics (adaptation,
exponential spike initiation) matter for the specific computation.

References:
    Zhang Z et al. (2024). iScience 27(5):109863.
    Brette R, Gerstner W (2005). J Comp Neurosci 19(2):175-197.
"""

import numpy as np
import pandas as pd

from bravli.simulation.engine import simulate
from bravli.simulation.adex_engine import (
    simulate_adex, AdExParams, ADEX_PRESETS,
)
from bravli.simulation.stimulus import poisson_stimulus
from bravli.simulation.analysis import (
    firing_rates, active_fraction, active_fraction_by_group,
    population_rate, population_sparseness,
)
from bravli.explore.mushroom_body import neuron_groups

from bravli.utils import get_logger

LOG = get_logger("explore.lif_vs_adex")


def compare_models(circuit, mb_neurons=None,
                   adex_params=None,
                   stimulus=None,
                   pn_rate_hz=50.0, pn_weight=68.75,
                   odor_fraction=0.1,
                   duration_ms=500.0, dt=0.1, seed=42):
    """Compare LIF and AdEx on the same circuit with the same stimulus.

    Parameters
    ----------
    circuit : Circuit
        Neural circuit.
    mb_neurons : pd.DataFrame, optional
        MB neuron annotations (for group-level analysis).
    adex_params : AdExParams, optional
        AdEx parameters. Default: regular_spiking preset.
    stimulus : np.ndarray, optional
        Shared stimulus. If None, builds Poisson PN input.
    pn_rate_hz : float
        PN rate for auto-generated stimulus.
    pn_weight : float
        PN weight.
    odor_fraction : float
        Fraction of PNs to drive.
    duration_ms : float
        Simulation duration.
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    dict
        lif : dict with rates, sparseness, result
        adex : dict with rates, sparseness, result
        comparison : dict with correlation, divergence metrics
    """
    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / dt)

    # Build stimulus if not provided
    if stimulus is None and mb_neurons is not None:
        groups = neuron_groups(circuit, mb_neurons)
        pn_indices = groups.get("PN", np.array([], dtype=np.int32))
        if len(pn_indices) > 0:
            n_active = max(1, int(odor_fraction * len(pn_indices)))
            active_pns = rng.choice(pn_indices, size=n_active, replace=False)
            stimulus, _ = poisson_stimulus(
                circuit.n_neurons, n_steps, active_pns,
                rate_hz=pn_rate_hz, weight=pn_weight, seed=seed,
            )

    # Run LIF
    LOG.info("Running LIF simulation...")
    lif_result = simulate(
        circuit, duration=duration_ms, dt=dt,
        stimulus=stimulus, seed=seed,
    )

    # Run AdEx
    LOG.info("Running AdEx simulation...")
    adex_result = simulate_adex(
        circuit, adex_params=adex_params,
        duration=duration_ms, dt=dt,
        stimulus=stimulus, seed=seed,
    )

    # Compute rates
    lif_rates = firing_rates(lif_result)
    adex_rates = firing_rates(adex_result)

    # Sparseness
    lif_sparseness = population_sparseness(lif_rates)
    adex_sparseness = population_sparseness(adex_rates)

    # Active fractions
    lif_active = active_fraction(lif_result)
    adex_active = active_fraction(adex_result)

    # Per-group analysis
    groups = {}
    lif_group_rates = {}
    adex_group_rates = {}
    if mb_neurons is not None:
        groups = neuron_groups(circuit, mb_neurons)
        lif_group_active = active_fraction_by_group(lif_result, groups)
        adex_group_active = active_fraction_by_group(adex_result, groups)
        for name, indices in groups.items():
            if len(indices) > 0:
                lif_group_rates[name] = float(np.mean(lif_rates[indices]))
                adex_group_rates[name] = float(np.mean(adex_rates[indices]))
    else:
        lif_group_active = {}
        adex_group_active = {}

    # Rate correlation (across neurons)
    if np.std(lif_rates) > 0 and np.std(adex_rates) > 0:
        rate_correlation = float(np.corrcoef(lif_rates, adex_rates)[0, 1])
    else:
        rate_correlation = 0.0 if (np.std(lif_rates) + np.std(adex_rates)) > 0 else 1.0

    # Population rate correlation (across time)
    bin_ms = 5.0
    lif_times, lif_pop = population_rate(lif_result, bin_ms=bin_ms)
    adex_times, adex_pop = population_rate(adex_result, bin_ms=bin_ms)
    n_bins = min(len(lif_pop), len(adex_pop))
    if n_bins > 1 and np.std(lif_pop[:n_bins]) > 0 and np.std(adex_pop[:n_bins]) > 0:
        temporal_correlation = float(
            np.corrcoef(lif_pop[:n_bins], adex_pop[:n_bins])[0, 1]
        )
    else:
        temporal_correlation = 0.0

    # Rate divergence per neuron
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_diff = np.abs(lif_rates - adex_rates)
        rate_sum = lif_rates + adex_rates
        relative_diff = np.where(rate_sum > 0, 2 * rate_diff / rate_sum, 0.0)
    mean_relative_diff = float(np.mean(relative_diff))

    return {
        "lif": {
            "rates": lif_rates,
            "mean_rate": float(np.mean(lif_rates)),
            "sparseness": lif_sparseness,
            "active_fraction": lif_active,
            "group_rates": lif_group_rates,
            "group_active": lif_group_active,
            "result": lif_result,
        },
        "adex": {
            "rates": adex_rates,
            "mean_rate": float(np.mean(adex_rates)),
            "sparseness": adex_sparseness,
            "active_fraction": adex_active,
            "group_rates": adex_group_rates,
            "group_active": adex_group_active,
            "result": adex_result,
        },
        "comparison": {
            "rate_correlation": rate_correlation,
            "temporal_correlation": temporal_correlation,
            "mean_relative_diff": mean_relative_diff,
            "lif_mean_rate": float(np.mean(lif_rates)),
            "adex_mean_rate": float(np.mean(adex_rates)),
            "lif_sparseness": lif_sparseness,
            "adex_sparseness": adex_sparseness,
            "lif_active": lif_active,
            "adex_active": adex_active,
        },
    }


def adaptation_sweep(circuit, mb_neurons=None,
                     b_values=None, stimulus=None,
                     pn_rate_hz=50.0, pn_weight=68.75,
                     odor_fraction=0.1,
                     duration_ms=500.0, seed=42):
    """Sweep adaptation strength and compare with LIF.

    The adaptation parameter b controls how much the adaptation current
    increases with each spike. At b=0, AdEx reduces to exponential LIF
    (no adaptation). As b increases, firing rates decrease and inter-spike
    intervals become more irregular (adaptation).

    Parameters
    ----------
    circuit : Circuit
        Neural circuit.
    mb_neurons : pd.DataFrame, optional
        MB neuron annotations.
    b_values : list of float, optional
        Adaptation increments to test.
    stimulus : np.ndarray, optional
        Shared stimulus.
    pn_rate_hz, pn_weight, odor_fraction : float
        Stimulus parameters.
    duration_ms : float
        Simulation duration.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: b, adex_mean_rate, lif_mean_rate, rate_correlation,
        relative_diff.
    """
    if b_values is None:
        b_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    # Run LIF once (doesn't depend on b)
    rng = np.random.RandomState(seed)
    n_steps = int(duration_ms / 0.1)
    if stimulus is None and mb_neurons is not None:
        groups = neuron_groups(circuit, mb_neurons)
        pn_indices = groups.get("PN", np.array([], dtype=np.int32))
        if len(pn_indices) > 0:
            n_active = max(1, int(odor_fraction * len(pn_indices)))
            active_pns = rng.choice(pn_indices, size=n_active, replace=False)
            stimulus, _ = poisson_stimulus(
                circuit.n_neurons, n_steps, active_pns,
                rate_hz=pn_rate_hz, weight=pn_weight, seed=seed,
            )

    lif_result = simulate(circuit, duration=duration_ms, stimulus=stimulus, seed=seed)
    lif_rates = firing_rates(lif_result)
    lif_mean = float(np.mean(lif_rates))

    rows = []
    for b in b_values:
        params = AdExParams(delta_t=2.0, a=0.0, b=b, tau_w=100.0)
        adex_result = simulate_adex(
            circuit, adex_params=params,
            duration=duration_ms, stimulus=stimulus, seed=seed,
        )
        adex_rates = firing_rates(adex_result)
        adex_mean = float(np.mean(adex_rates))

        if np.std(lif_rates) > 0 and np.std(adex_rates) > 0:
            corr = float(np.corrcoef(lif_rates, adex_rates)[0, 1])
        else:
            corr = 1.0 if np.allclose(lif_rates, adex_rates) else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_sum = lif_rates + adex_rates
            rel_diff = np.where(
                rate_sum > 0,
                2 * np.abs(lif_rates - adex_rates) / rate_sum,
                0.0,
            )

        rows.append({
            "b": b,
            "adex_mean_rate": adex_mean,
            "lif_mean_rate": lif_mean,
            "rate_correlation": corr,
            "mean_relative_diff": float(np.mean(rel_diff)),
        })

    return pd.DataFrame(rows)


def comparison_report(results):
    """Print a structured LIF vs AdEx comparison report.

    Parameters
    ----------
    results : dict
        Output of compare_models().

    Returns
    -------
    str
        Formatted report.
    """
    c = results["comparison"]
    lif = results["lif"]
    adex = results["adex"]

    lines = [
        "=" * 60,
        "LIF vs AdEx COMPARISON — Topology vs Neuron Model",
        "=" * 60,
        "",
        "--- Global metrics ---",
        f"  {'Metric':25s} {'LIF':>10s} {'AdEx':>10s}",
        f"  {'-'*47}",
        f"  {'Mean rate (Hz)':25s} {lif['mean_rate']:10.1f} {adex['mean_rate']:10.1f}",
        f"  {'Active fraction':25s} {lif['active_fraction']:10.3f} {adex['active_fraction']:10.3f}",
        f"  {'Population sparseness':25s} {lif['sparseness']:10.3f} {adex['sparseness']:10.3f}",
        "",
        "--- Similarity metrics ---",
        f"  Rate correlation (neurons):  {c['rate_correlation']:.3f}",
        f"  Temporal correlation:        {c['temporal_correlation']:.3f}",
        f"  Mean relative difference:    {c['mean_relative_diff']:.3f}",
        "",
    ]

    # Per-group comparison
    if lif["group_rates"] and adex["group_rates"]:
        lines.append("--- Per-group mean rates (Hz) ---")
        lines.append(f"  {'Group':10s} {'LIF':>10s} {'AdEx':>10s} {'Diff':>10s}")
        lines.append(f"  {'-'*42}")
        for group in sorted(lif["group_rates"].keys()):
            lr = lif["group_rates"].get(group, 0.0)
            ar = adex["group_rates"].get(group, 0.0)
            lines.append(f"  {group:10s} {lr:10.1f} {ar:10.1f} {ar-lr:+10.1f}")
        lines.append("")

    # Interpretation
    lines.append("--- Interpretation ---")
    corr = c["rate_correlation"]
    if corr > 0.9:
        lines.append("  TOPOLOGY DOMINATES: rate correlation > 0.9.")
        lines.append("  The single-neuron model has minimal effect on network output.")
        lines.append("  This supports Zhang et al. (2024): connectivity determines")
        lines.append("  activation patterns; the neuron model is secondary.")
    elif corr > 0.7:
        lines.append("  PARTIAL AGREEMENT: rate correlation 0.7-0.9.")
        lines.append("  Topology explains most variance, but single-neuron dynamics")
        lines.append("  (adaptation, exponential spike initiation) contribute.")
    elif corr > 0.5:
        lines.append("  MODERATE DIVERGENCE: rate correlation 0.5-0.7.")
        lines.append("  Single-neuron model matters substantially. The AdEx features")
        lines.append("  (adaptation) change which neurons fire and how fast.")
    else:
        lines.append("  STRONG DIVERGENCE: rate correlation < 0.5.")
        lines.append("  Topology alone is NOT sufficient. The neuron model qualitatively")
        lines.append("  changes network dynamics for this circuit/stimulus combination.")

    rel_diff = c["mean_relative_diff"]
    lines.append(f"  Mean relative rate difference: {rel_diff:.1%}")
    if rel_diff < 0.1:
        lines.append("  Quantitative agreement is excellent (<10% relative difference).")
    elif rel_diff < 0.3:
        lines.append("  Quantitative agreement is reasonable (<30% relative difference).")
    else:
        lines.append("  Quantitative disagreement is substantial (>30% relative difference).")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
