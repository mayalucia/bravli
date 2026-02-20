"""Neuromodulatory state switching in the mushroom body.

Demonstrates Marder's principle: the same connectome produces different
behavioral outputs depending on neuromodulatory state. We implement
multiplicative gain modulation per compartment:

    w_eff = w_base * m(compartment)

where m > 1 enhances and m < 1 suppresses synaptic transmission in
that compartment. By shifting m values across compartments, the same
odor input drives different MBON output patterns — and thus different
behavioral decisions (approach vs avoidance).

References:
    Marder E, Thirumalai V (2002). Neural Networks 15(4-6):507-524.
    Marder E (2012). Neuron 76(1):1-11.
    Aso Y et al. (2014). eLife 3:e04577.
    Cohn R et al. (2015). Cell 163(7):1742-1753.
"""

import numpy as np
import pandas as pd

from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import poisson_stimulus
from bravli.simulation.analysis import firing_rates, active_fraction_by_group
from bravli.explore.mushroom_body import neuron_groups
from bravli.explore.mb_compartments import (
    MB_COMPARTMENTS, build_compartment_index,
)

from bravli.utils import get_logger

LOG = get_logger("explore.neuromodulation")


# ---------------------------------------------------------------------------
# Modulatory state definitions
# ---------------------------------------------------------------------------

# Predefined modulatory states: per-compartment gain factors.
# Each state represents a different neuromodulatory context.
MODULATORY_STATES = {
    "naive": {
        # Baseline: all compartments at unity gain
        # The circuit's "default" state — no bias toward approach or avoidance
    },
    "appetitive": {
        # PAM DANs active: enhance appetitive compartments,
        # suppress aversive compartments.
        # Biases MBON output toward approach.
        "gamma2": 1.5,
        "gamma3": 1.5,
        "gamma4": 1.5,
        "gamma5": 1.5,
        "alpha1p": 1.3,
        "alpha2p": 1.3,
        "beta2p": 1.3,
        "gamma1": 0.6,
        "beta1p": 0.6,
        "alpha1": 0.6,
        "alpha2": 0.6,
        "alpha3": 0.6,
        "beta1": 0.6,
    },
    "aversive": {
        # PPL1 DANs active: enhance aversive compartments,
        # suppress appetitive compartments.
        # Biases MBON output toward avoidance.
        "gamma1": 1.5,
        "beta1p": 1.5,
        "alpha1": 1.5,
        "alpha2": 1.5,
        "alpha3": 1.5,
        "beta1": 1.5,
        "gamma2": 0.6,
        "gamma3": 0.6,
        "gamma4": 0.6,
        "gamma5": 0.6,
        "alpha1p": 0.6,
        "alpha2p": 0.6,
        "beta2p": 0.6,
    },
    "aroused": {
        # Global gain increase — octopamine-like arousal state.
        # All compartments enhanced; increases overall responsiveness.
        "gamma1": 1.3, "gamma2": 1.3, "gamma3": 1.3,
        "gamma4": 1.3, "gamma5": 1.3,
        "alpha1p": 1.3, "alpha2p": 1.3, "alpha3p": 1.3,
        "beta1p": 1.3, "beta2p": 1.3,
        "alpha1": 1.3, "alpha2": 1.3, "alpha3": 1.3,
        "beta1": 1.3, "beta2": 1.3,
    },
    "quiescent": {
        # Global gain decrease — sleep/rest state.
        # All compartments suppressed; reduces responsiveness.
        "gamma1": 0.5, "gamma2": 0.5, "gamma3": 0.5,
        "gamma4": 0.5, "gamma5": 0.5,
        "alpha1p": 0.5, "alpha2p": 0.5, "alpha3p": 0.5,
        "beta1p": 0.5, "beta2p": 0.5,
        "alpha1": 0.5, "alpha2": 0.5, "alpha3": 0.5,
        "beta1": 0.5, "beta2": 0.5,
    },
}


def apply_modulatory_state(circuit, comp_index, state_gains):
    """Apply multiplicative gain modulation to circuit weights.

    For each compartment, scales KC->MBON synaptic weights by the
    gain factor m: w_eff = w_base * m.

    Parameters
    ----------
    circuit : Circuit
        MB circuit (weights are modified in-place).
    comp_index : dict
        From build_compartment_index().
    state_gains : dict
        Compartment name -> gain factor. Missing compartments get m=1.0.

    Returns
    -------
    dict
        Per-compartment: n_synapses modulated, gain applied.
    """
    report = {}
    for comp, info in comp_index.items():
        gain = state_gains.get(comp, 1.0)
        mask = info["kc_mbon_syn_mask"]
        n_syn = int(mask.sum())
        if n_syn > 0 and gain != 1.0:
            circuit.weights[mask] *= gain
        report[comp] = {"n_synapses": n_syn, "gain": gain}
    return report


def restore_weights(circuit, original_weights):
    """Restore circuit weights to their original values.

    Parameters
    ----------
    circuit : Circuit
        Circuit with possibly modified weights.
    original_weights : np.ndarray
        Original weight array to restore.
    """
    circuit.weights[:] = original_weights


def compute_valence_score(rates, mbon_indices, comp_index):
    """Compute approach vs avoidance valence from MBON firing rates.

    Each MBON's rate is weighted by its compartment valence:
    - Appetitive compartment MBONs contribute positively (approach)
    - Aversive compartment MBONs contribute negatively (avoidance)
    - Mixed/unknown contribute zero

    The valence score is:
        V = sum(rate_appetitive) - sum(rate_aversive)
        V > 0 -> approach, V < 0 -> avoidance

    Parameters
    ----------
    rates : np.ndarray
        Per-neuron firing rates.
    mbon_indices : np.ndarray
        MBON neuron indices.
    comp_index : dict
        Compartment index from build_compartment_index().

    Returns
    -------
    dict
        valence_score : float
        approach_drive : float (sum of appetitive MBON rates)
        avoidance_drive : float (sum of aversive MBON rates)
        per_compartment : dict of compartment -> mean MBON rate
    """
    approach_drive = 0.0
    avoidance_drive = 0.0
    per_comp = {}
    mbon_set = set(mbon_indices.tolist())

    for comp, info in comp_index.items():
        comp_mbons = info["mbon_indices"]
        # Only count MBONs that are in our mbon_indices
        valid = np.array([m for m in comp_mbons if m in mbon_set])
        if len(valid) == 0:
            continue
        mean_rate = float(np.mean(rates[valid]))
        per_comp[comp] = mean_rate
        valence = info["valence"]
        if valence == "appetitive":
            approach_drive += mean_rate
        elif valence == "aversive":
            avoidance_drive += mean_rate

    return {
        "valence_score": approach_drive - avoidance_drive,
        "approach_drive": approach_drive,
        "avoidance_drive": avoidance_drive,
        "per_compartment": per_comp,
    }


def state_switching_experiment(circuit, mb_neurons,
                               states=None,
                               odor_fraction=0.1,
                               pn_rate_hz=50.0, pn_weight=68.75,
                               duration_ms=500.0, seed=42):
    """Run the same odor through the MB under different modulatory states.

    For each state:
    1. Apply the modulatory gain to KC->MBON weights
    2. Present the same odor (Poisson PN input)
    3. Measure MBON responses and compute valence score
    4. Restore original weights

    Parameters
    ----------
    circuit : Circuit
        MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    states : dict, optional
        State name -> compartment gain dict. If None, uses MODULATORY_STATES.
    odor_fraction : float
        Fraction of PNs to drive (odor identity).
    pn_rate_hz : float
        PN firing rate.
    pn_weight : float
        PN spike weight.
    duration_ms : float
        Simulation duration per state.
    seed : int
        Random seed.

    Returns
    -------
    dict
        state_name -> {
            rates, mbon_rates, valence, active_fractions,
            gain_report, result
        }
    """
    if states is None:
        states = MODULATORY_STATES

    dt = 0.1
    rng = np.random.RandomState(seed)
    groups = neuron_groups(circuit, mb_neurons)
    pn_indices = groups.get("PN", np.array([], dtype=np.int32))
    mbon_indices = groups.get("MBON", np.array([], dtype=np.int32))

    if len(pn_indices) == 0:
        LOG.error("No PN neurons found")
        return {"error": "No PN neurons"}

    # Select odor pattern (same for all states)
    n_active = max(1, int(odor_fraction * len(pn_indices)))
    active_pns = rng.choice(pn_indices, size=n_active, replace=False)

    # Build stimulus (same for all states)
    n_steps = int(duration_ms / dt)
    stim, _ = poisson_stimulus(
        circuit.n_neurons, n_steps, active_pns,
        rate_hz=pn_rate_hz, weight=pn_weight, seed=seed,
    )

    # Build compartment index
    comp_index = build_compartment_index(circuit, mb_neurons)

    # Save original weights
    original_weights = circuit.weights.copy()

    results = {}
    for state_idx, (state_name, state_gains) in enumerate(states.items()):
        LOG.info("State: %s", state_name)

        # Restore original weights before applying new state
        restore_weights(circuit, original_weights)

        # Apply modulatory state
        gain_report = apply_modulatory_state(circuit, comp_index, state_gains)

        # Simulate
        result = simulate(
            circuit, duration=duration_ms, dt=dt,
            stimulus=stim, seed=seed + state_idx,
        )

        # Analyze
        rates = firing_rates(result)
        mbon_rates = rates[mbon_indices] if len(mbon_indices) > 0 else np.array([])
        active = active_fraction_by_group(result, groups)
        valence = compute_valence_score(rates, mbon_indices, comp_index)

        results[state_name] = {
            "rates": rates,
            "mbon_rates": mbon_rates,
            "mean_mbon_rate": float(np.mean(mbon_rates)) if len(mbon_rates) > 0 else 0.0,
            "valence": valence,
            "active_fractions": active,
            "gain_report": gain_report,
            "result": result,
        }

        LOG.info("  Mean MBON rate: %.1f Hz, valence score: %.1f",
                 results[state_name]["mean_mbon_rate"],
                 valence["valence_score"])

    # Restore original weights
    restore_weights(circuit, original_weights)

    return results


def dose_response(circuit, mb_neurons, target_compartments,
                  gain_values=None,
                  odor_fraction=0.1, pn_rate_hz=50.0, pn_weight=68.75,
                  duration_ms=500.0, seed=42):
    """Sweep modulatory gain strength for specific compartments.

    Applies a range of gain values to target compartments while keeping
    all others at unity. Measures how MBON output and valence change.

    Parameters
    ----------
    circuit : Circuit
        MB circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    target_compartments : list of str
        Compartments to modulate.
    gain_values : list of float, optional
        Gain values to test. Default: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0].
    odor_fraction : float
        Fraction of PNs to drive.
    pn_rate_hz : float
        PN firing rate.
    pn_weight : float
        PN spike weight.
    duration_ms : float
        Simulation duration.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: gain, mean_mbon_rate, valence_score, approach_drive,
        avoidance_drive.
    """
    if gain_values is None:
        gain_values = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]

    rows = []
    for gain in gain_values:
        state_gains = {comp: gain for comp in target_compartments}
        results = state_switching_experiment(
            circuit, mb_neurons,
            states={"test": state_gains},
            odor_fraction=odor_fraction,
            pn_rate_hz=pn_rate_hz,
            pn_weight=pn_weight,
            duration_ms=duration_ms,
            seed=seed,
        )
        if "error" in results:
            continue
        r = results["test"]
        rows.append({
            "gain": gain,
            "mean_mbon_rate": r["mean_mbon_rate"],
            "valence_score": r["valence"]["valence_score"],
            "approach_drive": r["valence"]["approach_drive"],
            "avoidance_drive": r["valence"]["avoidance_drive"],
        })

    return pd.DataFrame(rows)


def neuromodulation_report(results):
    """Print a structured report for the state switching experiment.

    Parameters
    ----------
    results : dict
        Output of state_switching_experiment().

    Returns
    -------
    str
        Formatted report.
    """
    if "error" in results:
        return f"Experiment failed: {results['error']}"

    lines = [
        "=" * 60,
        "NEUROMODULATORY STATE SWITCHING — Marder's Principle",
        "=" * 60,
        "",
        "Same connectome, same odor, different modulatory states.",
        "Question: does the circuit produce different behavioral outputs?",
        "",
        f"{'State':12s} {'MBON rate':>10s} {'Valence':>9s} {'Approach':>10s} {'Avoidance':>10s} {'Decision':>10s}",
        "-" * 63,
    ]

    for state_name, r in results.items():
        v = r["valence"]
        decision = "APPROACH" if v["valence_score"] > 0 else "AVOID"
        if abs(v["valence_score"]) < 1.0:
            decision = "NEUTRAL"
        lines.append(
            f"{state_name:12s} {r['mean_mbon_rate']:10.1f} "
            f"{v['valence_score']:+9.1f} "
            f"{v['approach_drive']:10.1f} {v['avoidance_drive']:10.1f} "
            f"{decision:>10s}"
        )

    lines.append("")

    # Per-compartment detail for each state
    lines.append("--- Per-compartment MBON rates ---")
    lines.append("")

    # Collect all compartments with data
    all_comps = set()
    for r in results.values():
        all_comps.update(r["valence"]["per_compartment"].keys())
    all_comps = sorted(all_comps)

    if all_comps:
        header = f"{'Compartment':12s} {'Valence':12s}"
        for state_name in results:
            header += f" {state_name:>10s}"
        lines.append(header)
        lines.append("-" * (24 + 11 * len(results)))

        for comp in all_comps:
            valence = MB_COMPARTMENTS.get(comp, {}).get("valence", "?")
            row = f"{comp:12s} {valence:12s}"
            for state_name, r in results.items():
                rate = r["valence"]["per_compartment"].get(comp, 0.0)
                row += f" {rate:10.1f}"
            lines.append(row)

    # Interpretation
    lines.append("")
    lines.append("--- Interpretation ---")

    state_names = list(results.keys())
    if len(state_names) >= 2:
        # Check if different states produce different decisions
        decisions = {}
        for state_name, r in results.items():
            v = r["valence"]["valence_score"]
            decisions[state_name] = "approach" if v > 0 else "avoid" if v < 0 else "neutral"

        unique_decisions = set(decisions.values())
        if len(unique_decisions) > 1:
            lines.append("  MARDER'S PRINCIPLE CONFIRMED: the same connectome produces")
            lines.append("  different behavioral outputs under different modulatory states.")
            for state, dec in decisions.items():
                lines.append(f"    {state}: {dec}")
        else:
            lines.append("  All states produce the same decision — modulatory gain")
            lines.append("  shifts rates but doesn't cross the approach/avoid boundary.")
            lines.append("  This may indicate the circuit is strongly biased toward")
            lines.append(f"  {list(unique_decisions)[0]}, or that gain modulation")
            lines.append("  alone is insufficient for behavioral switching.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
