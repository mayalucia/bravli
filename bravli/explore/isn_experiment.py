"""Inhibition-Stabilized Network (ISN) paradoxical response test.

Tests whether the mushroom body (or any circuit) operates in the
inhibition-stabilized regime. In an ISN, stimulating inhibitory neurons
causes a paradoxical *decrease* in network-level inhibition because:

1. Extra I-drive → I-cells fire more → E-cells fire less
2. E-cells fire less → I-cells lose excitatory recurrent input
3. Net effect: I-cell rate may decrease despite extra drive

The paradoxical response is the hallmark of strong recurrent excitation
balanced by inhibition (Tsodyks et al. 1997, J Neurosci).

References:
    Tsodyks MV et al. (1997). J Neurosci 17(11):4382-4388.
    Ozeki H et al. (2009). Neuron 62(4):578-592.
"""

import numpy as np
import pandas as pd

from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import step_stimulus, poisson_stimulus, combine_stimuli
from bravli.simulation.analysis import firing_rates, ei_balance, active_fraction_by_group

from bravli.utils import get_logger

LOG = get_logger("explore.isn")


def identify_ei_groups(circuit, mb_neurons):
    """Classify MB neurons as excitatory, inhibitory, or modulatory.

    Uses the neuron's dominant neurotransmitter:
    - Excitatory: acetylcholine (the dominant excitatory NT in Drosophila)
    - Inhibitory: GABA, glutamate (both inhibitory in the fly CNS)
    - Modulatory: dopamine, serotonin, octopamine

    Parameters
    ----------
    circuit : Circuit
        MB circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations with 'circuit_role' column.

    Returns
    -------
    dict
        {"E": np.ndarray, "I": np.ndarray, "modulatory": np.ndarray}
        Each value is an array of dense neuron indices.
    """
    from bravli.explore.mushroom_body import neuron_groups

    groups = neuron_groups(circuit, mb_neurons)
    labels = circuit.neuron_labels

    if labels is None or "top_nt" not in labels.columns:
        # Fallback: classify by circuit role
        # KCs and PNs are cholinergic (E), APL is GABAergic (I), DANs are modulatory
        e_indices = np.concatenate([
            groups.get("KC", np.array([], dtype=np.int32)),
            groups.get("PN", np.array([], dtype=np.int32)),
            groups.get("MBON", np.array([], dtype=np.int32)),
        ])
        i_indices = np.concatenate([
            groups.get("APL", np.array([], dtype=np.int32)),
            groups.get("MBIN", np.array([], dtype=np.int32)),
        ])
        mod_indices = groups.get("DAN", np.array([], dtype=np.int32))
        LOG.warning("No 'top_nt' column — using circuit role for E/I classification")
    else:
        nt = labels["top_nt"].values
        all_indices = np.arange(circuit.n_neurons)

        exc_nts = {"acetylcholine"}
        inh_nts = {"GABA", "glutamate"}
        mod_nts = {"dopamine", "serotonin", "octopamine"}

        e_mask = np.array([str(n) in exc_nts for n in nt])
        i_mask = np.array([str(n) in inh_nts for n in nt])
        mod_mask = np.array([str(n) in mod_nts for n in nt])

        e_indices = all_indices[e_mask].astype(np.int32)
        i_indices = all_indices[i_mask].astype(np.int32)
        mod_indices = all_indices[mod_mask].astype(np.int32)

    LOG.info("E/I groups: %d E, %d I, %d modulatory",
             len(e_indices), len(i_indices), len(mod_indices))

    return {"E": e_indices, "I": i_indices, "modulatory": mod_indices}


def isn_experiment(circuit, mb_neurons, e_drive=10.0, i_perturbation=5.0,
                   duration_ms=800.0, onset_ms=100.0,
                   baseline_epoch=(100, 400), perturbation_epoch=(400, 700),
                   stimulus_mode="step", pn_rate_hz=50.0, seed=42):
    """Run the ISN paradoxical response test.

    Protocol:
    1. Onset (0 - onset_ms): silence, let network settle
    2. Baseline epoch: drive E neurons (or PNs) to establish ongoing activity
    3. Perturbation epoch: additionally drive I neurons with extra current
    4. Measure: compare E-cell and I-cell rates between epochs

    Paradoxical response: if E-cell rate INCREASES (or I-cell rate DECREASES)
    during the perturbation despite extra I-cell drive.

    Parameters
    ----------
    circuit : Circuit
        MB circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    e_drive : float
        Amplitude of excitatory drive (mV for step, Hz for Poisson).
    i_perturbation : float
        Additional amplitude applied to I neurons during perturbation epoch.
    duration_ms : float
        Total simulation duration.
    onset_ms : float
        Time before any stimulus.
    baseline_epoch : tuple
        (start_ms, end_ms) for baseline measurement.
    perturbation_epoch : tuple
        (start_ms, end_ms) for perturbation measurement.
    stimulus_mode : str
        "step" for DC current, "poisson" for Poisson input.
    pn_rate_hz : float
        PN firing rate if using Poisson mode.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results including:
        - 'e_rate_baseline', 'e_rate_perturbation': mean E-cell rates
        - 'i_rate_baseline', 'i_rate_perturbation': mean I-cell rates
        - 'e_rate_change': perturbation - baseline for E cells
        - 'i_rate_change': perturbation - baseline for I cells
        - 'paradoxical_response': bool — True if E rate increased
        - 'result': SimulationResult
        - 'ei_groups': E/I group indices
    """
    from bravli.explore.mushroom_body import neuron_groups

    ei_groups = identify_ei_groups(circuit, mb_neurons)
    groups = neuron_groups(circuit, mb_neurons)
    e_indices = ei_groups["E"]
    i_indices = ei_groups["I"]

    n_steps = int(duration_ms / 0.1)
    pn_indices = groups.get("PN", np.array([], dtype=np.int32))

    # Build stimulus
    if stimulus_mode == "poisson" and len(pn_indices) > 0:
        # Drive PNs with Poisson input throughout both epochs
        stim_base, _ = poisson_stimulus(
            circuit.n_neurons, n_steps, pn_indices,
            rate_hz=pn_rate_hz, weight=68.75, seed=seed,
        )
    else:
        # Step current to E neurons during both epochs
        stim_base, _ = step_stimulus(
            circuit.n_neurons, n_steps, e_indices,
            amplitude=e_drive,
            start_ms=baseline_epoch[0], end_ms=perturbation_epoch[1],
        )

    # Additional I-cell drive during perturbation epoch only
    stim_perturb, _ = step_stimulus(
        circuit.n_neurons, n_steps, i_indices,
        amplitude=i_perturbation,
        start_ms=perturbation_epoch[0], end_ms=perturbation_epoch[1],
    )

    stim_total = combine_stimuli(stim_base, stim_perturb)

    # Run simulation
    result = simulate(
        circuit, duration=duration_ms, dt=0.1, stimulus=stim_total,
        record_v=True,
        record_idx=list(range(min(20, circuit.n_neurons))),
        seed=seed,
    )

    # Measure rates per epoch
    e_rates_base = firing_rates(result, time_window=baseline_epoch)
    e_rates_pert = firing_rates(result, time_window=perturbation_epoch)
    i_rates_base = firing_rates(result, time_window=baseline_epoch)
    i_rates_pert = firing_rates(result, time_window=perturbation_epoch)

    e_mean_base = np.mean(e_rates_base[e_indices]) if len(e_indices) > 0 else 0.0
    e_mean_pert = np.mean(e_rates_pert[e_indices]) if len(e_indices) > 0 else 0.0
    i_mean_base = np.mean(i_rates_base[i_indices]) if len(i_indices) > 0 else 0.0
    i_mean_pert = np.mean(i_rates_pert[i_indices]) if len(i_indices) > 0 else 0.0

    e_change = e_mean_pert - e_mean_base
    i_change = i_mean_pert - i_mean_base

    # Paradoxical response: E-cell rate increases despite extra I drive,
    # OR I-cell rate paradoxically decreases
    paradoxical = (e_change > 0) or (i_change < 0)

    LOG.info("ISN test: E rate %.1f -> %.1f Hz (delta=%.1f), "
             "I rate %.1f -> %.1f Hz (delta=%.1f), paradoxical=%s",
             e_mean_base, e_mean_pert, e_change,
             i_mean_base, i_mean_pert, i_change, paradoxical)

    return {
        "e_rate_baseline": e_mean_base,
        "e_rate_perturbation": e_mean_pert,
        "i_rate_baseline": i_mean_base,
        "i_rate_perturbation": i_mean_pert,
        "e_rate_change": e_change,
        "i_rate_change": i_change,
        "paradoxical_response": paradoxical,
        "result": result,
        "ei_groups": ei_groups,
        "stimulus_mode": stimulus_mode,
        "e_drive": e_drive,
        "i_perturbation": i_perturbation,
    }


def isn_dose_response(circuit, mb_neurons, i_amplitudes=None, **kwargs):
    """Run ISN test at multiple I-cell perturbation strengths.

    Parameters
    ----------
    circuit : Circuit
        MB circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    i_amplitudes : list of float, optional
        Perturbation amplitudes to test. Default: [1, 2, 5, 10, 20].
    **kwargs
        Passed to isn_experiment().

    Returns
    -------
    list of dict
        One result dict per amplitude.
    """
    if i_amplitudes is None:
        i_amplitudes = [1.0, 2.0, 5.0, 10.0, 20.0]

    results = []
    for amp in i_amplitudes:
        LOG.info("ISN dose-response: i_perturbation=%.1f", amp)
        r = isn_experiment(circuit, mb_neurons, i_perturbation=amp, **kwargs)
        results.append(r)

    return results


def isn_report(results):
    """Print a structured ISN analysis report.

    Parameters
    ----------
    results : dict or list of dict
        Output of isn_experiment() or isn_dose_response().

    Returns
    -------
    str
        Formatted report.
    """
    if isinstance(results, dict):
        results = [results]

    lines = [
        "=" * 60,
        "ISN PARADOXICAL RESPONSE TEST",
        "=" * 60,
        "",
    ]

    for i, r in enumerate(results):
        lines.append(f"--- Perturbation amplitude: {r['i_perturbation']:.1f} ---")
        lines.append(f"  E-cell rate: {r['e_rate_baseline']:.1f} -> "
                     f"{r['e_rate_perturbation']:.1f} Hz "
                     f"(delta = {r['e_rate_change']:+.1f})")
        lines.append(f"  I-cell rate: {r['i_rate_baseline']:.1f} -> "
                     f"{r['i_rate_perturbation']:.1f} Hz "
                     f"(delta = {r['i_rate_change']:+.1f})")
        lines.append(f"  Paradoxical response: {r['paradoxical_response']}")
        lines.append("")

    # Interpretation
    any_paradoxical = any(r["paradoxical_response"] for r in results)
    lines.append("--- Interpretation ---")
    if any_paradoxical:
        lines.append("  PARADOXICAL RESPONSE DETECTED.")
        lines.append("  The circuit operates in the inhibition-stabilized regime (ISN).")
        lines.append("  Recurrent excitation is strong enough that adding inhibition")
        lines.append("  causes a net disinhibitory effect through the E-I loop.")
    else:
        lines.append("  No paradoxical response detected.")
        lines.append("  The circuit does NOT appear to be in the ISN regime")
        lines.append("  at the tested perturbation strengths. This could mean:")
        lines.append("  - Recurrent excitation is too weak (not ISN)")
        lines.append("  - The perturbation is too small to trigger the paradox")
        lines.append("  - The circuit operates in a non-ISN balanced regime")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
