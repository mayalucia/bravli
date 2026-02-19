"""Aversive olfactory conditioning experiment.

Simulates the standard Drosophila T-maze conditioning protocol:

1. Pre-test: present CS+ odor, measure MBON baseline response
2. Training: pair CS+ odor with DAN activation (electric shock proxy)
   — KC→MBON synapses depress via three-factor STDP
3. Post-test: present CS+ odor alone, measure MBON response (should decrease)
4. Control: present CS- (novel odor), measure MBONs (should be unchanged)

The "learning" emerges from compartment-specific KC→MBON depression:
KCs encoding the CS+ odor become eligible for plasticity, and DAN activation
converts that eligibility into synaptic depression. After training, the
approach-driving MBONs in the trained compartment respond less to CS+,
shifting the fly's behavior toward avoidance.

References:
    Hige T et al. (2015). Neuron 88(5):985-998.
    Aso Y et al. (2014). eLife 3:e04577.
    Handler A et al. (2019). Cell 178(1):60-75.
"""

import numpy as np

from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import poisson_stimulus, step_stimulus, combine_stimuli
from bravli.simulation.analysis import (
    firing_rates, mbon_response_change, performance_index, weight_evolution,
)
from bravli.simulation.plasticity import ThreeFactorSTDP
from bravli.explore.mushroom_body import neuron_groups
from bravli.explore.mb_compartments import build_compartment_index

from bravli.utils import get_logger

LOG = get_logger("explore.conditioning")


def _build_odor_stimulus(circuit, pn_indices, active_pn_indices, n_steps,
                         pn_rate_hz, pn_weight, start_step, end_step, seed):
    """Build a Poisson stimulus for a subset of PNs in a time window.

    Returns stimulus array (n_neurons, n_steps) with Poisson input
    only during [start_step, end_step) and only to active_pn_indices.
    """
    rng = np.random.RandomState(seed)
    stim = np.zeros((circuit.n_neurons, n_steps), dtype=np.float64)
    dt = 0.1  # ms
    p_spike = pn_rate_hz * dt / 1000.0

    for step in range(start_step, min(end_step, n_steps)):
        spikes = rng.random(len(active_pn_indices)) < p_spike
        stim[active_pn_indices[spikes], step] += pn_weight

    return stim


def _build_dan_stimulus(circuit, dan_indices, n_steps,
                        dan_rate_hz, dan_weight, start_step, end_step, seed):
    """Build a Poisson stimulus for DAN neurons in a time window."""
    rng = np.random.RandomState(seed + 10000)
    stim = np.zeros((circuit.n_neurons, n_steps), dtype=np.float64)
    dt = 0.1
    p_spike = dan_rate_hz * dt / 1000.0

    for step in range(start_step, min(end_step, n_steps)):
        spikes = rng.random(len(dan_indices)) < p_spike
        stim[dan_indices[spikes], step] += dan_weight

    return stim


def aversive_conditioning(circuit, mb_neurons, mb_edges=None,
                          cs_odor_fraction=0.1, us_compartment="gamma1",
                          n_training_trials=6, trial_duration_ms=500.0,
                          iti_ms=300.0, test_duration_ms=500.0,
                          pn_rate_hz=50.0, pn_weight=68.75,
                          dan_rate_hz=30.0, dan_weight=50.0,
                          lr=0.001, tau_eligibility=1000.0,
                          tau_dopamine=500.0, w_min=0.0,
                          seed=42):
    """Simulate aversive olfactory conditioning.

    The full protocol runs as a single continuous simulation so that
    weights evolve continuously (biologically correct). Time windows
    define trial boundaries for analysis.

    Protocol timeline (all in one simulation):
    - Phase 1: Pre-test CS+ (test_duration_ms)
    - Phase 2: N training trials, each = CS+ odor + DAN activation (trial_duration_ms)
               separated by ITIs (iti_ms, no stimulus)
    - Phase 3: Post-test CS+ (test_duration_ms)
    - Phase 4: Control CS- (different odor, test_duration_ms)

    Parameters
    ----------
    circuit : Circuit
        MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    mb_edges : pd.DataFrame, optional
        MB edge table (unused, kept for API consistency).
    cs_odor_fraction : float
        Fraction of PNs activated by the CS+ odor.
    us_compartment : str
        Compartment whose DANs carry the US (aversive signal).
        Default "gamma1" — PPL1-gamma1 carries electric shock signal.
    n_training_trials : int
        Number of CS+/US pairings.
    trial_duration_ms : float
        Duration of each training trial.
    iti_ms : float
        Inter-trial interval (silence).
    test_duration_ms : float
        Duration of pre-test and post-test presentations.
    pn_rate_hz : float
        PN firing rate for odor presentation.
    pn_weight : float
        Weight of PN Poisson spikes.
    dan_rate_hz : float
        DAN firing rate during US.
    dan_weight : float
        Weight of DAN Poisson spikes.
    lr : float
        Three-factor STDP learning rate.
    tau_eligibility : float
        Eligibility trace time constant (ms).
    tau_dopamine : float
        Dopamine signal time constant (ms).
    w_min : float
        Minimum weight for plastic synapses.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results including pre/post MBON rates, weight evolution,
        learning index, performance index, and timing info.
    """
    rng = np.random.RandomState(seed)
    dt = 0.1
    groups = neuron_groups(circuit, mb_neurons)
    pn_indices = groups.get("PN", np.array([], dtype=np.int32))

    if len(pn_indices) == 0:
        LOG.error("No PN neurons — cannot run conditioning")
        return {"error": "No PN neurons in circuit"}

    # Select CS+ and CS- odor patterns (non-overlapping PN subsets)
    n_active = max(1, int(cs_odor_fraction * len(pn_indices)))
    shuffled_pns = rng.permutation(pn_indices)
    cs_plus_pns = shuffled_pns[:n_active]
    cs_minus_pns = shuffled_pns[n_active:2 * n_active]
    if len(cs_minus_pns) < n_active:
        # If not enough PNs for non-overlapping, allow some overlap
        cs_minus_pns = rng.choice(pn_indices, size=n_active, replace=False)

    LOG.info("CS+ odor: %d PNs, CS- odor: %d PNs (overlap: %d)",
             len(cs_plus_pns), len(cs_minus_pns),
             len(set(cs_plus_pns) & set(cs_minus_pns)))

    # Build compartment index for plasticity
    comp_index = build_compartment_index(circuit, mb_neurons)

    # Get US DAN indices
    us_info = comp_index.get(us_compartment)
    if us_info is None or len(us_info["dan_indices"]) == 0:
        LOG.warning("No DANs in compartment %s — US will have no effect",
                    us_compartment)

    # Calculate total duration and phase boundaries
    # Phase 1: pre-test CS+
    pre_test_start = 0.0
    pre_test_end = test_duration_ms

    # Phase 2: training trials
    training_start = pre_test_end + iti_ms
    training_phases = []
    t = training_start
    for trial in range(n_training_trials):
        trial_start = t
        trial_end = t + trial_duration_ms
        training_phases.append((trial_start, trial_end))
        t = trial_end + iti_ms

    # Phase 3: post-test CS+
    post_test_start = t
    post_test_end = post_test_start + test_duration_ms

    # Phase 4: control CS-
    control_start = post_test_end + iti_ms
    control_end = control_start + test_duration_ms

    total_duration = control_end
    n_steps = int(total_duration / dt)

    LOG.info("Conditioning protocol: total %.0f ms, %d training trials",
             total_duration, n_training_trials)
    LOG.info("  Pre-test: %.0f-%.0f ms", pre_test_start, pre_test_end)
    for i, (ts, te) in enumerate(training_phases):
        LOG.info("  Training %d: %.0f-%.0f ms", i, ts, te)
    LOG.info("  Post-test: %.0f-%.0f ms", post_test_start, post_test_end)
    LOG.info("  Control: %.0f-%.0f ms", control_start, control_end)

    # Build stimulus array
    stim = np.zeros((circuit.n_neurons, n_steps), dtype=np.float64)

    def ms_to_step(ms):
        return int(ms / dt)

    # Pre-test: CS+ odor
    stim += _build_odor_stimulus(
        circuit, pn_indices, cs_plus_pns, n_steps,
        pn_rate_hz, pn_weight,
        ms_to_step(pre_test_start), ms_to_step(pre_test_end),
        seed=seed + 1,
    )

    # Training trials: CS+ odor + DAN activation
    for i, (ts, te) in enumerate(training_phases):
        stim += _build_odor_stimulus(
            circuit, pn_indices, cs_plus_pns, n_steps,
            pn_rate_hz, pn_weight,
            ms_to_step(ts), ms_to_step(te),
            seed=seed + 100 + i,
        )
        # DAN (US) activation — all DANs in the target compartment
        if us_info is not None and len(us_info["dan_indices"]) > 0:
            stim += _build_dan_stimulus(
                circuit, us_info["dan_indices"], n_steps,
                dan_rate_hz, dan_weight,
                ms_to_step(ts), ms_to_step(te),
                seed=seed + 200 + i,
            )

    # Post-test: CS+ odor alone
    stim += _build_odor_stimulus(
        circuit, pn_indices, cs_plus_pns, n_steps,
        pn_rate_hz, pn_weight,
        ms_to_step(post_test_start), ms_to_step(post_test_end),
        seed=seed + 2,
    )

    # Control: CS- odor
    stim += _build_odor_stimulus(
        circuit, pn_indices, cs_minus_pns, n_steps,
        pn_rate_hz, pn_weight,
        ms_to_step(control_start), ms_to_step(control_end),
        seed=seed + 3,
    )

    # Build plasticity rule
    plasticity = ThreeFactorSTDP(
        compartment_index=comp_index,
        tau_eligibility=tau_eligibility,
        tau_dopamine=tau_dopamine,
        lr=lr,
        w_min=w_min,
        snapshot_interval_ms=100.0,
    )

    # Save initial weights
    initial_weights = circuit.weights.copy()

    # Run simulation
    LOG.info("Running conditioning simulation (%.0f ms)...", total_duration)
    result = simulate(
        circuit, duration=total_duration, dt=dt, stimulus=stim,
        plasticity_fn=plasticity, seed=seed,
    )

    # Analyze results
    mbon_indices = groups.get("MBON", np.array([], dtype=np.int32))

    pre_rates = firing_rates(result, time_window=(pre_test_start, pre_test_end))
    post_rates = firing_rates(result, time_window=(post_test_start, post_test_end))
    control_rates = firing_rates(result, time_window=(control_start, control_end))

    # Per-trial MBON rates during training
    trial_mbon_rates = []
    for ts, te in training_phases:
        trial_rates = firing_rates(result, time_window=(ts, te))
        if len(mbon_indices) > 0:
            trial_mbon_rates.append(float(np.mean(trial_rates[mbon_indices])))
        else:
            trial_mbon_rates.append(0.0)

    # Learning index
    li = mbon_response_change(pre_rates, post_rates, mbon_indices)

    # Performance index
    pi = performance_index(post_rates, control_rates, mbon_indices)

    # Weight evolution
    w_evo = weight_evolution(plasticity.weight_snapshots, plasticity.snapshot_times)

    # Weight change summary
    w_summary = plasticity.weight_change_summary(circuit, initial_weights)

    results = {
        # Rates
        "pre_test_rates": pre_rates,
        "post_test_rates": post_rates,
        "control_rates": control_rates,
        "trial_mbon_rates": trial_mbon_rates,
        # MBON-specific
        "mbon_indices": mbon_indices,
        "pre_mbon_mean": float(np.mean(pre_rates[mbon_indices])) if len(mbon_indices) > 0 else 0.0,
        "post_mbon_mean": float(np.mean(post_rates[mbon_indices])) if len(mbon_indices) > 0 else 0.0,
        "control_mbon_mean": float(np.mean(control_rates[mbon_indices])) if len(mbon_indices) > 0 else 0.0,
        # Learning metrics
        "learning_index": li,
        "mean_learning_index": float(np.mean(li)) if len(li) > 0 else 0.0,
        "performance_index": pi,
        "mean_performance_index": float(np.mean(pi)) if len(pi) > 0 else 0.0,
        # Weight evolution
        "weight_evolution": w_evo,
        "weight_change_summary": w_summary,
        # Plasticity object (for inspection)
        "plasticity": plasticity,
        "initial_weights": initial_weights,
        # Protocol info
        "result": result,
        "cs_plus_pns": cs_plus_pns,
        "cs_minus_pns": cs_minus_pns,
        "us_compartment": us_compartment,
        "n_training_trials": n_training_trials,
        "timing": {
            "pre_test": (pre_test_start, pre_test_end),
            "training": training_phases,
            "post_test": (post_test_start, post_test_end),
            "control": (control_start, control_end),
        },
    }

    LOG.info("Conditioning complete. MBON rates: pre=%.1f, post=%.1f, control=%.1f Hz",
             results["pre_mbon_mean"], results["post_mbon_mean"],
             results["control_mbon_mean"])
    LOG.info("Mean learning index: %.3f, mean performance index: %.3f",
             results["mean_learning_index"], results["mean_performance_index"])

    return results


def conditioning_report(results):
    """Print a structured conditioning analysis report.

    Parameters
    ----------
    results : dict
        Output of aversive_conditioning().

    Returns
    -------
    str
        Formatted report.
    """
    if "error" in results:
        return f"Conditioning failed: {results['error']}"

    timing = results["timing"]
    lines = [
        "=" * 60,
        "AVERSIVE OLFACTORY CONDITIONING — Report",
        "=" * 60,
        "",
        "--- Protocol ---",
        f"  US compartment:     {results['us_compartment']}",
        f"  Training trials:    {results['n_training_trials']}",
        f"  CS+ PNs:            {len(results['cs_plus_pns'])}",
        f"  CS- PNs:            {len(results['cs_minus_pns'])}",
        "",
        "--- MBON Response ---",
        f"  Pre-training:       {results['pre_mbon_mean']:.1f} Hz",
        f"  Post-training:      {results['post_mbon_mean']:.1f} Hz",
        f"  Control (CS-):      {results['control_mbon_mean']:.1f} Hz",
        "",
        f"  Response change:    {results['post_mbon_mean'] - results['pre_mbon_mean']:+.1f} Hz",
        "",
        "--- Learning Metrics ---",
        f"  Mean learning index:     {results['mean_learning_index']:.3f}",
        f"  Mean performance index:  {results['mean_performance_index']:.3f}",
        "",
        "--- Training curve (MBON rate per trial) ---",
    ]

    for i, rate in enumerate(results["trial_mbon_rates"]):
        lines.append(f"  Trial {i}: {rate:.1f} Hz")

    # Weight changes
    ws = results["weight_change_summary"]
    lines.append("")
    lines.append("--- Weight Changes ---")
    lines.append(f"  Mean weight change: {ws['global_mean_change']:.4f} "
                 f"({ws['global_mean_pct_change']:.1f}%)")
    lines.append(f"  Synapses depressed: {ws['n_depressed']}")
    lines.append(f"  Synapses unchanged: {ws['n_unchanged']}")

    for comp, cinfo in ws.get("compartments", {}).items():
        lines.append(f"  {comp}: {cinfo['n_depressed']}/{cinfo['n_synapses']} "
                     f"depressed (mean Δ={cinfo['mean_change']:.4f})")

    # Interpretation
    lines.append("")
    lines.append("--- Interpretation ---")
    li = results["mean_learning_index"]
    pi = results["mean_performance_index"]

    if li > 0.05:
        lines.append("  LEARNING DETECTED: MBONs respond less to CS+ after training.")
        lines.append(f"  Learning index {li:.3f} > 0 indicates synaptic depression.")
    elif li > -0.05:
        lines.append("  MARGINAL: Learning index near zero — weak or no learning.")
    else:
        lines.append("  NO LEARNING: MBONs respond MORE to CS+ after training.")
        lines.append("  This is unexpected — check parameters.")

    if pi > 0.05:
        lines.append(f"  DISCRIMINATION: PI={pi:.3f} > 0 — fly distinguishes CS+ from CS-.")
    else:
        lines.append(f"  NO DISCRIMINATION: PI={pi:.3f} — similar response to both odors.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
