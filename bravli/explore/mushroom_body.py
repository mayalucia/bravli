"""Explore the mushroom body: bravli's integration demonstration.

This script ties together all bravli modules (parcellation, composition,
factology, visualization) to produce a comprehensive analysis of the
fly brain's mushroom body — the center for olfactory learning and memory.

Usage:
    python -m bravli.explore.mushroom_body <path-to-annotations-tsv>

Or import individual functions:
    from bravli.explore.mushroom_body import explore_mushroom_body
"""

from pathlib import Path

import pandas as pd
import numpy as np

from bravli.parcellation.load_flywire import load_flywire_annotations
from bravli.composition.composition import (
    cell_type_distribution,
    neurotransmitter_profile,
    top_types,
)
from bravli.factology.factology import NeuropilFacts
from bravli.utils import get_logger

LOG = get_logger("explore.mb")


# ---------------------------------------------------------------------------
# Constants: which cell_class values belong to the mushroom body circuit
# ---------------------------------------------------------------------------

MB_CELL_CLASSES = ["Kenyon_Cell", "MBON", "MBIN", "DAN"]
"""Cell classes that form the core mushroom body circuit.

- Kenyon_Cell: ~5,200 intrinsic neurons forming the sparse odor code
- MBON: ~100 mushroom body output neurons driving behavior
- MBIN: mushroom body input neurons (non-DAN modulatory, includes APL)
- DAN: ~330 dopaminergic neurons carrying reinforcement signals
"""

MB_AFFERENT_CLASSES = ["ALPN"]
"""Cell classes providing afferent input to the mushroom body.

- ALPN: ~685 antennal lobe projection neurons carrying odor representations
"""

MB_ALL_CLASSES = MB_CELL_CLASSES + MB_AFFERENT_CLASSES
"""All cell classes in the extended MB circuit (core + afferents)."""

KC_SUBTYPES = {
    "gamma":       ["KCg-m", "KCg-d"],
    "alpha_beta":  ["KCab", "KCab-p"],
    "alpha_beta_prime": ["KCapbp-m", "KCapbp-ap1", "KCapbp-ap2"],
}
"""Kenyon cell subtype groupings by lobe projection."""


# ---------------------------------------------------------------------------
# Step 1: Extract mushroom body neurons
# ---------------------------------------------------------------------------

def extract_mb_neurons(annotations):
    """Filter annotations to mushroom body circuit neurons.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full FlyWire annotation table.

    Returns
    -------
    pd.DataFrame
        Subset containing only MB-related neurons.
    """
    mask = annotations["cell_class"].isin(MB_CELL_CLASSES)
    mb = annotations[mask].copy()
    LOG.info("Extracted %d MB neurons from %d total", len(mb), len(annotations))
    return mb


# ---------------------------------------------------------------------------
# Step 2: Composition analysis
# ---------------------------------------------------------------------------

def mb_composition(mb_neurons):
    """Analyze the composition of the mushroom body.

    Returns a dict with:
    - class_counts: neuron count per cell_class
    - kc_subtypes: count per KC subtype group
    - nt_profile: neurotransmitter breakdown
    - top_cell_types: top 20 most abundant cell types
    - hemisphere_balance: left vs right counts

    Parameters
    ----------
    mb_neurons : pd.DataFrame
        MB-filtered annotations (from extract_mb_neurons).

    Returns
    -------
    dict
    """
    result = {}

    # Count by cell class
    result["class_counts"] = (
        mb_neurons["cell_class"]
        .value_counts()
        .to_dict()
    )

    # Kenyon cell subtype breakdown
    kc = mb_neurons[mb_neurons["cell_class"] == "Kenyon_Cell"]
    kc_groups = {}
    for group_name, types in KC_SUBTYPES.items():
        kc_groups[group_name] = kc["cell_type"].isin(types).sum()
    kc_groups["other"] = len(kc) - sum(kc_groups.values())
    result["kc_subtypes"] = kc_groups

    # Neurotransmitter profile
    result["nt_profile"] = neurotransmitter_profile(mb_neurons)

    # Top cell types
    result["top_cell_types"] = top_types(mb_neurons, n=20)

    # Hemisphere balance
    if "side" in mb_neurons.columns:
        result["hemisphere_balance"] = (
            mb_neurons["side"]
            .value_counts()
            .to_dict()
        )

    return result


# ---------------------------------------------------------------------------
# Step 3: Structured factsheet
# ---------------------------------------------------------------------------

def mb_factsheet(mb_neurons, target="mushroom_body"):
    """Generate a structured factsheet for the mushroom body.

    Uses the NeuropilFacts factology class from Lesson 03 to produce
    typed, named facts in a standardized format.

    Parameters
    ----------
    mb_neurons : pd.DataFrame
        MB-filtered annotations.
    target : str
        Name for the factsheet target.

    Returns
    -------
    pd.DataFrame
        Factsheet with columns: name, value, unit, category.
    """
    facts = NeuropilFacts(annotations=mb_neurons, target=target)
    df = facts.to_dataframe()
    LOG.info("Generated factsheet with %d facts for '%s'", len(df), target)
    return df


# ---------------------------------------------------------------------------
# Step 4: Summary report
# ---------------------------------------------------------------------------

def mb_summary_report(annotations):
    """Run the complete MB exploration pipeline and print a report.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full FlyWire annotation table.

    Returns
    -------
    dict
        Keys: 'mb_neurons', 'composition', 'factsheet'
    """
    # Extract
    mb = extract_mb_neurons(annotations)

    # Compose
    comp = mb_composition(mb)

    # Factsheet
    factsheet = mb_factsheet(mb)

    # Print report
    lines = [
        "=" * 60,
        "MUSHROOM BODY — Exploration Report",
        "=" * 60,
        "",
        f"Total MB neurons: {len(mb):,}",
        "",
        "--- Cell class breakdown ---",
    ]
    for cls, n in sorted(comp["class_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:20s}: {n:,}")

    lines.append("")
    lines.append("--- Kenyon cell subtypes ---")
    for group, n in comp["kc_subtypes"].items():
        lines.append(f"  {group:25s}: {n:,}")

    lines.append("")
    lines.append("--- Neurotransmitter profile ---")
    nt = comp["nt_profile"]
    if isinstance(nt, pd.DataFrame) and "neuron_count" in nt.columns:
        for nt_name, row in nt.iterrows():
            lines.append(f"  {str(nt_name):20s}: {int(row['neuron_count']):,}"
                         f"  ({row.get('proportion', 0):.1%})")
    elif isinstance(nt, (pd.Series, dict)):
        for nt_name, n in (nt.items() if hasattr(nt, 'items') else nt.items()):
            lines.append(f"  {str(nt_name):20s}: {n:,}")

    lines.append("")
    lines.append("--- Hemisphere balance ---")
    for side, n in comp.get("hemisphere_balance", {}).items():
        lines.append(f"  {side:10s}: {n:,}")

    lines.append("")
    lines.append("--- Top 10 cell types ---")
    top = comp["top_cell_types"]
    if isinstance(top, pd.DataFrame) and "neuron_count" in top.columns:
        for ct, row in top.head(10).iterrows():
            lines.append(f"  {str(ct):20s}: {int(row['neuron_count']):,}"
                         f"  ({row.get('proportion', 0):.1%})")
    elif isinstance(top, (pd.Series, dict)):
        items = top.head(10).items() if hasattr(top, 'head') else list(top.items())[:10]
        for ct, n in items:
            lines.append(f"  {str(ct):20s}: {n:,}")

    lines.append("")
    lines.append("--- Structured Factsheet ---")
    lines.append(factsheet.to_string(index=False))
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    return {
        "mb_neurons": mb,
        "composition": comp,
        "factsheet": factsheet,
    }


# ---------------------------------------------------------------------------
# Step 5: MB circuit extraction from connectome
# ---------------------------------------------------------------------------

def extract_mb_circuit(annotations, edges, include_afferents=True):
    """Extract mushroom body neurons and their internal connectivity.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full FlyWire annotation table.
    edges : pd.DataFrame
        Processed edge table with columns: pre_pt_root_id,
        post_pt_root_id, syn_count, weight, dominant_nt.
    include_afferents : bool
        If True, include ALPNs (projection neurons) as input layer.

    Returns
    -------
    mb_neurons : pd.DataFrame
        MB neuron annotations with added 'circuit_role' column.
    mb_edges : pd.DataFrame
        Edges internal to the MB circuit.
    """
    # Select cell classes
    classes = MB_ALL_CLASSES if include_afferents else MB_CELL_CLASSES
    mask = annotations["cell_class"].isin(classes)
    mb_neurons = annotations[mask].copy()

    # Assign circuit roles
    role_map = {
        "Kenyon_Cell": "KC",
        "MBON": "MBON",
        "MBIN": "MBIN",
        "DAN": "DAN",
        "ALPN": "PN",
    }
    mb_neurons["circuit_role"] = mb_neurons["cell_class"].map(role_map)

    # Tag APL specifically (it's MBIN but functionally distinct)
    apl_mask = mb_neurons["cell_type"].str.contains("APL", na=False)
    mb_neurons.loc[apl_mask, "circuit_role"] = "APL"

    # Filter edges to MB-internal connections
    mb_ids = set(mb_neurons["root_id"].values)
    edge_mask = (edges["pre_pt_root_id"].isin(mb_ids) &
                 edges["post_pt_root_id"].isin(mb_ids))
    mb_edges = edges[edge_mask].copy()

    # Log statistics
    role_counts = mb_neurons["circuit_role"].value_counts()
    LOG.info("MB circuit: %d neurons (%s), %d edges",
             len(mb_neurons), dict(role_counts), len(mb_edges))

    return mb_neurons, mb_edges


def mb_circuit_stats(mb_neurons, mb_edges):
    """Compute pathway-level statistics for the MB circuit.

    Parameters
    ----------
    mb_neurons : pd.DataFrame
        MB neurons with 'circuit_role' column.
    mb_edges : pd.DataFrame
        MB-internal edges.

    Returns
    -------
    dict
        Keys: 'neuron_counts', 'pathway_counts', 'pathway_synapses'
    """
    id_to_role = dict(zip(mb_neurons["root_id"], mb_neurons["circuit_role"]))

    neuron_counts = mb_neurons["circuit_role"].value_counts().to_dict()

    # Map edges to pathway (pre_role -> post_role)
    pre_roles = mb_edges["pre_pt_root_id"].map(id_to_role)
    post_roles = mb_edges["post_pt_root_id"].map(id_to_role)
    pathways = pre_roles + " -> " + post_roles

    pathway_counts = pathways.value_counts().to_dict()
    pathway_synapses = (
        mb_edges.groupby(pathways)["syn_count"].sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "neuron_counts": neuron_counts,
        "pathway_counts": pathway_counts,
        "pathway_synapses": pathway_synapses,
    }


def build_mb_circuit(annotations, edges, mode="class_aware", dt=0.1):
    """Build a simulation-ready Circuit for the mushroom body.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full FlyWire annotation table.
    edges : pd.DataFrame
        Processed edge table (with weights assigned).
    mode : str
        Cell model assignment mode: "class_aware" or "uniform".
    dt : float
        Simulation timestep (ms).

    Returns
    -------
    circuit : Circuit
        Simulation-ready circuit.
    mb_neurons : pd.DataFrame
        MB neuron annotations with circuit_role and model params.
    mb_edges : pd.DataFrame
        MB-internal edges.
    """
    from bravli.models import assign_cell_models
    from bravli.simulation import build_circuit

    mb_neurons, mb_edges = extract_mb_circuit(annotations, edges)

    # Assign cell models
    mb_neurons_with_models = assign_cell_models(mb_neurons, mode=mode)

    # Build circuit
    circuit = build_circuit(mb_neurons_with_models, mb_edges, dt=dt)

    LOG.info("MB circuit built: %s", circuit.summary().split('\n')[0])
    return circuit, mb_neurons_with_models, mb_edges


def neuron_groups(circuit, mb_neurons):
    """Map circuit roles to dense neuron indices.

    Parameters
    ----------
    circuit : Circuit
        MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame
        MB neuron annotations with 'circuit_role' column.

    Returns
    -------
    dict
        Role name -> np.ndarray of dense indices in the circuit.
    """
    groups = {}
    for role, group_df in mb_neurons.groupby("circuit_role"):
        ids = group_df["root_id"].values
        indices = [circuit.id_to_idx[rid] for rid in ids
                   if rid in circuit.id_to_idx]
        groups[role] = np.array(indices, dtype=np.int32)
    return groups


def simulate_odor_presentation(circuit, mb_neurons, duration_ms=500.0,
                                odor_fraction=0.1, pn_rate_hz=50.0,
                                pn_weight=68.75, n_trials=1, seed=42):
    """Simulate odor presentation to the MB circuit.

    An "odor" activates a random fraction of PNs with Poisson input.
    KCs receive convergent PN input through the connectome wiring.
    The question: does sparse KC activation emerge?

    Parameters
    ----------
    circuit : Circuit
        MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame
        MB neuron annotations with 'circuit_role' column.
    duration_ms : float
        Simulation duration (ms).
    odor_fraction : float
        Fraction of PNs activated (0-1). Typical odors activate ~10-30%.
    pn_rate_hz : float
        Firing rate of activated PNs (Hz). AL oscillation: ~20-50 Hz.
    pn_weight : float
        Weight of external Poisson spikes (mV).
    n_trials : int
        Number of trials with different random odor patterns.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Per-trial results, each containing:
        - 'result': SimulationResult
        - 'active_pns': indices of activated PNs
        - 'groups': role -> indices mapping
        - 'sparseness': KC population sparseness
        - 'kc_active_fraction': fraction of KCs active (>1 Hz)
    """
    from bravli.simulation import simulate, poisson_stimulus
    from bravli.simulation.analysis import (
        firing_rates, population_sparseness, active_fraction_by_group,
    )

    groups = neuron_groups(circuit, mb_neurons)
    pn_indices = groups.get("PN", np.array([], dtype=np.int32))

    if len(pn_indices) == 0:
        LOG.warning("No PN neurons in circuit — cannot simulate odor input. "
                    "Did you set include_afferents=True?")
        return []

    rng = np.random.RandomState(seed)
    n_active_pns = max(1, int(odor_fraction * len(pn_indices)))
    n_steps = int(duration_ms / 0.1)

    trials = []
    for trial in range(n_trials):
        # Select random subset of PNs for this "odor"
        active_pn_indices = rng.choice(pn_indices, size=n_active_pns,
                                        replace=False)

        # Build stimulus
        stim, protocol = poisson_stimulus(
            circuit.n_neurons, n_steps, active_pn_indices,
            rate_hz=pn_rate_hz, weight=pn_weight,
            seed=seed + trial,
        )

        # Simulate
        result = simulate(
            circuit, duration=duration_ms, dt=0.1, stimulus=stim,
            record_v=True,
            record_idx=list(range(min(20, circuit.n_neurons))),
        )

        # Analyze KC sparseness
        kc_indices = groups.get("KC", np.array([], dtype=np.int32))
        rates = firing_rates(result)
        kc_rates = rates[kc_indices] if len(kc_indices) > 0 else np.array([])
        sparseness = population_sparseness(kc_rates)
        group_activity = active_fraction_by_group(result, groups)

        kc_active = group_activity.get("KC", (0, 0, 0.0))

        LOG.info("Trial %d: KC sparseness=%.3f, KC active=%.1f%% (%d/%d), "
                 "mean KC rate=%.1f Hz",
                 trial, sparseness, kc_active[2] * 100,
                 kc_active[0], kc_active[1],
                 np.mean(kc_rates) if len(kc_rates) > 0 else 0.0)

        trials.append({
            "result": result,
            "active_pns": active_pn_indices,
            "groups": groups,
            "sparseness": sparseness,
            "kc_active_fraction": kc_active[2],
            "kc_mean_rate": float(np.mean(kc_rates)) if len(kc_rates) > 0 else 0.0,
            "group_activity": group_activity,
            "rates": rates,
        })

    return trials


def mb_analysis_report(trials, mb_neurons):
    """Print a structured analysis report for MB simulation trials.

    Parameters
    ----------
    trials : list of dict
        Output of simulate_odor_presentation.
    mb_neurons : pd.DataFrame
        MB neuron annotations.
    """
    lines = [
        "=" * 60,
        "MUSHROOM BODY MICROCIRCUIT — Simulation Report",
        "=" * 60,
        "",
    ]

    # Neuron counts
    role_counts = mb_neurons["circuit_role"].value_counts()
    lines.append("--- Circuit composition ---")
    for role, n in role_counts.items():
        lines.append(f"  {role:10s}: {n:,}")
    lines.append("")

    # Per-trial results
    for i, trial in enumerate(trials):
        lines.append(f"--- Trial {i} ---")
        lines.append(f"  Active PNs:         {len(trial['active_pns']):,}")
        lines.append(f"  KC sparseness (TR):  {trial['sparseness']:.4f}")
        lines.append(f"  KC active fraction:  {trial['kc_active_fraction']:.3f} "
                     f"({trial['kc_active_fraction']*100:.1f}%)")
        lines.append(f"  KC mean rate:        {trial['kc_mean_rate']:.1f} Hz")
        lines.append(f"  Total spikes:        {trial['result'].n_spikes:,}")
        lines.append("")

        # Per-group activity
        lines.append("  Group activity (>1 Hz):")
        for role, (active, total, frac) in trial["group_activity"].items():
            lines.append(f"    {role:10s}: {active:4d}/{total:4d} ({frac*100:5.1f}%)")
        lines.append("")

    # Summary across trials
    if len(trials) > 1:
        sparsenesses = [t["sparseness"] for t in trials]
        fractions = [t["kc_active_fraction"] for t in trials]
        lines.append("--- Summary across trials ---")
        lines.append(f"  KC sparseness:      {np.mean(sparsenesses):.4f} "
                     f"+/- {np.std(sparsenesses):.4f}")
        lines.append(f"  KC active fraction: {np.mean(fractions):.3f} "
                     f"+/- {np.std(fractions):.3f}")

    # Scientific interpretation
    mean_frac = np.mean([t["kc_active_fraction"] for t in trials])
    lines.append("")
    lines.append("--- Interpretation ---")
    if mean_frac < 0.05:
        lines.append("  KC activation is VERY SPARSE (<5%). May be too sparse —")
        lines.append("  check that PN→KC synapses are strong enough.")
    elif mean_frac < 0.15:
        lines.append("  KC activation is in the EXPERIMENTAL RANGE (5-15%).")
        lines.append("  Sparseness emerges from the wiring alone.")
    elif mean_frac < 0.30:
        lines.append("  KC activation is MODERATE (15-30%). Higher than expected.")
        lines.append("  APL feedback inhibition may need strengthening.")
    else:
        lines.append("  KC activation is DENSE (>30%). Not sparse.")
        lines.append("  Something is wrong — check E/I balance, weights, thresholds.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run MB exploration from the command line."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m bravli.explore.mushroom_body <annotations.tsv>")
        sys.exit(1)

    path = sys.argv[1]
    annotations = load_flywire_annotations(path)
    mb_summary_report(annotations)


if __name__ == "__main__":
    main()
