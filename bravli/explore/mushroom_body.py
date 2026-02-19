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
"""Cell classes that form the mushroom body circuit.

- Kenyon_Cell: ~5,200 intrinsic neurons forming the sparse odor code
- MBON: ~100 mushroom body output neurons driving behavior
- MBIN: mushroom body input neurons (non-DAN modulatory)
- DAN: ~330 dopaminergic neurons carrying reinforcement signals
"""

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
