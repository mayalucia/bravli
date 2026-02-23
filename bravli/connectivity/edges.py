"""Edge list loading, filtering, and neurotransmitter annotation.

The FlyWire aggregated edge list contains one row per
(pre_neuron, post_neuron, neuropil) triple. This module loads it,
applies thresholds, and annotates with dominant neurotransmitter.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bravli.bench.dataset import evaluate_datasets
from bravli.utils import get_logger

LOG = get_logger("connectivity.edges")

# Re-export NT constants from their canonical location
from bravli.connectivity.neurotransmitters import NT_COLUMNS, NT_NAMES, NT_SIGN


def load_edges(path):
    """Load the FlyWire aggregated edge list from a Feather file.

    Parameters
    ----------
    path : str or Path
        Path to proofread_connections_783.feather

    Returns
    -------
    pd.DataFrame
        16.8M rows, columns: pre_pt_root_id, post_pt_root_id,
        neuropil, syn_count, gaba_avg, ach_avg, glut_avg,
        oct_avg, ser_avg, da_avg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Edge list not found: {path}\n"
            "Download from: https://zenodo.org/records/10676866/files/"
            "proofread_connections_783.feather"
        )

    LOG.info("Loading edge list from %s", path)
    df = pd.read_feather(path)
    LOG.info("Loaded %d edges across %d neuropils, %d unique neurons",
             len(df), df["neuropil"].nunique(),
             len(set(df["pre_pt_root_id"]) | set(df["post_pt_root_id"])))
    return df


def threshold_edges(edges, min_syn=5):
    """Keep only edges with at least min_syn synapses.

    The default threshold of 5 follows Dorkenwald et al. (2024), who
    validated this cutoff against known circuit motifs.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with 'syn_count' column.
    min_syn : int
        Minimum synapse count to retain an edge.

    Returns
    -------
    pd.DataFrame
        Filtered edge list.
    """
    n_before = len(edges)
    result = edges[edges["syn_count"] >= min_syn].copy()
    n_after = len(result)
    LOG.info("Threshold >= %d synapses: %d → %d edges (%.1f%% retained)",
             min_syn, n_before, n_after, 100 * n_after / n_before)
    return result


def assign_dominant_nt(edges):
    """Add a 'dominant_nt' column based on max NT probability.

    Also adds 'nt_sign' (excitatory / inhibitory / mixed / modulatory)
    based on the dominant neurotransmitter.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with NT probability columns.

    Returns
    -------
    pd.DataFrame
        Input with 'dominant_nt' and 'nt_sign' columns added.
    """
    result = edges.copy()
    present_nt_cols = [c for c in NT_COLUMNS if c in result.columns]
    if not present_nt_cols:
        LOG.warning("No NT probability columns found; skipping NT assignment")
        result["dominant_nt"] = "unknown"
        result["nt_sign"] = "unknown"
        return result

    nt_probs = result[present_nt_cols]
    dominant_col = nt_probs.idxmax(axis=1)
    result["dominant_nt"] = dominant_col.map(NT_NAMES)
    result["nt_sign"] = result["dominant_nt"].map(NT_SIGN)

    LOG.info("NT assignment: %s",
             dict(result["dominant_nt"].value_counts().head(6)))
    return result


@evaluate_datasets
def aggregate_by_pair(edges):
    """Collapse per-neuropil edges into per-(pre, post) totals.

    A single (pre, post) neuron pair may have synapses in multiple
    neuropils. This function sums syn_count and averages NT probabilities
    across neuropils to produce one row per neuron pair.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with per-neuropil rows.

    Returns
    -------
    pd.DataFrame
        One row per (pre, post) pair with total syn_count and
        weighted-average NT probabilities.
    """
    present_nt_cols = [c for c in NT_COLUMNS if c in edges.columns]

    # Weighted average: weight by syn_count within each neuropil
    agg = {"syn_count": "sum"}
    for col in present_nt_cols:
        edges[f"_{col}_weighted"] = edges[col] * edges["syn_count"]
        agg[f"_{col}_weighted"] = "sum"

    grouped = edges.groupby(
        ["pre_pt_root_id", "post_pt_root_id"], as_index=False
    ).agg(agg)

    # Recover weighted averages
    for col in present_nt_cols:
        grouped[col] = grouped[f"_{col}_weighted"] / grouped["syn_count"]
        grouped.drop(columns=[f"_{col}_weighted"], inplace=True)

    # Clean up temp columns from input
    for col in present_nt_cols:
        wt_col = f"_{col}_weighted"
        if wt_col in edges.columns:
            edges.drop(columns=[wt_col], inplace=True)

    LOG.info("Aggregated %d per-neuropil edges → %d neuron pairs",
             len(edges), len(grouped))
    return grouped
