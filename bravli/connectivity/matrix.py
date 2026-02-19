"""Neuropil-level connectivity matrices.

Aggregates the edge list into 79x79 neuropil connectivity matrices,
optionally stratified by neurotransmitter type.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from bravli.bench.dataset import evaluate_datasets
from bravli.utils import get_logger

LOG = get_logger("connectivity.matrix")


@evaluate_datasets
def neuropil_synapse_counts(edges):
    """Total synapse count per neuropil.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with 'neuropil' and 'syn_count' columns.

    Returns
    -------
    pd.Series
        Synapse counts indexed by neuropil name, sorted descending.
    """
    counts = (edges.groupby("neuropil")["syn_count"]
              .sum()
              .sort_values(ascending=False))
    counts.name = "total_synapses"
    return counts


@evaluate_datasets
def neuropil_connectivity_matrix(edges, annotations=None):
    """Build a neuropil-to-neuropil connectivity matrix.

    Each edge in the edge list has a 'neuropil' column indicating where
    the synapses are located. To build a neuropil x neuropil matrix, we
    need to know which neuropil the pre and post neurons "belong to".

    Strategy: Use the neuropil where the most presynaptic outputs of a
    neuron fall as its "home neuropil" (the neuropil where it contributes
    the most synapses as a presynaptic partner).

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with neuropil, pre_pt_root_id, post_pt_root_id,
        syn_count columns.
    annotations : pd.DataFrame, optional
        Neuron annotations. Not used in this implementation but reserved
        for future cell-type-aware matrices.

    Returns
    -------
    pd.DataFrame
        Square DataFrame with neuropil names as both index and columns.
        Values are total synapse counts from pre-neuropil to post-neuropil.
    """
    # Determine each neuron's "home neuropil" = where it sends the most synapses
    pre_home = (edges.groupby(["pre_pt_root_id", "neuropil"])["syn_count"]
                .sum()
                .reset_index())
    pre_home = (pre_home.sort_values("syn_count", ascending=False)
                .drop_duplicates("pre_pt_root_id")
                .set_index("pre_pt_root_id")["neuropil"]
                .rename("pre_neuropil"))

    post_home = (edges.groupby(["post_pt_root_id", "neuropil"])["syn_count"]
                 .sum()
                 .reset_index())
    post_home = (post_home.sort_values("syn_count", ascending=False)
                 .drop_duplicates("post_pt_root_id")
                 .set_index("post_pt_root_id")["neuropil"]
                 .rename("post_neuropil"))

    # Join home neuropils onto edges
    enriched = edges.copy()
    enriched = enriched.join(pre_home, on="pre_pt_root_id")
    enriched = enriched.join(post_home, on="post_pt_root_id")

    # Pivot to matrix
    matrix = (enriched.groupby(["pre_neuropil", "post_neuropil"])["syn_count"]
              .sum()
              .unstack(fill_value=0))

    # Ensure square and sorted
    all_nps = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=all_nps, columns=all_nps, fill_value=0)

    LOG.info("Built %d x %d neuropil connectivity matrix, "
             "total synapses: %s",
             matrix.shape[0], matrix.shape[1],
             f"{matrix.values.sum():,.0f}")
    return matrix


@evaluate_datasets
def neuropil_nt_matrices(edges):
    """Build per-NT neuropil synapse count matrices.

    For each of the 6 neurotransmitters, produces a Series of synapse
    counts per neuropil, weighted by NT probability.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with neuropil, syn_count, and NT probability columns.

    Returns
    -------
    dict of str -> pd.Series
        Keys are NT names, values are per-neuropil weighted synapse counts.
    """
    from bravli.connectivity.edges import NT_COLUMNS, NT_NAMES

    result = {}
    present_nt_cols = [c for c in NT_COLUMNS if c in edges.columns]

    for col in present_nt_cols:
        nt_name = NT_NAMES[col]
        weighted = edges["syn_count"] * edges[col]
        counts = (pd.DataFrame({"neuropil": edges["neuropil"], "weighted": weighted})
                  .groupby("neuropil")["weighted"]
                  .sum()
                  .sort_values(ascending=False))
        counts.name = f"{nt_name}_weighted_synapses"
        result[nt_name] = counts

    LOG.info("Built NT matrices for: %s", list(result.keys()))
    return result
