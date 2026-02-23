"""Pathway-level connectivity analysis.

A pathway is a connection pattern between groups of neurons (by neuropil,
by cell type, or by neurotransmitter identity). This module provides
tools for summarizing and comparing pathways.
"""

import numpy as np
import pandas as pd

from bravli.bench.dataset import evaluate_datasets
from bravli.connectivity.neurotransmitters import NT_COLUMNS, NT_NAMES, NT_SIGN
from bravli.utils import get_logger

LOG = get_logger("connectivity.pathways")


@evaluate_datasets
def pathway_stats(edges):
    """Compute summary statistics for each neuropil's connectivity.

    For each neuropil, reports: number of edges, total synapses,
    unique pre/post neurons, mean/median synapse count per edge.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with neuropil, syn_count, pre/post root ID columns.

    Returns
    -------
    pd.DataFrame
        One row per neuropil with summary statistics.
    """
    stats = []
    for np_name, group in edges.groupby("neuropil"):
        stats.append({
            "neuropil": np_name,
            "n_edges": len(group),
            "total_synapses": group["syn_count"].sum(),
            "n_pre_neurons": group["pre_pt_root_id"].nunique(),
            "n_post_neurons": group["post_pt_root_id"].nunique(),
            "mean_syn_per_edge": group["syn_count"].mean(),
            "median_syn_per_edge": group["syn_count"].median(),
            "max_syn_per_edge": group["syn_count"].max(),
        })
    result = pd.DataFrame(stats).set_index("neuropil")
    return result.sort_values("total_synapses", ascending=False)


@evaluate_datasets
def top_pathways(edges, n=20):
    """Find the strongest neuron-to-neuron connections.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with pre/post root IDs and syn_count.
    n : int
        Number of top connections to return.

    Returns
    -------
    pd.DataFrame
        Top n edges by synapse count.
    """
    return (edges.nlargest(n, "syn_count")
            [["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"]]
            .reset_index(drop=True))


@evaluate_datasets
def convergence_divergence(edges):
    """Compute convergence and divergence per neuron.

    Divergence: how many post-synaptic partners does each neuron have?
    Convergence: how many pre-synaptic partners does each neuron have?

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list (ideally after thresholding).

    Returns
    -------
    dict with keys 'divergence' and 'convergence', each a pd.Series.
    """
    divergence = (edges.groupby("pre_pt_root_id")["post_pt_root_id"]
                  .nunique()
                  .rename("n_post_partners"))

    convergence = (edges.groupby("post_pt_root_id")["pre_pt_root_id"]
                   .nunique()
                   .rename("n_pre_partners"))

    LOG.info("Divergence: mean=%.1f, max=%d; Convergence: mean=%.1f, max=%d",
             divergence.mean(), divergence.max(),
             convergence.mean(), convergence.max())

    return {
        "divergence": divergence,
        "convergence": convergence,
    }


@evaluate_datasets
def nt_pathway_breakdown(edges):
    """Break down connectivity by dominant neurotransmitter per neuropil.

    For each neuropil, computes the fraction of synapses carried by each
    neurotransmitter.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with neuropil, syn_count, and NT probability columns.

    Returns
    -------
    pd.DataFrame
        Neuropils as rows, NT types as columns, values are synapse
        fractions (summing to ~1.0 per row).
    """
    present_nt_cols = [c for c in NT_COLUMNS if c in edges.columns]
    if not present_nt_cols:
        return pd.DataFrame()

    rows = []
    for np_name, group in edges.groupby("neuropil"):
        total_syn = group["syn_count"].sum()
        row = {"neuropil": np_name, "total_synapses": total_syn}
        for col in present_nt_cols:
            nt_name = NT_NAMES[col]
            weighted = (group[col] * group["syn_count"]).sum()
            row[nt_name] = weighted / total_syn if total_syn > 0 else 0.0
        rows.append(row)

    result = pd.DataFrame(rows).set_index("neuropil")
    return result.sort_values("total_synapses", ascending=False)
