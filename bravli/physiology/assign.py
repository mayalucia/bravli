"""Assign synapse physiology models to connectome edges.

Bridges the parameter database (synapse_models.py) with the connectivity
edge list (Lesson 08). Given edges with dominant_nt annotations, assigns
biophysical parameters and computes effective synaptic weights.
"""

import pandas as pd

from bravli.bench.dataset import evaluate_datasets
from bravli.physiology.synapse_models import (
    SYNAPSE_DB, get_synapse_model, simple_sign,
)
from bravli.utils import get_logger

LOG = get_logger("physiology.assign")


@evaluate_datasets
def assign_synapse_models(edges, mode="biophysical", synapse_db=None):
    """Assign synapse model parameters to each edge.

    Adds columns for sign, tau_rise, tau_decay, e_rev, g_peak,
    and confidence based on the edge's dominant_nt.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with 'dominant_nt' column (from assign_dominant_nt).
    mode : str
        "biophysical" — per-NT kinetics and sign.
        "shiu" — uniform kinetics, sign-only distinction.

    Returns
    -------
    pd.DataFrame
        Input with added physiology columns.
    """
    if synapse_db is None:
        synapse_db = SYNAPSE_DB

    if "dominant_nt" not in edges.columns:
        raise ValueError(
            "Edge list lacks 'dominant_nt' column. "
            "Run assign_dominant_nt() from bravli.connectivity.edges first."
        )

    result = edges.copy()

    if mode == "shiu":
        result["sign"] = result["dominant_nt"].map(
            lambda nt: simple_sign(nt, mode="shiu")
        )
        result["tau_syn"] = 5.0       # Shiu uniform
        result["w_syn"] = 0.275       # mV per synapse
        result["model_confidence"] = "N/A (Shiu simplified)"
        LOG.info("Assigned Shiu-mode signs to %d edges", len(result))
        return result

    # Biophysical mode: per-NT parameters
    signs = []
    tau_rises = []
    tau_decays = []
    e_revs = []
    g_peaks = []
    confidences = []

    for nt in result["dominant_nt"]:
        if nt in synapse_db:
            model = synapse_db[nt]
            signs.append(model.sign)
            tau_rises.append(model.tau_rise)
            tau_decays.append(model.tau_decay)
            e_revs.append(model.e_rev)
            g_peaks.append(model.g_peak)
            confidences.append(model.confidence)
        else:
            signs.append(0)
            tau_rises.append(None)
            tau_decays.append(None)
            e_revs.append(None)
            g_peaks.append(None)
            confidences.append("UNKNOWN")

    result["sign"] = signs
    result["tau_rise"] = tau_rises
    result["tau_decay"] = tau_decays
    result["e_rev"] = e_revs
    result["g_peak"] = g_peaks
    result["model_confidence"] = confidences

    LOG.info("Assigned biophysical models to %d edges "
             "(HIGH: %d, MEDIUM: %d, LOW: %d)",
             len(result),
             sum(1 for c in confidences if c == "HIGH"),
             sum(1 for c in confidences if c == "MEDIUM"),
             sum(1 for c in confidences if c == "LOW"))
    return result


@evaluate_datasets
def compute_synaptic_weights(edges, mode="shiu"):
    """Compute effective synaptic weight for each edge.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with 'syn_count' and 'dominant_nt' columns.
    mode : str
        "shiu" — w = syn_count * sign * W_syn (0.275 mV).
        "conductance" — w = syn_count * sign * g_peak (nS).

    Returns
    -------
    pd.DataFrame
        Input with 'weight' column added.
    """
    result = edges.copy()

    if "sign" not in result.columns:
        result = assign_synapse_models(result, mode="biophysical")

    if mode == "shiu":
        W_SYN = 0.275  # mV per synapse (Shiu et al. 2024)
        shiu_signs = result["dominant_nt"].map(
            lambda nt: simple_sign(nt, mode="shiu")
        )
        result["weight"] = result["syn_count"] * shiu_signs * W_SYN
        LOG.info("Computed Shiu weights: range [%.2f, %.2f] mV",
                 result["weight"].min(), result["weight"].max())

    elif mode == "conductance":
        g = result["g_peak"].fillna(0.0)
        result["weight"] = result["syn_count"] * result["sign"] * g
        LOG.info("Computed conductance weights: range [%.2f, %.2f] nS",
                 result["weight"].min(), result["weight"].max())
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'shiu' or 'conductance'.")

    return result


@evaluate_datasets
def physiology_summary(edges):
    """Summarize physiology assignments across the edge list.

    Parameters
    ----------
    edges : pd.DataFrame
        Edge list with physiology columns (from assign_synapse_models).

    Returns
    -------
    pd.DataFrame
        Per-NT summary: count, total synapses, sign, confidence.
    """
    if "dominant_nt" not in edges.columns:
        raise ValueError("No 'dominant_nt' column in edges")

    rows = []
    for nt, group in edges.groupby("dominant_nt"):
        row = {
            "nt_type": nt,
            "n_edges": len(group),
            "total_synapses": group["syn_count"].sum(),
        }
        if nt in SYNAPSE_DB:
            model = SYNAPSE_DB[nt]
            row["sign"] = model.sign
            row["sign_label"] = model.sign_label
            row["tau_rise_ms"] = model.tau_rise
            row["tau_decay_ms"] = model.tau_decay
            row["e_rev_mV"] = model.e_rev
            row["g_peak_nS"] = model.g_peak
            row["confidence"] = model.confidence
        else:
            row["sign"] = 0
            row["sign_label"] = "unknown"
            row["confidence"] = "UNKNOWN"
        rows.append(row)

    return (pd.DataFrame(rows)
            .set_index("nt_type")
            .sort_values("total_synapses", ascending=False))
