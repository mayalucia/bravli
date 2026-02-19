"""Mushroom body compartment mapping from Aso et al. 2014.

The MB is divided into ~15 compartments, each defined by the intersection of
KC axon projections, specific MBON dendrites, and specific DAN innervation.
Each compartment is an independent learning unit — a separate associative
memory channel with its own valence (approach vs. avoidance).

FlyWire uses numbered cell_types (MBON01–35, PAM01–15, PPL101–108) that
map onto the Aso et al. compartment naming. This module provides that
mapping and utilities to assign compartments to neurons and synapses.

References:
    Aso Y et al. (2014). eLife 3:e04577.
    Li F et al. (2020). eLife 9:e62576.
"""

import numpy as np
import pandas as pd

from bravli.utils import get_logger

LOG = get_logger("explore.mb_compartments")


# ---------------------------------------------------------------------------
# The Aso 2014 compartment table
# ---------------------------------------------------------------------------

MB_COMPARTMENTS = {
    # Gamma lobe — short-term memory, rapid acquisition
    "gamma1": {
        "mbon": ["MBON01"],
        "dan": ["PPL101"],
        "lobe": "gamma",
        "valence": "aversive",
    },
    "gamma2": {
        "mbon": ["MBON02"],
        "dan": ["PAM01"],
        "lobe": "gamma",
        "valence": "appetitive",
    },
    "gamma3": {
        "mbon": ["MBON03"],
        "dan": ["PAM02", "PAM09"],
        "lobe": "gamma",
        "valence": "appetitive",
    },
    "gamma4": {
        "mbon": ["MBON04"],
        "dan": ["PAM03", "PAM10", "PAM11"],
        "lobe": "gamma",
        "valence": "appetitive",
    },
    "gamma5": {
        "mbon": ["MBON05", "MBON06"],
        "dan": ["PAM12"],
        "lobe": "gamma",
        "valence": "appetitive",
    },
    # Alpha-prime / beta-prime lobes — intermediate-term memory
    "alpha1p": {
        "mbon": ["MBON11"],
        "dan": ["PAM07", "PAM08"],
        "lobe": "alpha_prime",
        "valence": "appetitive",
    },
    "alpha2p": {
        "mbon": ["MBON12", "MBON13"],
        "dan": ["PAM05", "PAM06"],
        "lobe": "alpha_prime",
        "valence": "appetitive",
    },
    "alpha3p": {
        "mbon": ["MBON14"],
        "dan": ["PAM04"],
        "lobe": "alpha_prime",
        "valence": "appetitive",
    },
    "beta1p": {
        "mbon": ["MBON07"],
        "dan": ["PAM15"],
        "lobe": "beta_prime",
        "valence": "aversive",
    },
    "beta2p": {
        "mbon": ["MBON09", "MBON10"],
        "dan": ["PAM08", "PAM13", "PAM14"],
        "lobe": "beta_prime",
        "valence": "appetitive",
    },
    # Alpha / beta lobes — long-term memory, consolidation
    "alpha1": {
        "mbon": ["MBON15", "MBON16"],
        "dan": ["PPL104", "PPL105"],
        "lobe": "alpha_beta",
        "valence": "aversive",
    },
    "alpha2": {
        "mbon": ["MBON17", "MBON18"],
        "dan": ["PPL103"],
        "lobe": "alpha_beta",
        "valence": "aversive",
    },
    "alpha3": {
        "mbon": ["MBON19"],
        "dan": ["PPL103", "PPL106"],
        "lobe": "alpha_beta",
        "valence": "aversive",
    },
    "beta1": {
        "mbon": ["MBON20", "MBON21"],
        "dan": ["PPL102"],
        "lobe": "alpha_beta",
        "valence": "aversive",
    },
    "beta2": {
        "mbon": ["MBON22", "MBON23"],
        "dan": ["PAM04", "PPL107"],
        "lobe": "alpha_beta",
        "valence": "mixed",
    },
}
"""Aso et al. 2014 compartment table.

Each entry maps a compartment name to its constituent cell types
(MBON types, DAN types), the KC lobe it belongs to, and its
behavioral valence (approach/avoidance).
"""


# Reverse lookups: cell_type -> compartment
_MBON_TO_COMPARTMENT = {}
_DAN_TO_COMPARTMENT = {}
for _comp, _info in MB_COMPARTMENTS.items():
    for _mbon in _info["mbon"]:
        _MBON_TO_COMPARTMENT[_mbon] = _comp
    for _dan in _info["dan"]:
        # DANs can innervate multiple compartments (e.g. PPL103 -> alpha2, alpha3)
        _DAN_TO_COMPARTMENT.setdefault(_dan, []).append(_comp)

# KC lobe assignment by cell_type prefix
KC_LOBE_MAP = {
    "KCg": "gamma",
    "KCab": "alpha_beta",
    "KCab-p": "alpha_beta",
    "KCapbp": "alpha_beta_prime",
    "KCa'b'": "alpha_beta_prime",
}


def assign_compartments(mb_neurons):
    """Add 'compartment' column to MB neuron annotations.

    Assignment rules:
    - MBONs: mapped by cell_type to specific compartment (MBON01 -> gamma1)
    - DANs: mapped by cell_type to compartment(s) (PAM01 -> gamma2)
      If a DAN innervates multiple compartments, uses the first.
    - KCs: assigned to lobe level (gamma, alpha_beta, alpha_beta_prime)
      based on cell_type prefix
    - PNs: assigned to "calyx" (they innervate the MB input region)
    - APL: assigned to "global" (it spans all compartments)
    - Other MBINs: assigned to "unknown"

    Parameters
    ----------
    mb_neurons : pd.DataFrame
        MB neuron annotations with 'circuit_role' and 'cell_type' columns.

    Returns
    -------
    pd.DataFrame
        Copy with added 'compartment' column.
    """
    df = mb_neurons.copy()
    compartments = []

    for _, row in df.iterrows():
        role = row.get("circuit_role", "")
        ct = str(row.get("cell_type", ""))

        if role == "MBON":
            compartments.append(_MBON_TO_COMPARTMENT.get(ct, "unknown"))
        elif role == "DAN":
            comps = _DAN_TO_COMPARTMENT.get(ct, [])
            compartments.append(comps[0] if comps else "unknown")
        elif role == "KC":
            lobe = _kc_lobe(ct)
            compartments.append(lobe if lobe else "unknown")
        elif role == "PN":
            compartments.append("calyx")
        elif role == "APL":
            compartments.append("global")
        else:
            compartments.append("unknown")

    df["compartment"] = compartments
    return df


def _kc_lobe(cell_type):
    """Determine KC lobe from cell_type string."""
    ct = str(cell_type)
    # Try exact matches first, then prefix matches (longest first)
    for prefix in sorted(KC_LOBE_MAP.keys(), key=len, reverse=True):
        if ct.startswith(prefix):
            return KC_LOBE_MAP[prefix]
    return None


def build_compartment_index(circuit, mb_neurons):
    """Build an index mapping compartments to neuron and synapse indices.

    For each compartment, identifies:
    - KC indices (neurons in the corresponding lobe)
    - MBON indices (neurons assigned to this compartment)
    - DAN indices (neurons assigned to this compartment)
    - KC->MBON synapse mask (boolean over circuit.weights)

    Parameters
    ----------
    circuit : Circuit
        MB circuit (from build_mb_circuit).
    mb_neurons : pd.DataFrame
        MB neuron annotations with 'circuit_role' column.

    Returns
    -------
    dict
        compartment_name -> {
            'kc_indices': np.ndarray,
            'mbon_indices': np.ndarray,
            'dan_indices': np.ndarray,
            'kc_mbon_syn_mask': np.ndarray (bool, n_synapses),
        }
    """
    from bravli.explore.mushroom_body import neuron_groups

    # Assign compartments
    mb_with_comp = assign_compartments(mb_neurons)

    # Build neuron groups by role
    groups = neuron_groups(circuit, mb_neurons)
    kc_set = set(groups.get("KC", []).tolist())
    mbon_set = set(groups.get("MBON", []).tolist())

    # Build root_id -> dense index mapping
    id_to_idx = circuit.id_to_idx

    index = {}
    for comp, info in MB_COMPARTMENTS.items():
        lobe = info["lobe"]

        # KCs in this lobe
        kc_in_lobe = mb_with_comp[
            (mb_with_comp["circuit_role"] == "KC") &
            (mb_with_comp["compartment"] == lobe)
        ]
        kc_indices = np.array([
            id_to_idx[rid] for rid in kc_in_lobe["root_id"].values
            if rid in id_to_idx
        ], dtype=np.int32)

        # MBONs in this compartment
        mbon_in_comp = mb_with_comp[
            (mb_with_comp["circuit_role"] == "MBON") &
            (mb_with_comp["compartment"] == comp)
        ]
        mbon_indices = np.array([
            id_to_idx[rid] for rid in mbon_in_comp["root_id"].values
            if rid in id_to_idx
        ], dtype=np.int32)

        # DANs in this compartment
        dan_in_comp = mb_with_comp[
            (mb_with_comp["circuit_role"] == "DAN") &
            (mb_with_comp["compartment"] == comp)
        ]
        dan_indices = np.array([
            id_to_idx[rid] for rid in dan_in_comp["root_id"].values
            if rid in id_to_idx
        ], dtype=np.int32)

        # KC->MBON synapse mask: pre is KC in this lobe, post is MBON in this compartment
        kc_idx_set = set(kc_indices.tolist())
        mbon_idx_set = set(mbon_indices.tolist())
        kc_mbon_mask = np.array([
            int(circuit.pre_idx[i]) in kc_idx_set and
            int(circuit.post_idx[i]) in mbon_idx_set
            for i in range(len(circuit.weights))
        ], dtype=bool)

        index[comp] = {
            "kc_indices": kc_indices,
            "mbon_indices": mbon_indices,
            "dan_indices": dan_indices,
            "kc_mbon_syn_mask": kc_mbon_mask,
            "lobe": lobe,
            "valence": info["valence"],
        }

        n_syn = kc_mbon_mask.sum()
        if n_syn > 0:
            LOG.debug("Compartment %s: %d KCs, %d MBONs, %d DANs, %d KC->MBON synapses",
                      comp, len(kc_indices), len(mbon_indices),
                      len(dan_indices), n_syn)

    # Summary
    total_kc_mbon = sum(v["kc_mbon_syn_mask"].sum() for v in index.values())
    LOG.info("Compartment index: %d compartments, %d total KC->MBON synapses",
             len(index), total_kc_mbon)

    return index


def compartment_summary(index):
    """Print a summary of the compartment index.

    Parameters
    ----------
    index : dict
        Output of build_compartment_index().

    Returns
    -------
    str
        Formatted summary.
    """
    lines = [
        "=" * 60,
        "MB Compartment Index",
        "=" * 60,
        "",
        f"{'Compartment':12s} {'Lobe':14s} {'Valence':12s} "
        f"{'KCs':>6s} {'MBONs':>6s} {'DANs':>6s} {'KC→MBON':>8s}",
        "-" * 70,
    ]
    for comp in MB_COMPARTMENTS:
        info = index.get(comp, {})
        lines.append(
            f"{comp:12s} {info.get('lobe', '?'):14s} {info.get('valence', '?'):12s} "
            f"{len(info.get('kc_indices', [])):6d} "
            f"{len(info.get('mbon_indices', [])):6d} "
            f"{len(info.get('dan_indices', [])):6d} "
            f"{info.get('kc_mbon_syn_mask', np.array([])).sum():8d}"
        )
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report
