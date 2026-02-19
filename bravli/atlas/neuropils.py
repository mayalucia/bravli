"""Load and query the 78 FlyWire neuropil boundary meshes.

Data source: PLY files bundled with fafbseg, originally from the Ito et al.
(2014) neuropil nomenclature, transformed to FlyWire/FAFB14.1 coordinates.
No authentication required.
"""

from typing import Dict, List, Optional, Union
from functools import lru_cache

import numpy as np

from bravli.utils import get_logger

LOG = get_logger("atlas.neuropils")

try:
    from fafbseg import flywire as _fw
    HAS_FAFBSEG = True
except ImportError:
    HAS_FAFBSEG = False


# ---------------------------------------------------------------------------
# Semantic grouping of the 78 neuropils
# ---------------------------------------------------------------------------

NEUROPIL_GROUPS: Dict[str, List[str]] = {
    "mushroom_body": [
        "MB_CA_R", "MB_CA_L", "MB_PED_R", "MB_PED_L",
        "MB_VL_R", "MB_VL_L", "MB_ML_R", "MB_ML_L",
    ],
    "antennal_lobe": ["AL_R", "AL_L"],
    "lateral_horn": ["LH_R", "LH_L"],
    "central_complex": ["EB", "FB", "PB", "NO"],
    "optic_medulla": ["ME_R", "ME_L"],
    "optic_lobula": ["LO_R", "LO_L"],
    "optic_lobula_plate": ["LOP_R", "LOP_L"],
    "optic_lamina": ["LA_R", "LA_L"],
    "optic_accessory_medulla": ["AME_R", "AME_L"],
    "superior_protocerebrum": [
        "SMP_R", "SMP_L", "SIP_R", "SIP_L", "SLP_R", "SLP_L",
    ],
    "ventrolateral_protocerebrum": [
        "AVLP_R", "AVLP_L", "PVLP_R", "PVLP_L",
    ],
    "lateral_accessory_lobe": ["LAL_R", "LAL_L"],
    "bulb": ["BU_R", "BU_L"],
    "anterior_optic_tubercle": ["AOTU_R", "AOTU_L"],
    "gnathal_ganglion": ["GNG"],
    "other_midline": ["SAD", "PRW", "OCG"],
}
"""Semantic grouping of neuropil names by brain region."""


def _require_fafbseg():
    if not HAS_FAFBSEG:
        raise ImportError(
            "fafbseg is required for neuropil meshes. "
            "Install with: pip install fafbseg"
        )


def list_neuropils():
    """Return sorted list of all 78 neuropil names.

    Returns
    -------
    list of str
    """
    _require_fafbseg()
    return sorted(_fw.get_neuropil_volumes(None))


@lru_cache(maxsize=128)
def load_neuropil(name: str):
    """Load a single neuropil mesh by name.

    Parameters
    ----------
    name : str
        Neuropil name (e.g., 'MB_CA_R', 'AL_L', 'EB').

    Returns
    -------
    navis.Volume
        Mesh with .vertices (N,3) and .faces (M,3) in nanometres.
    """
    _require_fafbseg()
    vol = _fw.get_neuropil_volumes(name)
    LOG.debug("Loaded %s: %d vertices, %d faces",
              name, len(vol.vertices), len(vol.faces))
    return vol


def load_neuropils(
    names: Optional[Union[str, List[str]]] = None,
    group: Optional[str] = None,
) -> Dict[str, object]:
    """Load multiple neuropil meshes.

    Parameters
    ----------
    names : str or list of str, optional
        Specific neuropil names to load.  If None, loads all 78.
    group : str, optional
        A group name from NEUROPIL_GROUPS (e.g., 'mushroom_body').
        Overrides `names` if provided.

    Returns
    -------
    dict
        {neuropil_name: navis.Volume}
    """
    if group is not None:
        if group not in NEUROPIL_GROUPS:
            raise KeyError(
                f"Unknown group '{group}'. "
                f"Available: {sorted(NEUROPIL_GROUPS.keys())}"
            )
        names = NEUROPIL_GROUPS[group]
    elif names is None:
        names = list_neuropils()
    elif isinstance(names, str):
        names = [names]

    LOG.info("Loading %d neuropil meshes", len(names))
    return {name: load_neuropil(name) for name in names}
