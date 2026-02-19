"""Load FlyWire annotation data and build the neuropil hierarchy.

Data source: flywire_annotations GitHub repository
  Supplemental_file1_neuron_annotations.tsv

This file is a static snapshot from the FlyWire 783 release,
published with Schlegel et al., Nature 2024. No authentication required.
"""

from pathlib import Path

import pandas as pd

from bravli.parcellation.parcellation import NeuropilRegion, FlyBrainParcellation
from bravli.utils import get_logger

LOG = get_logger("load_flywire")


# ---------------------------------------------------------------------------
# Neuropil hierarchy from annotations
# ---------------------------------------------------------------------------

# Known groupings of FlyWire super_classes into broad anatomical divisions.
# This is our minimal hierarchy until we integrate the full neuropil meshes.

ANATOMICAL_DIVISIONS = {
    "central_brain": {
        "description": "Central brain neuropils",
        "super_classes": ["central"],
    },
    "optic_lobe": {
        "description": "Optic lobe neuropils",
        "super_classes": ["optic"],
    },
    "sensory": {
        "description": "Sensory neurons",
        "super_classes": ["sensory"],
    },
    "motor_and_descending": {
        "description": "Motor, descending, and ascending neurons",
        "super_classes": ["ascending", "descending", "motor"],
    },
    "neuroendocrine": {
        "description": "Endocrine neurons",
        "super_classes": ["endocrine"],
    },
    "visual_projection": {
        "description": "Visual projection and centrifugal neurons",
        "super_classes": ["visual_projection", "visual_centrifugal"],
    },
}


def build_neuropil_hierarchy(annotations):
    """Build a NeuropilRegion tree from the annotation DataFrame.

    The tree has three levels:
      root → anatomical division → cell_class (within each super_class)

    Parameters
    ----------
    annotations : pd.DataFrame
        The FlyWire neuron annotation table with 'super_class' and
        'cell_class' columns.

    Returns
    -------
    NeuropilRegion
        Root of the hierarchy tree.
    """
    children = []

    for division_name, info in ANATOMICAL_DIVISIONS.items():
        super_classes = info["super_classes"]
        mask = annotations["super_class"].isin(super_classes)
        subset = annotations[mask]

        if subset.empty:
            continue

        # Build cell_class children within this division
        class_children = []
        if "cell_class" in annotations.columns:
            for cell_class in sorted(subset["cell_class"].dropna().unique()):
                class_children.append({
                    "name": cell_class,
                    "acronym": cell_class,
                    "description": f"Cell class '{cell_class}' in {division_name}",
                })

        children.append({
            "name": division_name,
            "acronym": division_name[:3].upper(),
            "description": info["description"],
            "children": class_children,
        })

    root = NeuropilRegion(
        name="fly_brain",
        acronym="FB",
        description="Drosophila melanogaster whole brain (FlyWire 783)",
        children=children,
    )

    LOG.info("Built neuropil hierarchy: %d divisions, %d total regions",
             len(children), len(root.collect_hierarchy()))
    return root


# ---------------------------------------------------------------------------
# Load the annotation TSV
# ---------------------------------------------------------------------------

# Columns we actually use (the full TSV has many more)
_CORE_COLUMNS = [
    "root_id",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemibrain_type",
    "top_nt",
    "top_nt_conf",
    "side",
    "flow",
]


def load_flywire_annotations(path, columns=None):
    """Load the FlyWire neuron annotation TSV.

    Parameters
    ----------
    path : str or Path
        Path to Supplemental_file1_neuron_annotations.tsv
    columns : list of str, optional
        Columns to load. If None, loads a curated subset of the most
        useful columns (to save memory — the full file has 30+ columns).

    Returns
    -------
    pd.DataFrame
        Neuron annotations indexed by root_id.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}\n"
            "Download from: https://github.com/flyconnectome/flywire_annotations"
        )

    LOG.info("Loading FlyWire annotations from %s", path)

    # Determine which columns to load
    use_columns = columns or _CORE_COLUMNS

    # Read only the columns we need (if they exist in the file)
    all_columns = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    valid_columns = [c for c in use_columns if c in all_columns]

    if not valid_columns:
        LOG.warning("None of the requested columns found. Loading all columns.")
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, sep="\t", usecols=valid_columns, low_memory=False)

    LOG.info("Loaded %d neurons with columns: %s",
             len(df), ", ".join(df.columns.tolist()))

    return df


def load_parcellation(annotation_path):
    """One-shot: load annotations and build a FlyBrainParcellation.

    Parameters
    ----------
    annotation_path : str or Path
        Path to the FlyWire annotation TSV.

    Returns
    -------
    FlyBrainParcellation
        Ready-to-use parcellation with annotations and hierarchy.
    """
    annotations = load_flywire_annotations(annotation_path)
    root = build_neuropil_hierarchy(annotations)
    parcellation = FlyBrainParcellation(root=root, annotations=annotations)
    LOG.info("Created %r", parcellation)
    return parcellation
