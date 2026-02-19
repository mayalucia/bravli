"""Neuropil region hierarchy and fly brain parcellation.

Heritage: the recursive NeuropilRegion tree descends from BrainRegion
in explore_bba/build/parcellate/parcellate.py (Blue Brain Project).
Adapted for the fly brain: no voxel volumes, table-based parcellation.
"""

from dataclasses import dataclass, field
from typing import Any, List
from copy import deepcopy

import numpy as np
import pandas as pd

from bravli.bench.dataset import evaluate_datasets
from bravli.utils import get_logger

LOG = get_logger("parcellation")


# ---------------------------------------------------------------------------
# NeuropilRegion — a node in the region hierarchy
# ---------------------------------------------------------------------------

@dataclass
class NeuropilRegion:
    """A node representing a brain neuropil or a group of neuropils.

    Parameters
    ----------
    name : str
        Full name (e.g., "mushroom_body" or "MB_CA_R").
    acronym : str
        Short label, often the same as name for leaf neuropils.
    children : list
        Child regions. Each can be a dict (for recursive construction)
        or a NeuropilRegion.
    parent : NeuropilRegion, optional
        Back-reference to parent (set during tree construction).
    description : str, optional
        What this region does / what it contains.
    """

    name: str
    acronym: str
    children: list = field(default_factory=list)
    parent: Any = None
    description: str = ""

    def __post_init__(self):
        """Recursively build child NeuropilRegion instances."""
        raw_children = deepcopy(self.children)
        self.children = pd.Series({
            c["name"] if isinstance(c, dict) else c.name:
                NeuropilRegion(**c, parent=self) if isinstance(c, dict)
                else c
            for c in raw_children
        }) if raw_children else pd.Series(dtype=object)

    @property
    def is_leaf(self):
        """True if this region has no children."""
        return self.children.empty

    def collect_hierarchy(self):
        """Flatten the subtree rooted at this node into a pd.Series."""
        me = pd.Series([self], index=[self.name])
        if self.is_leaf:
            return me
        descendants = pd.concat([
            child.collect_hierarchy() for child in self.children
        ])
        return pd.concat([me, descendants])

    @property
    def leaves(self):
        """All leaf neuropil names in this subtree."""
        if self.is_leaf:
            return [self.name]
        result = []
        for child in self.children:
            result.extend(child.leaves)
        return result

    @property
    def hierarchy_path(self):
        """Path from this node up to root, as a list of names."""
        path = [self.name]
        node = self.parent
        while node is not None:
            path.append(node.name)
            node = node.parent
        return list(reversed(path))

    def find(self, name):
        """Find a region by name in this subtree. Returns None if not found."""
        if self.name == name or self.acronym == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found is not None:
                return found
        return None

    def __repr__(self):
        n_children = len(self.children)
        n_leaves = len(self.leaves)
        return (f"NeuropilRegion('{self.name}', "
                f"{n_children} children, {n_leaves} leaves)")


# ---------------------------------------------------------------------------
# FlyBrainParcellation — the top-level parcellation object
# ---------------------------------------------------------------------------

@dataclass
class FlyBrainParcellation:
    """A parcellation of the fly brain into neuropils.

    Combines a region hierarchy (tree of NeuropilRegion) with the neuron
    annotation table. Provides methods to query neurons by neuropil,
    list neuropils, and navigate the hierarchy.

    Parameters
    ----------
    root : NeuropilRegion
        Root of the region hierarchy tree.
    annotations : pd.DataFrame
        The FlyWire neuron annotation table. Expected columns include
        at least 'root_id' and a neuropil-related column.
    neuropil_column : str
        Which column in annotations holds the primary neuropil assignment.
        FlyWire annotations don't have a single 'neuropil' column — neurons
        span multiple neuropils. We use 'super_class' for broad grouping
        and provide neuropil queries via the supplementary neuropil data.
    """

    root: NeuropilRegion
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def regions(self):
        """All regions in the hierarchy as a flat pd.Series."""
        return self.root.collect_hierarchy()

    @property
    def neuropil_names(self):
        """Names of all leaf neuropils."""
        return self.root.leaves

    @property
    def n_neurons(self):
        """Total number of annotated neurons."""
        return len(self.annotations)

    def find(self, name):
        """Find a region by name. Raises KeyError if not found."""
        region = self.root.find(name)
        if region is None:
            raise KeyError(f"Region '{name}' not found in parcellation")
        return region

    def neurons_in_class(self, super_class):
        """Return annotations for neurons of a given super_class.

        Parameters
        ----------
        super_class : str
            One of the super_class values (e.g., 'central', 'optic',
            'sensory', 'ascending', 'descending', 'motor', 'endocrine',
            'visual_projection', 'visual_centrifugal').

        Returns
        -------
        pd.DataFrame
            Subset of annotations matching the super_class.
        """
        if "super_class" not in self.annotations.columns:
            raise ValueError("Annotations lack 'super_class' column")
        return self.annotations[
            self.annotations["super_class"] == super_class
        ]

    def neurons_of_type(self, cell_type):
        """Return annotations for a specific cell type."""
        if "cell_type" not in self.annotations.columns:
            raise ValueError("Annotations lack 'cell_type' column")
        return self.annotations[
            self.annotations["cell_type"] == cell_type
        ]

    def cell_type_counts(self):
        """Count neurons per cell_type across the whole brain."""
        if "cell_type" not in self.annotations.columns:
            raise ValueError("Annotations lack 'cell_type' column")
        return (self.annotations
                .groupby("cell_type")
                .size()
                .sort_values(ascending=False)
                .rename("neuron_count"))

    def super_class_counts(self):
        """Count neurons per super_class."""
        if "super_class" not in self.annotations.columns:
            raise ValueError("Annotations lack 'super_class' column")
        return (self.annotations
                .groupby("super_class")
                .size()
                .sort_values(ascending=False)
                .rename("neuron_count"))

    def neurotransmitter_profile(self):
        """Count neurons by predicted neurotransmitter type."""
        if "top_nt" not in self.annotations.columns:
            raise ValueError("Annotations lack 'top_nt' column")
        return (self.annotations
                .groupby("top_nt")
                .size()
                .sort_values(ascending=False)
                .rename("neuron_count"))

    def summary(self):
        """Print a brief summary of the parcellation."""
        lines = [
            f"FlyBrainParcellation",
            f"  Regions:   {len(self.regions)} ({len(self.neuropil_names)} leaves)",
            f"  Neurons:   {self.n_neurons:,}",
        ]
        if "super_class" in self.annotations.columns:
            sc = self.super_class_counts()
            lines.append(f"  Super-classes: {len(sc)}")
            for cls, n in sc.head(5).items():
                lines.append(f"    {cls}: {n:,}")
        if "top_nt" in self.annotations.columns:
            nt = self.neurotransmitter_profile()
            lines.append(f"  Neurotransmitters: {len(nt)}")
            for nt_name, n in nt.head(5).items():
                lines.append(f"    {nt_name}: {n:,}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"FlyBrainParcellation("
                f"{len(self.neuropil_names)} neuropils, "
                f"{self.n_neurons:,} neurons)")
