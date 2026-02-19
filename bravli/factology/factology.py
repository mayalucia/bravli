"""Factology: structured factsheets for brain regions.

Heritage: the Factology ABC pattern comes from circuit-factology (BBP).
NeuropilFacts is the fly-brain-specific concrete implementation.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd

from bravli.factology.fact import Fact, fact, structural, connectomic
from bravli.utils import get_logger

LOG = get_logger("factology")


class Factology(ABC):
    """Abstract base: a collection of Facts about a subject.

    Subclasses define measurement methods decorated with @fact, @structural,
    or @connectomic. The .collect() method gathers all defined facts.
    """

    def __init__(self, annotations, target=None, helper=None):
        """
        Parameters
        ----------
        annotations : pd.DataFrame
            Neuron annotation data for the subject.
        target : str, optional
            Name of the target (e.g., neuropil name).
        helper : object, optional
            Backend providing @interface implementations.
        """
        self.annotations = annotations
        self.target = target
        self._helper = helper
        self._structural_facts = []
        self._connectomic_facts = []

    @classmethod
    def fact_methods(cls):
        """List all methods decorated with @fact."""
        return [name for name in dir(cls)
                if hasattr(getattr(cls, name, None), '__defines_a_fact__')]

    def collect(self, mode="prod"):
        """Collect all defined facts.

        Parameters
        ----------
        mode : str
            "prod" raises on errors; "dev" tolerates and logs them.

        Returns
        -------
        list of Fact
        """
        results = []
        for method_name in self.fact_methods():
            try:
                value = getattr(self, method_name)
                if callable(value):
                    value = value()
                results.append(value)
            except Exception as e:
                if mode == "prod":
                    raise
                LOG.warning("Skipping fact '%s': %s", method_name, e)
                results.append(None)
        return [r for r in results if r is not None]

    def collect_dicts(self, mode="prod"):
        """Collect facts as a list of dicts (for JSON/DataFrame export)."""
        return [f.to_dict() for f in self.collect(mode=mode)]

    def to_dataframe(self, mode="prod"):
        """Collect facts into a DataFrame."""
        return pd.DataFrame(self.collect_dicts(mode=mode))


class NeuropilFacts(Factology):
    """Factsheet for a fly brain neuropil or cell class.

    Given a subset of the FlyWire annotation table (e.g., all neurons
    in a particular super_class or cell_class), computes a standard set
    of structural facts.
    """

    @structural
    @fact("Neuron count", "neurons")
    def neuron_count(self):
        """Total number of annotated neurons."""
        return len(self.annotations)

    @structural
    @fact("Cell type count", "types")
    def cell_type_count(self):
        """Number of distinct cell types."""
        if "cell_type" not in self.annotations.columns:
            return 0
        return self.annotations["cell_type"].nunique()

    @structural
    @fact("Top cell types", None)
    def top_cell_types(self):
        """Five most abundant cell types (name: count)."""
        if "cell_type" not in self.annotations.columns:
            return {}
        counts = self.annotations["cell_type"].value_counts().head(5)
        return dict(counts)

    @structural
    @fact("Neurotransmitter breakdown", None)
    def neurotransmitter_breakdown(self):
        """Neuron counts by predicted neurotransmitter."""
        if "top_nt" not in self.annotations.columns:
            return {}
        counts = self.annotations["top_nt"].value_counts()
        return dict(counts)

    @structural
    @fact("Dominant neurotransmitter", None)
    def dominant_neurotransmitter(self):
        """The most common predicted neurotransmitter."""
        if "top_nt" not in self.annotations.columns:
            return "unknown"
        return self.annotations["top_nt"].value_counts().index[0]

    @structural
    @fact("Hemisphere balance", None)
    def hemisphere_balance(self):
        """Neuron counts by hemisphere (left vs right)."""
        if "side" not in self.annotations.columns:
            return {}
        counts = self.annotations["side"].value_counts()
        return dict(counts)
