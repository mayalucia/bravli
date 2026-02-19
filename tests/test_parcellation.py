"""Tests for parcellation: NeuropilRegion tree and FlyBrainParcellation."""

import pandas as pd
import pytest

from bravli.parcellation.parcellation import NeuropilRegion, FlyBrainParcellation
from bravli.parcellation.load_flywire import build_neuropil_hierarchy


# ---------------------------------------------------------------------------
# Fixtures: synthetic annotation data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """A small synthetic annotation DataFrame mimicking FlyWire format."""
    return pd.DataFrame({
        "root_id": range(1, 21),
        "super_class": (
            ["central"] * 8 +
            ["optic"] * 6 +
            ["sensory"] * 3 +
            ["descending"] * 2 +
            ["endocrine"] * 1
        ),
        "cell_class": (
            ["MBIN", "MBIN", "KC", "KC", "KC", "MBON", "MBON", "CX"] +
            ["Mi", "Mi", "Tm", "Tm", "T4", "T5"] +
            ["ORN", "ORN", "ORN"] +
            ["DN", "DN"] +
            ["NS"]
        ),
        "cell_sub_class": [""] * 20,
        "cell_type": (
            ["MBIN_a", "MBIN_b", "KC_a", "KC_a", "KC_b",
             "MBON_a", "MBON_a", "PFN"] +
            ["Mi1", "Mi1", "Tm1", "Tm2", "T4a", "T5a"] +
            ["ORN_a", "ORN_a", "ORN_b"] +
            ["DN_a", "DN_b"] +
            ["NS_a"]
        ),
        "top_nt": (
            ["acetylcholine"] * 5 +
            ["glutamate"] * 2 +
            ["GABA"] * 1 +
            ["acetylcholine"] * 4 +
            ["GABA"] * 2 +
            ["acetylcholine"] * 3 +
            ["acetylcholine"] * 2 +
            ["unknown"] * 1
        ),
        "top_nt_conf": [0.9] * 20,
        "side": ["right"] * 10 + ["left"] * 10,
        "flow": ["intrinsic"] * 14 + ["sensory"] * 3 + ["efferent"] * 3,
    })


@pytest.fixture
def sample_parcellation(sample_annotations):
    """A FlyBrainParcellation built from sample data."""
    root = build_neuropil_hierarchy(sample_annotations)
    return FlyBrainParcellation(root=root, annotations=sample_annotations)


# ---------------------------------------------------------------------------
# NeuropilRegion tree
# ---------------------------------------------------------------------------

class TestNeuropilRegion:

    def test_leaf_node(self):
        leaf = NeuropilRegion(name="MB_CA_R", acronym="MB_CA_R")
        assert leaf.is_leaf
        assert leaf.leaves == ["MB_CA_R"]

    def test_tree_construction(self):
        root = NeuropilRegion(
            name="brain",
            acronym="BR",
            children=[
                {"name": "central", "acronym": "CB", "children": [
                    {"name": "MB", "acronym": "MB"},
                    {"name": "AL", "acronym": "AL"},
                ]},
                {"name": "optic", "acronym": "OL", "children": [
                    {"name": "ME", "acronym": "ME"},
                ]},
            ],
        )
        assert not root.is_leaf
        assert len(root.children) == 2
        assert set(root.leaves) == {"MB", "AL", "ME"}

    def test_find(self):
        root = NeuropilRegion(
            name="brain", acronym="BR",
            children=[
                {"name": "central", "acronym": "CB", "children": [
                    {"name": "MB", "acronym": "MB"},
                ]},
            ],
        )
        assert root.find("MB") is not None
        assert root.find("MB").name == "MB"
        assert root.find("nonexistent") is None

    def test_hierarchy_path(self):
        root = NeuropilRegion(
            name="brain", acronym="BR",
            children=[
                {"name": "central", "acronym": "CB", "children": [
                    {"name": "MB", "acronym": "MB"},
                ]},
            ],
        )
        mb = root.find("MB")
        assert mb.hierarchy_path == ["brain", "central", "MB"]

    def test_collect_hierarchy(self):
        root = NeuropilRegion(
            name="brain", acronym="BR",
            children=[
                {"name": "A", "acronym": "A"},
                {"name": "B", "acronym": "B", "children": [
                    {"name": "B1", "acronym": "B1"},
                ]},
            ],
        )
        flat = root.collect_hierarchy()
        assert "brain" in flat.index
        assert "B1" in flat.index
        assert len(flat) == 4  # brain, A, B, B1


# ---------------------------------------------------------------------------
# Hierarchy from annotations
# ---------------------------------------------------------------------------

class TestBuildHierarchy:

    def test_builds_from_annotations(self, sample_annotations):
        root = build_neuropil_hierarchy(sample_annotations)
        assert root.name == "fly_brain"
        assert not root.is_leaf
        # Should have divisions for: central_brain, optic_lobe, sensory,
        # motor_and_descending, neuroendocrine
        assert len(root.children) >= 4

    def test_central_brain_has_cell_classes(self, sample_annotations):
        root = build_neuropil_hierarchy(sample_annotations)
        cb = root.find("central_brain")
        assert cb is not None
        # Should contain cell classes: MBIN, KC, MBON, CX
        leaves = cb.leaves
        assert "KC" in leaves
        assert "MBON" in leaves


# ---------------------------------------------------------------------------
# FlyBrainParcellation
# ---------------------------------------------------------------------------

class TestFlyBrainParcellation:

    def test_n_neurons(self, sample_parcellation):
        assert sample_parcellation.n_neurons == 20

    def test_neuropil_names(self, sample_parcellation):
        names = sample_parcellation.neuropil_names
        assert len(names) > 0
        assert "KC" in names

    def test_super_class_counts(self, sample_parcellation):
        counts = sample_parcellation.super_class_counts()
        assert counts["central"] == 8
        assert counts["optic"] == 6

    def test_cell_type_counts(self, sample_parcellation):
        counts = sample_parcellation.cell_type_counts()
        assert counts["KC_a"] == 2

    def test_neurotransmitter_profile(self, sample_parcellation):
        nt = sample_parcellation.neurotransmitter_profile()
        assert "acetylcholine" in nt.index
        assert "GABA" in nt.index

    def test_neurons_in_class(self, sample_parcellation):
        central = sample_parcellation.neurons_in_class("central")
        assert len(central) == 8

    def test_neurons_of_type(self, sample_parcellation):
        kc_a = sample_parcellation.neurons_of_type("KC_a")
        assert len(kc_a) == 2

    def test_find_raises_for_missing(self, sample_parcellation):
        with pytest.raises(KeyError):
            sample_parcellation.find("nonexistent_region")

    def test_summary(self, sample_parcellation):
        text = sample_parcellation.summary()
        assert "FlyBrainParcellation" in text
        assert "20" in text  # neuron count

    def test_repr(self, sample_parcellation):
        r = repr(sample_parcellation)
        assert "neuropils" in r
        assert "20" in r
