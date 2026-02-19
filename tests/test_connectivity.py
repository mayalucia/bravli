"""Tests for the connectivity module: edges, matrix, pathways."""

import numpy as np
import pandas as pd
import pytest

from bravli.connectivity.edges import (
    threshold_edges,
    assign_dominant_nt,
    aggregate_by_pair,
)
from bravli.connectivity.matrix import (
    neuropil_synapse_counts,
    neuropil_connectivity_matrix,
    neuropil_nt_matrices,
)
from bravli.connectivity.pathways import (
    pathway_stats,
    top_pathways,
    convergence_divergence,
    nt_pathway_breakdown,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic edge data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_edges():
    """A small synthetic edge list mimicking FlyWire format.

    3 neuropils (MB, AL, LH), 6 neurons.
    Neuron 1->2: 10 syn in MB (ACh), 5 syn in AL (ACh)
    Neuron 1->3: 3 syn in MB (GABA)
    Neuron 2->4: 8 syn in AL (ACh)
    Neuron 3->5: 2 syn in MB (GABA)
    Neuron 4->5: 6 syn in LH (glutamate)
    Neuron 5->6: 1 syn in LH (GABA) -- below threshold
    """
    return pd.DataFrame({
        "pre_pt_root_id":  [1, 1, 1, 2, 3, 4, 5],
        "post_pt_root_id": [2, 2, 3, 4, 5, 5, 6],
        "neuropil":        ["MB", "AL", "MB", "AL", "MB", "LH", "LH"],
        "syn_count":       [10,   5,   3,   8,   2,   6,   1],
        "gaba_avg":        [0.05, 0.03, 0.85, 0.04, 0.88, 0.10, 0.80],
        "ach_avg":         [0.90, 0.92, 0.05, 0.91, 0.05, 0.10, 0.05],
        "glut_avg":        [0.03, 0.03, 0.05, 0.03, 0.04, 0.75, 0.10],
        "oct_avg":         [0.005, 0.005, 0.02, 0.005, 0.01, 0.02, 0.02],
        "ser_avg":         [0.005, 0.005, 0.02, 0.005, 0.01, 0.02, 0.02],
        "da_avg":          [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    })


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestThresholdEdges:
    def test_default_threshold(self, sample_edges):
        result = threshold_edges(sample_edges, min_syn=5)
        assert len(result) == 4  # edges with syn >= 5: 10, 5, 8, 6
        assert all(result["syn_count"] >= 5)

    def test_threshold_one_keeps_all(self, sample_edges):
        result = threshold_edges(sample_edges, min_syn=1)
        assert len(result) == len(sample_edges)

    def test_high_threshold(self, sample_edges):
        result = threshold_edges(sample_edges, min_syn=100)
        assert len(result) == 0


class TestAssignNT:
    def test_assigns_dominant_nt(self, sample_edges):
        result = assign_dominant_nt(sample_edges)
        assert "dominant_nt" in result.columns
        assert "nt_sign" in result.columns

        # Neuron 1->2 in MB: ACh dominant (0.90)
        row_1_2_mb = result[
            (result["pre_pt_root_id"] == 1) &
            (result["post_pt_root_id"] == 2) &
            (result["neuropil"] == "MB")
        ]
        assert row_1_2_mb.iloc[0]["dominant_nt"] == "acetylcholine"
        assert row_1_2_mb.iloc[0]["nt_sign"] == "excitatory"

        # Neuron 1->3 in MB: GABA dominant (0.85)
        row_1_3 = result[
            (result["pre_pt_root_id"] == 1) &
            (result["post_pt_root_id"] == 3)
        ]
        assert row_1_3.iloc[0]["dominant_nt"] == "GABA"
        assert row_1_3.iloc[0]["nt_sign"] == "inhibitory"

    def test_handles_missing_nt_columns(self):
        df = pd.DataFrame({
            "pre_pt_root_id": [1],
            "post_pt_root_id": [2],
            "neuropil": ["MB"],
            "syn_count": [10],
        })
        result = assign_dominant_nt(df)
        assert result.iloc[0]["dominant_nt"] == "unknown"


class TestAggregate:
    def test_aggregates_across_neuropils(self, sample_edges):
        result = aggregate_by_pair(sample_edges)
        # Neuron 1->2 appears in MB (10 syn) and AL (5 syn) -> total 15
        pair = result[
            (result["pre_pt_root_id"] == 1) &
            (result["post_pt_root_id"] == 2)
        ]
        assert len(pair) == 1
        assert pair.iloc[0]["syn_count"] == 15

    def test_preserves_unique_pairs(self, sample_edges):
        result = aggregate_by_pair(sample_edges)
        # 7 edges collapse to 6 unique pairs (1->2 merges)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Matrix tests
# ---------------------------------------------------------------------------

class TestNeuropilSynapseCounts:
    def test_counts_per_neuropil(self, sample_edges):
        counts = neuropil_synapse_counts(sample_edges)
        assert counts["MB"] == 10 + 3 + 2  # three MB edges
        assert counts["AL"] == 5 + 8
        assert counts["LH"] == 6 + 1


class TestNeuropilConnectivityMatrix:
    def test_matrix_is_square(self, sample_edges):
        mat = neuropil_connectivity_matrix(sample_edges)
        assert mat.shape[0] == mat.shape[1]
        assert set(mat.index) == set(mat.columns)

    def test_matrix_has_all_neuropils(self, sample_edges):
        mat = neuropil_connectivity_matrix(sample_edges)
        assert "MB" in mat.index
        assert "AL" in mat.index
        assert "LH" in mat.index

    def test_matrix_values_are_nonnegative(self, sample_edges):
        mat = neuropil_connectivity_matrix(sample_edges)
        assert (mat.values >= 0).all()


class TestNeuropilNTMatrices:
    def test_returns_all_nt_types(self, sample_edges):
        nt_mats = neuropil_nt_matrices(sample_edges)
        assert "acetylcholine" in nt_mats
        assert "GABA" in nt_mats
        assert "glutamate" in nt_mats

    def test_nt_values_are_nonnegative(self, sample_edges):
        nt_mats = neuropil_nt_matrices(sample_edges)
        for nt_name, counts in nt_mats.items():
            assert (counts.values >= 0).all(), f"Negative values in {nt_name}"


# ---------------------------------------------------------------------------
# Pathway tests
# ---------------------------------------------------------------------------

class TestPathwayStats:
    def test_returns_all_neuropils(self, sample_edges):
        stats = pathway_stats(sample_edges)
        assert "MB" in stats.index
        assert "AL" in stats.index
        assert "LH" in stats.index

    def test_correct_edge_counts(self, sample_edges):
        stats = pathway_stats(sample_edges)
        assert stats.loc["MB", "n_edges"] == 3  # three edges in MB
        assert stats.loc["AL", "n_edges"] == 2
        assert stats.loc["LH", "n_edges"] == 2

    def test_correct_synapse_totals(self, sample_edges):
        stats = pathway_stats(sample_edges)
        assert stats.loc["MB", "total_synapses"] == 15
        assert stats.loc["AL", "total_synapses"] == 13
        assert stats.loc["LH", "total_synapses"] == 7


class TestTopPathways:
    def test_returns_n_strongest(self, sample_edges):
        top = top_pathways(sample_edges, n=3)
        assert len(top) == 3
        assert top.iloc[0]["syn_count"] == 10  # strongest edge

    def test_n_larger_than_edges(self, sample_edges):
        top = top_pathways(sample_edges, n=100)
        assert len(top) == len(sample_edges)


class TestConvergenceDivergence:
    def test_divergence(self, sample_edges):
        cd = convergence_divergence(sample_edges)
        div = cd["divergence"]
        # Neuron 1 sends to neurons 2 and 3 -> divergence = 2
        assert div[1] == 2

    def test_convergence(self, sample_edges):
        cd = convergence_divergence(sample_edges)
        conv = cd["convergence"]
        # Neuron 5 receives from neurons 3 and 4 -> convergence = 2
        assert conv[5] == 2


class TestNTPathwayBreakdown:
    def test_fractions_sum_to_approximately_one(self, sample_edges):
        breakdown = nt_pathway_breakdown(sample_edges)
        nt_cols = ["acetylcholine", "GABA", "glutamate",
                   "octopamine", "serotonin", "dopamine"]
        present = [c for c in nt_cols if c in breakdown.columns]
        row_sums = breakdown[present].sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_has_all_neuropils(self, sample_edges):
        breakdown = nt_pathway_breakdown(sample_edges)
        assert "MB" in breakdown.index
        assert "AL" in breakdown.index
