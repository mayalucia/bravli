"""Tests for composition analysis functions."""

import pandas as pd
import pytest

from bravli.composition.composition import (
    count_by,
    cell_type_distribution,
    neurotransmitter_profile,
    compare_divisions,
    top_types,
)
from bravli.bench.dataset import Dataset


@pytest.fixture
def neurons():
    """Synthetic neuron annotation table."""
    return pd.DataFrame({
        "root_id": range(1, 11),
        "super_class": ["central"] * 6 + ["optic"] * 4,
        "cell_class": ["KC", "KC", "KC", "MBON", "MBON", "CX",
                        "Mi", "Mi", "Tm", "T4"],
        "cell_type": ["KC_a", "KC_a", "KC_b", "MBON_a", "MBON_b", "PFN",
                       "Mi1", "Mi1", "Tm1", "T4a"],
        "top_nt": ["acetylcholine"] * 3 + ["glutamate"] * 2 +
                  ["GABA"] + ["acetylcholine"] * 2 + ["GABA"] * 2,
        "top_nt_conf": [0.95, 0.90, 0.85, 0.92, 0.88, 0.91,
                        0.93, 0.87, 0.90, 0.86],
    })


class TestCountBy:
    def test_by_super_class(self, neurons):
        result = count_by(neurons, "super_class")
        assert result["central"] == 6
        assert result["optic"] == 4

    def test_by_cell_type(self, neurons):
        result = count_by(neurons, "cell_type")
        assert result["KC_a"] == 2

    def test_accepts_dataset(self, neurons):
        ds = Dataset(name="n", ftype="csv").with_data(neurons)
        result = count_by(ds, "super_class")
        assert result["central"] == 6


class TestCellTypeDistribution:
    def test_counts(self, neurons):
        dist = cell_type_distribution(neurons)
        assert "neuron_count" in dist.columns
        assert dist.loc["KC_a", "neuron_count"] == 2

    def test_normalized(self, neurons):
        dist = cell_type_distribution(neurons, normalize=True)
        assert "proportion" in dist.columns
        assert abs(dist["proportion"].sum() - 1.0) < 1e-10


class TestNeurotransmitterProfile:
    def test_profile(self, neurons):
        nt = neurotransmitter_profile(neurons)
        assert "acetylcholine" in nt.index
        assert nt.loc["acetylcholine", "neuron_count"] == 5
        assert "proportion" in nt.columns

    def test_confidence_filter(self, neurons):
        nt = neurotransmitter_profile(neurons, min_confidence=0.90)
        # Only neurons with conf >= 0.90 included
        total = nt["neuron_count"].sum()
        assert total < len(neurons)

    def test_mean_confidence(self, neurons):
        nt = neurotransmitter_profile(neurons)
        assert "mean_confidence" in nt.columns


class TestCompareDivisions:
    def test_crosstab(self, neurons):
        ct = compare_divisions(neurons, "top_nt")
        assert "central" in ct.columns
        assert "optic" in ct.columns
        # Each column sums to ~1.0 (proportions)
        for col in ct.columns:
            assert abs(ct[col].sum() - 1.0) < 1e-10


class TestTopTypes:
    def test_top_n(self, neurons):
        top = top_types(neurons, n=3)
        assert len(top) == 3
        assert top.index[0] == "KC_a"  # most abundant
