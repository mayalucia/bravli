"""Tests for the Factology system."""

import pandas as pd
import pytest

from bravli.factology.fact import Fact, fact, structural, connectomic, interface
from bravli.factology.factology import Factology, NeuropilFacts


@pytest.fixture
def mushroom_body_neurons():
    """Synthetic mushroom body annotation data."""
    return pd.DataFrame({
        "root_id": range(1, 16),
        "super_class": ["central"] * 15,
        "cell_class": ["KC"] * 10 + ["MBON"] * 3 + ["MBIN"] * 2,
        "cell_type": (
            ["KC_alpha"] * 4 + ["KC_beta"] * 3 + ["KC_gamma"] * 3 +
            ["MBON_01"] * 2 + ["MBON_05"] * 1 +
            ["MBIN_a"] + ["MBIN_b"]
        ),
        "top_nt": (
            ["acetylcholine"] * 10 +
            ["glutamate"] * 3 +
            ["dopamine"] * 2
        ),
        "top_nt_conf": [0.92] * 15,
        "side": ["right"] * 8 + ["left"] * 7,
    })


class TestFact:
    def test_fact_creation(self):
        f = Fact("n", "Count", "How many", "neurons", 42)
        assert f.value == 42
        assert f.unit == "neurons"

    def test_fact_str(self):
        f = Fact("n", "Count", "How many", "neurons", 42)
        assert "42" in str(f)
        assert "neurons" in str(f)

    def test_fact_to_dict(self):
        f = Fact("n", "Count", "How many", "neurons", 42)
        d = f.to_dict()
        assert d["label"] == "n"
        assert d["value"] == 42


class TestFactDecorator:
    def test_fact_wraps_return_value(self):
        class MyFacts:
            @fact("Test", "units")
            def measure(self):
                """A test measurement."""
                return 99

        f = MyFacts()
        result = f.measure()
        assert isinstance(result, Fact)
        assert result.value == 99
        assert result.name == "Test"
        assert result.unit == "units"
        assert result.label == "measure"


class TestNeuropilFacts:
    def test_neuron_count(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        result = facts.neuron_count()
        assert isinstance(result, Fact)
        assert result.value == 15

    def test_cell_type_count(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        result = facts.cell_type_count()
        # KC_alpha, KC_beta, KC_gamma, MBON_01, MBON_05, MBIN_a, MBIN_b = 7
        assert result.value == 7

    def test_dominant_neurotransmitter(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        result = facts.dominant_neurotransmitter()
        assert result.value == "acetylcholine"

    def test_collect(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        all_facts = facts.collect()
        assert len(all_facts) >= 5
        labels = [f.label for f in all_facts]
        assert "neuron_count" in labels
        assert "neurotransmitter_breakdown" in labels

    def test_collect_dicts(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        dicts = facts.collect_dicts()
        assert all(isinstance(d, dict) for d in dicts)
        assert all("label" in d for d in dicts)

    def test_to_dataframe(self, mushroom_body_neurons):
        facts = NeuropilFacts(mushroom_body_neurons, target="MB")
        df = facts.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns
        assert len(df) >= 5

    def test_dev_mode_tolerates_errors(self):
        """Dev mode should skip broken facts, not crash."""
        empty = pd.DataFrame(columns=["root_id"])
        facts = NeuropilFacts(empty, target="empty")
        # This should not raise even if some facts fail
        results = facts.collect(mode="dev")
        # May have some None-filtered results
        assert isinstance(results, list)


class TestInterface:
    def test_delegates_to_helper(self):
        class Helper:
            def compute(self):
                return 42

        class MyFacts(Factology):
            @interface
            def compute(self):
                raise NotImplementedError

        helper = Helper()
        f = MyFacts(pd.DataFrame(), helper=helper)
        assert f.compute() == 42

    def test_raises_without_helper(self):
        class MyFacts(Factology):
            @interface
            def compute(self):
                raise NotImplementedError

        f = MyFacts(pd.DataFrame())
        with pytest.raises(NotImplementedError):
            f.compute()
