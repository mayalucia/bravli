"""Integration tests for the mushroom body exploration.

These tests run against the actual FlyWire annotation data and verify
the full bravli pipeline: load → filter → compose → fact-sheet.

Tests are skipped if the annotation file is not present.
"""

import pytest
from pathlib import Path

import pandas as pd

# Path to the annotation data (relative to repo root)
DATA_PATH = Path(__file__).parent.parent / "data" / "flywire_annotations" / "Supplemental_file1_neuron_annotations.tsv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"FlyWire annotation data not found at {DATA_PATH}",
)


@pytest.fixture(scope="module")
def annotations():
    """Load the full annotation table once for all tests."""
    from bravli.parcellation.load_flywire import load_flywire_annotations
    return load_flywire_annotations(DATA_PATH)


@pytest.fixture(scope="module")
def mb_neurons(annotations):
    """Extract MB neurons once for all tests."""
    from bravli.explore.mushroom_body import extract_mb_neurons
    return extract_mb_neurons(annotations)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_annotations_shape(self, annotations):
        """Annotations table has expected dimensions."""
        assert len(annotations) > 100_000, "Expected 100K+ neurons"
        assert "super_class" in annotations.columns
        assert "cell_type" in annotations.columns

    def test_annotations_have_mb_classes(self, annotations):
        """Cell classes include MB-related entries."""
        classes = annotations["cell_class"].unique()
        assert "Kenyon_Cell" in classes
        assert "MBON" in classes
        assert "DAN" in classes


# ---------------------------------------------------------------------------
# MB extraction tests
# ---------------------------------------------------------------------------

class TestMBExtraction:

    def test_mb_neuron_count(self, mb_neurons):
        """MB has expected number of neurons (~5,600)."""
        assert 5000 < len(mb_neurons) < 7000

    def test_kenyon_cells_dominate(self, mb_neurons):
        """Kenyon cells are the majority of MB neurons."""
        kc = mb_neurons[mb_neurons["cell_class"] == "Kenyon_Cell"]
        assert len(kc) > 0.8 * len(mb_neurons)

    def test_hemisphere_balance(self, mb_neurons):
        """MB neurons are roughly balanced across hemispheres."""
        sides = mb_neurons["side"].value_counts()
        if "left" in sides.index and "right" in sides.index:
            ratio = sides["left"] / sides["right"]
            assert 0.9 < ratio < 1.1, f"Hemisphere ratio {ratio:.2f} is unbalanced"


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------

class TestMBComposition:

    def test_composition_keys(self, mb_neurons):
        """Composition dict has all expected keys."""
        from bravli.explore.mushroom_body import mb_composition
        comp = mb_composition(mb_neurons)
        assert "class_counts" in comp
        assert "kc_subtypes" in comp
        assert "nt_profile" in comp
        assert "top_cell_types" in comp
        assert "hemisphere_balance" in comp

    def test_kc_subtypes_sum(self, mb_neurons):
        """KC subtype counts sum to total KC count."""
        from bravli.explore.mushroom_body import mb_composition
        comp = mb_composition(mb_neurons)
        kc_total = comp["class_counts"].get("Kenyon_Cell", 0)
        subtype_total = sum(comp["kc_subtypes"].values())
        assert subtype_total == kc_total

    def test_nt_profile_has_dopamine(self, mb_neurons):
        """Neurotransmitter profile includes dopamine (dominant in MB)."""
        from bravli.explore.mushroom_body import mb_composition
        comp = mb_composition(mb_neurons)
        nt = comp["nt_profile"]
        if isinstance(nt, pd.Series):
            assert "dopamine" in nt.index
        elif isinstance(nt, dict):
            assert "dopamine" in nt


# ---------------------------------------------------------------------------
# Factsheet tests
# ---------------------------------------------------------------------------

class TestMBFactsheet:

    def test_factsheet_is_dataframe(self, mb_neurons):
        """Factsheet returns a DataFrame."""
        from bravli.explore.mushroom_body import mb_factsheet
        df = mb_factsheet(mb_neurons)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_factsheet_has_neuron_count(self, mb_neurons):
        """Factsheet includes a neuron count fact."""
        from bravli.explore.mushroom_body import mb_factsheet
        df = mb_factsheet(mb_neurons)
        assert "Neuron count" in df["name"].values

    def test_neuron_count_value(self, mb_neurons):
        """Neuron count fact matches actual count."""
        from bravli.explore.mushroom_body import mb_factsheet
        df = mb_factsheet(mb_neurons)
        row = df[df["name"] == "Neuron count"]
        assert row.iloc[0]["value"] == len(mb_neurons)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_summary_report(self, annotations, capsys):
        """Full summary report executes without error."""
        from bravli.explore.mushroom_body import mb_summary_report
        result = mb_summary_report(annotations)
        assert "mb_neurons" in result
        assert "composition" in result
        assert "factsheet" in result
        # Check that it printed something
        captured = capsys.readouterr()
        assert "MUSHROOM BODY" in captured.out
        assert "Kenyon_Cell" in captured.out
