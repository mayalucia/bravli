"""Tests for the cell models module."""

import pandas as pd
import pytest

from bravli.models.cell_models import (
    LIFParams,
    GradedParams,
    CellModelDB,
    CELL_MODEL_DB,
    get_cell_params,
    list_cell_models,
    HIGH, MEDIUM, LOW,
)
from bravli.models.assign import (
    assign_cell_models,
    population_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """Synthetic neuron annotations spanning several super/cell classes."""
    return pd.DataFrame({
        "root_id": range(1, 11),
        "super_class": [
            "central", "central", "central", "central",
            "optic", "optic", "optic",
            "sensory", "motor", "descending",
        ],
        "cell_class": [
            "KC", "KC", "PN", "MBON",
            "Mi", "Tm", "T4",
            "ORN", "MN_fast", "DN",
        ],
        "cell_type": [
            "KC_a", "KC_b", "PN_DM1", "MBON_01",
            "Mi1", "Tm1", "T4a",
            "ORN_a", "MN5", "DN_a",
        ],
    })


# ---------------------------------------------------------------------------
# LIFParams tests
# ---------------------------------------------------------------------------

class TestLIFParams:
    def test_default_values(self):
        m = LIFParams(name="test")
        assert m.v_rest == -55.0
        assert m.v_thresh == -45.0
        assert m.mode == "spiking"

    def test_g_leak(self):
        m = LIFParams(name="test", r_input=500.0)
        assert abs(m.g_leak - 2.0) < 0.01  # 1000/500 = 2 nS

    def test_to_dict(self):
        m = LIFParams(name="test")
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["mode"] == "spiking"
        assert "v_rest_mV" in d
        assert "g_leak_nS" in d

    def test_frozen(self):
        m = LIFParams(name="test")
        with pytest.raises(AttributeError):
            m.v_rest = -60.0


class TestGradedParams:
    def test_default_threshold(self):
        m = GradedParams(name="test")
        assert m.v_thresh == 100.0  # unreachable
        assert m.mode == "graded"

    def test_v_reset_equals_v_rest(self):
        m = GradedParams(name="test", v_rest=-55.0)
        assert m.v_reset == -55.0

    def test_to_dict_has_v_range(self):
        m = GradedParams(name="test")
        d = m.to_dict()
        assert "v_range_mV" in d


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestCellModelDB:
    def test_db_has_models(self):
        assert len(CELL_MODEL_DB) >= 10

    def test_get_known_model(self):
        m = get_cell_params("kenyon_cell")
        assert m.name == "kenyon_cell"
        assert m.mode == "spiking"
        assert m.confidence == HIGH

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_cell_params("nonexistent")

    def test_resolve_by_cell_class(self):
        m = CELL_MODEL_DB.resolve(cell_class="KC")
        assert m.name == "kenyon_cell"

    def test_resolve_by_super_class(self):
        m = CELL_MODEL_DB.resolve(super_class="optic")
        assert m.mode == "graded"

    def test_resolve_fallback_to_default(self):
        m = CELL_MODEL_DB.resolve(cell_class="unknown_class")
        assert m.name == "default_spiking"

    def test_list_models(self):
        models = list_cell_models()
        assert len(models) >= 10
        names = [m["name"] for m in models]
        assert "kenyon_cell" in names
        assert "optic_graded" in names
        assert "shiu_uniform" in names

    def test_shiu_params(self):
        m = get_cell_params("shiu_uniform")
        assert m.v_rest == -52.0
        assert m.v_reset == -52.0
        assert m.tau_m == 20.0
        assert m.t_ref == 2.2

    def test_optic_is_graded(self):
        m = CELL_MODEL_DB.resolve(super_class="optic")
        assert isinstance(m, GradedParams)
        assert m.v_thresh == 100.0


# ---------------------------------------------------------------------------
# Assignment tests
# ---------------------------------------------------------------------------

class TestAssignCellModels:
    def test_uniform_mode(self, sample_annotations):
        result = assign_cell_models(sample_annotations, mode="uniform")
        assert all(result["model_name"] == "shiu_uniform")
        assert all(result["v_rest"] == -52.0)

    def test_class_aware_mode(self, sample_annotations):
        result = assign_cell_models(sample_annotations, mode="class_aware")

        # KCs should get kenyon_cell model
        kc = result[result["cell_class"] == "KC"]
        assert all(kc["model_name"] == "kenyon_cell")
        assert all(kc["tau_m"] == 5.0)

        # PNs should get projection_neuron model
        pn = result[result["cell_class"] == "PN"]
        assert all(pn["model_name"] == "projection_neuron")

        # Optic lobe neurons should get graded model
        optic = result[result["super_class"] == "optic"]
        assert all(optic["model_mode"] == "graded")

        # Fast motoneurons get their own model
        mn = result[result["cell_class"] == "MN_fast"]
        assert all(mn["model_name"] == "motoneuron_fast")

    def test_has_all_expected_columns(self, sample_annotations):
        result = assign_cell_models(sample_annotations)
        expected = ["model_name", "model_mode", "v_rest", "v_thresh",
                    "v_reset", "tau_m", "t_ref", "c_m", "r_input",
                    "model_confidence"]
        for col in expected:
            assert col in result.columns


class TestPopulationSummary:
    def test_summary_structure(self, sample_annotations):
        assigned = assign_cell_models(sample_annotations)
        summary = population_summary(assigned)
        assert "n_neurons" in summary.columns
        assert "mode" in summary.columns
        assert "confidence" in summary.columns

    def test_total_matches(self, sample_annotations):
        assigned = assign_cell_models(sample_annotations)
        summary = population_summary(assigned)
        assert summary["n_neurons"].sum() == len(sample_annotations)
