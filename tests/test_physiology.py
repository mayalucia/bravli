"""Tests for the physiology module: synapse models and assignment."""

import pandas as pd
import pytest

from bravli.physiology.synapse_models import (
    SynapseModel,
    STPParams,
    SYNAPSE_DB,
    get_synapse_model,
    list_models,
    simple_sign,
    HIGH, MEDIUM, LOW,
)
from bravli.physiology.assign import (
    assign_synapse_models,
    compute_synaptic_weights,
    physiology_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotated_edges():
    """Edge list with dominant_nt already assigned (from Lesson 08)."""
    return pd.DataFrame({
        "pre_pt_root_id":  [1, 1, 2, 3, 4, 5],
        "post_pt_root_id": [2, 3, 4, 5, 5, 6],
        "neuropil":        ["MB", "MB", "AL", "MB", "LH", "LH"],
        "syn_count":       [10, 3, 8, 2, 6, 1],
        "dominant_nt":     [
            "acetylcholine", "GABA", "acetylcholine",
            "GABA", "glutamate", "dopamine",
        ],
    })


# ---------------------------------------------------------------------------
# SynapseModel tests
# ---------------------------------------------------------------------------

class TestSynapseModel:
    def test_ach_model_exists(self):
        model = get_synapse_model("acetylcholine")
        assert model.sign == 1
        assert model.is_fast
        assert not model.is_modulatory

    def test_gaba_model_exists(self):
        model = get_synapse_model("GABA")
        assert model.sign == -1
        assert model.tau_decay > model.tau_rise
        assert model.e_rev < 0

    def test_glutamate_is_inhibitory(self):
        model = get_synapse_model("glutamate")
        assert model.sign == -1
        assert model.e_rev < 0

    def test_dopamine_is_modulatory(self):
        model = get_synapse_model("dopamine")
        assert model.sign == 0
        assert model.is_modulatory
        assert not model.is_fast
        assert model.modulation_tau > 0

    def test_unknown_nt_raises(self):
        with pytest.raises(KeyError):
            get_synapse_model("kryptonite")

    def test_all_six_nts_present(self):
        expected = {"acetylcholine", "GABA", "glutamate",
                    "dopamine", "serotonin", "octopamine"}
        assert set(SYNAPSE_DB.keys()) == expected

    def test_to_dict(self):
        model = get_synapse_model("acetylcholine")
        d = model.to_dict()
        assert d["nt_type"] == "acetylcholine"
        assert d["sign"] == 1
        assert "tau_rise_ms" in d
        assert "stp_U" in d  # has STP params

    def test_list_models(self):
        models = list_models()
        assert len(models) == 6
        assert all("nt_type" in m for m in models)


class TestSTPParams:
    def test_default_stp(self):
        stp = STPParams()
        assert stp.U == 0.4
        assert stp.tau_rec > 0
        assert stp.confidence == LOW

    def test_ach_has_facilitation(self):
        model = get_synapse_model("acetylcholine")
        assert model.stp is not None
        assert model.stp.tau_fac > 0

    def test_gaba_no_facilitation(self):
        model = get_synapse_model("GABA")
        assert model.stp is not None
        assert model.stp.tau_fac == 0


class TestSimpleSign:
    def test_biophysical_mode(self):
        assert simple_sign("acetylcholine") == 1
        assert simple_sign("GABA") == -1
        assert simple_sign("glutamate") == -1
        assert simple_sign("dopamine") == 0

    def test_shiu_mode(self):
        assert simple_sign("acetylcholine", mode="shiu") == 1
        assert simple_sign("GABA", mode="shiu") == -1
        assert simple_sign("dopamine", mode="shiu") == 1
        assert simple_sign("octopamine", mode="shiu") == 1


# ---------------------------------------------------------------------------
# Assignment tests
# ---------------------------------------------------------------------------

class TestAssignSynapseModels:
    def test_biophysical_mode(self, sample_annotated_edges):
        result = assign_synapse_models(sample_annotated_edges)
        assert "sign" in result.columns
        assert "tau_rise" in result.columns
        assert "tau_decay" in result.columns
        assert "e_rev" in result.columns
        assert "g_peak" in result.columns
        assert "model_confidence" in result.columns

        # ACh edges should be excitatory
        ach = result[result["dominant_nt"] == "acetylcholine"]
        assert all(ach["sign"] == 1)
        assert all(ach["tau_rise"] == 0.5)

        # GABA edges should be inhibitory
        gaba = result[result["dominant_nt"] == "GABA"]
        assert all(gaba["sign"] == -1)

        # DA edges should be modulatory (sign=0, no fast kinetics)
        da = result[result["dominant_nt"] == "dopamine"]
        assert all(da["sign"] == 0)
        assert all(da["tau_rise"].isna())

    def test_shiu_mode(self, sample_annotated_edges):
        result = assign_synapse_models(sample_annotated_edges, mode="shiu")
        assert "sign" in result.columns
        assert "tau_syn" in result.columns
        assert all(result["tau_syn"] == 5.0)

        # Shiu treats DA as excitatory
        da = result[result["dominant_nt"] == "dopamine"]
        assert all(da["sign"] == 1)

    def test_raises_without_dominant_nt(self):
        df = pd.DataFrame({"syn_count": [10]})
        with pytest.raises(ValueError, match="dominant_nt"):
            assign_synapse_models(df)


class TestComputeWeights:
    def test_shiu_weights(self, sample_annotated_edges):
        result = compute_synaptic_weights(sample_annotated_edges, mode="shiu")
        assert "weight" in result.columns

        # ACh edge with 10 syn: weight = 10 * 1 * 0.275 = 2.75
        ach_10 = result[
            (result["dominant_nt"] == "acetylcholine") &
            (result["syn_count"] == 10)
        ]
        assert abs(ach_10.iloc[0]["weight"] - 2.75) < 0.01

        # GABA edge with 3 syn: weight = 3 * -1 * 0.275 = -0.825
        gaba_3 = result[
            (result["dominant_nt"] == "GABA") &
            (result["syn_count"] == 3)
        ]
        assert abs(gaba_3.iloc[0]["weight"] - (-0.825)) < 0.01

    def test_conductance_weights(self, sample_annotated_edges):
        result = compute_synaptic_weights(
            sample_annotated_edges, mode="conductance"
        )
        assert "weight" in result.columns

        # ACh: g_peak=0.4, sign=1, syn_count=10 -> weight = 4.0
        ach_10 = result[
            (result["dominant_nt"] == "acetylcholine") &
            (result["syn_count"] == 10)
        ]
        assert abs(ach_10.iloc[0]["weight"] - 4.0) < 0.01


class TestPhysiologySummary:
    def test_summary_structure(self, sample_annotated_edges):
        summary = physiology_summary(sample_annotated_edges)
        assert "acetylcholine" in summary.index
        assert "GABA" in summary.index
        assert "n_edges" in summary.columns
        assert "total_synapses" in summary.columns
        assert "confidence" in summary.columns

    def test_synapse_totals(self, sample_annotated_edges):
        summary = physiology_summary(sample_annotated_edges)
        ach = summary.loc["acetylcholine"]
        assert ach["n_edges"] == 2
        assert ach["total_synapses"] == 18  # 10 + 8
