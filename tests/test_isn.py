"""Tests for ISN paradoxical response experiment and MB compartments.

Tests the Investigation 6 pipeline: E/I classification, ISN protocol,
dose-response, and report generation.
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit
from bravli.simulation.engine import simulate, SimulationResult
from bravli.simulation.stimulus import step_stimulus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """Minimal MB annotation DataFrame."""
    rng = np.random.RandomState(42)
    rows = []
    rid = 100000

    for i in range(100):
        rows.append({"root_id": rid, "cell_class": "Kenyon_Cell",
                      "cell_type": rng.choice(["KCg-m", "KCab", "KCapbp-m"]),
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    for i in range(5):
        rows.append({"root_id": rid, "cell_class": "MBON",
                      "cell_type": f"MBON0{i+1}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    for i in range(10):
        rows.append({"root_id": rid, "cell_class": "DAN",
                      "cell_type": f"PAM0{i+1}",
                      "super_class": "central", "top_nt": "dopamine"})
        rid += 1

    rows.append({"root_id": rid, "cell_class": "MBIN",
                  "cell_type": "APL",
                  "super_class": "central", "top_nt": "GABA"})
    rid += 1

    for i in range(30):
        rows.append({"root_id": rid, "cell_class": "ALPN",
                      "cell_type": f"adPN-{i}",
                      "super_class": "central",
                      "top_nt": "acetylcholine"})
        rid += 1

    return pd.DataFrame(rows)


@pytest.fixture
def sample_edges(sample_annotations):
    """Synthetic edge table with MB-like connectivity."""
    rng = np.random.RandomState(42)
    ann = sample_annotations
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    alpn_ids = ann[ann["cell_class"] == "ALPN"]["root_id"].values
    mbin_ids = ann[ann["cell_class"] == "MBIN"]["root_id"].values

    rows = []
    for kc in kc_ids:
        pns = rng.choice(alpn_ids, size=min(5, len(alpn_ids)), replace=False)
        for pn in pns:
            rows.append({"pre_pt_root_id": pn, "post_pt_root_id": kc,
                         "syn_count": rng.randint(5, 20),
                         "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                         "weight": rng.uniform(1.0, 5.0)})
    for kc in kc_ids:
        targets = rng.choice(mbon_ids, size=min(2, len(mbon_ids)), replace=False)
        for mbon in targets:
            rows.append({"pre_pt_root_id": kc, "post_pt_root_id": mbon,
                         "syn_count": rng.randint(5, 15),
                         "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                         "weight": rng.uniform(0.5, 3.0)})
    apl_id = mbin_ids[0]
    for kc in kc_ids[:80]:
        rows.append({"pre_pt_root_id": apl_id, "post_pt_root_id": kc,
                     "syn_count": rng.randint(5, 10),
                     "dominant_nt": "GABA", "nt_sign": "inhibitory",
                     "weight": rng.uniform(-3.0, -1.0)})
    for kc in kc_ids[:50]:
        rows.append({"pre_pt_root_id": kc, "post_pt_root_id": apl_id,
                     "syn_count": rng.randint(5, 10),
                     "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                     "weight": rng.uniform(0.5, 2.0)})

    return pd.DataFrame(rows)


@pytest.fixture
def mb_circuit(sample_annotations, sample_edges):
    """Build a small MB circuit for testing."""
    from bravli.explore.mushroom_body import build_mb_circuit
    circuit, mb_neurons, mb_edges = build_mb_circuit(
        sample_annotations, sample_edges, mode="class_aware"
    )
    return circuit, mb_neurons, mb_edges


# ---------------------------------------------------------------------------
# Tests: MB Compartment Mapping
# ---------------------------------------------------------------------------

class TestMBCompartments:
    def test_compartment_table_complete(self):
        from bravli.explore.mb_compartments import MB_COMPARTMENTS
        assert len(MB_COMPARTMENTS) == 15

    def test_compartment_has_required_keys(self):
        from bravli.explore.mb_compartments import MB_COMPARTMENTS
        for comp, info in MB_COMPARTMENTS.items():
            assert "mbon" in info, f"{comp} missing 'mbon'"
            assert "dan" in info, f"{comp} missing 'dan'"
            assert "lobe" in info, f"{comp} missing 'lobe'"
            assert "valence" in info, f"{comp} missing 'valence'"

    def test_gamma1_is_aversive(self):
        from bravli.explore.mb_compartments import MB_COMPARTMENTS
        assert MB_COMPARTMENTS["gamma1"]["valence"] == "aversive"
        assert "PPL101" in MB_COMPARTMENTS["gamma1"]["dan"]
        assert "MBON01" in MB_COMPARTMENTS["gamma1"]["mbon"]

    def test_assign_compartments(self, mb_circuit):
        from bravli.explore.mb_compartments import assign_compartments
        _, mb_neurons, _ = mb_circuit
        df = assign_compartments(mb_neurons)
        assert "compartment" in df.columns
        # PNs should be calyx
        pn_comps = df[df["circuit_role"] == "PN"]["compartment"].unique()
        assert "calyx" in pn_comps
        # APL should be global
        apl_comps = df[df["circuit_role"] == "APL"]["compartment"].unique()
        assert "global" in apl_comps

    def test_assign_compartments_kc_lobes(self, mb_circuit):
        from bravli.explore.mb_compartments import assign_compartments
        _, mb_neurons, _ = mb_circuit
        df = assign_compartments(mb_neurons)
        kc_comps = df[df["circuit_role"] == "KC"]["compartment"].unique()
        # KCs should be assigned to lobes, not compartments
        for lobe in kc_comps:
            assert lobe in {"gamma", "alpha_beta", "alpha_beta_prime", "unknown"}

    def test_build_compartment_index(self, mb_circuit):
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        assert isinstance(index, dict)
        assert len(index) == 15
        for comp, info in index.items():
            assert "kc_indices" in info
            assert "mbon_indices" in info
            assert "dan_indices" in info
            assert "kc_mbon_syn_mask" in info
            assert len(info["kc_mbon_syn_mask"]) == circuit.n_synapses

    def test_compartment_summary(self, mb_circuit, capsys):
        from bravli.explore.mb_compartments import (
            build_compartment_index, compartment_summary,
        )
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        report = compartment_summary(index)
        assert "MB Compartment Index" in report


# ---------------------------------------------------------------------------
# Tests: E/I Classification
# ---------------------------------------------------------------------------

class TestEIGroups:
    def test_identify_ei_groups(self, mb_circuit):
        from bravli.explore.isn_experiment import identify_ei_groups
        circuit, mb_neurons, _ = mb_circuit
        groups = identify_ei_groups(circuit, mb_neurons)
        assert "E" in groups
        assert "I" in groups
        assert "modulatory" in groups
        # Should have more E than I (KCs + PNs + MBONs vs APL)
        assert len(groups["E"]) > len(groups["I"])

    def test_ei_groups_cover_all_neurons(self, mb_circuit):
        from bravli.explore.isn_experiment import identify_ei_groups
        circuit, mb_neurons, _ = mb_circuit
        groups = identify_ei_groups(circuit, mb_neurons)
        total = len(groups["E"]) + len(groups["I"]) + len(groups["modulatory"])
        assert total == circuit.n_neurons

    def test_ei_indices_valid(self, mb_circuit):
        from bravli.explore.isn_experiment import identify_ei_groups
        circuit, mb_neurons, _ = mb_circuit
        groups = identify_ei_groups(circuit, mb_neurons)
        for key, indices in groups.items():
            assert all(0 <= idx < circuit.n_neurons for idx in indices)


# ---------------------------------------------------------------------------
# Tests: ISN Experiment
# ---------------------------------------------------------------------------

class TestISNExperiment:
    def test_isn_baseline_produces_activity(self, mb_circuit):
        """E-cell drive should cause some spikes."""
        from bravli.explore.isn_experiment import isn_experiment
        circuit, mb_neurons, _ = mb_circuit
        result = isn_experiment(
            circuit, mb_neurons, e_drive=15.0, i_perturbation=5.0,
            duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        assert result["result"].n_spikes > 0

    def test_isn_returns_expected_keys(self, mb_circuit):
        from bravli.explore.isn_experiment import isn_experiment
        circuit, mb_neurons, _ = mb_circuit
        result = isn_experiment(
            circuit, mb_neurons, e_drive=15.0, i_perturbation=5.0,
            duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        expected_keys = [
            "e_rate_baseline", "e_rate_perturbation",
            "i_rate_baseline", "i_rate_perturbation",
            "e_rate_change", "i_rate_change",
            "paradoxical_response", "result", "ei_groups",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_isn_paradoxical_is_bool(self, mb_circuit):
        from bravli.explore.isn_experiment import isn_experiment
        circuit, mb_neurons, _ = mb_circuit
        result = isn_experiment(
            circuit, mb_neurons, e_drive=15.0, i_perturbation=5.0,
            duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        assert isinstance(result["paradoxical_response"], (bool, np.bool_))


class TestISNDoseResponse:
    def test_dose_response_runs(self, mb_circuit):
        from bravli.explore.isn_experiment import isn_dose_response
        circuit, mb_neurons, _ = mb_circuit
        results = isn_dose_response(
            circuit, mb_neurons,
            i_amplitudes=[2.0, 5.0],
            e_drive=15.0, duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        assert len(results) == 2
        assert results[0]["i_perturbation"] == 2.0
        assert results[1]["i_perturbation"] == 5.0


class TestISNReport:
    def test_report_single(self, mb_circuit, capsys):
        from bravli.explore.isn_experiment import isn_experiment, isn_report
        circuit, mb_neurons, _ = mb_circuit
        result = isn_experiment(
            circuit, mb_neurons, e_drive=15.0, i_perturbation=5.0,
            duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        report = isn_report(result)
        assert "ISN PARADOXICAL RESPONSE" in report
        assert "E-cell rate" in report

    def test_report_dose_response(self, mb_circuit, capsys):
        from bravli.explore.isn_experiment import isn_dose_response, isn_report
        circuit, mb_neurons, _ = mb_circuit
        results = isn_dose_response(
            circuit, mb_neurons,
            i_amplitudes=[2.0, 5.0],
            e_drive=15.0, duration_ms=200.0, onset_ms=20.0,
            baseline_epoch=(20, 100), perturbation_epoch=(100, 180),
            seed=42,
        )
        report = isn_report(results)
        assert "Interpretation" in report
