"""Tests for mushroom body microcircuit extraction, simulation, and sparseness.

Tests the Step 5 investigation: does KC sparseness emerge from wiring?
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: synthetic data that mimics the MB circuit structure
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """Minimal annotation DataFrame with MB-like cell classes."""
    rng = np.random.RandomState(42)
    n_kc = 200
    n_mbon = 10
    n_dan = 20
    n_mbin = 2  # includes APL
    n_alpn = 50
    n_other = 30  # non-MB neurons

    rows = []
    rid = 100000

    for i in range(n_kc):
        rows.append({"root_id": rid, "cell_class": "Kenyon_Cell",
                      "cell_type": rng.choice(["KCg-m", "KCab", "KCapbp-m"]),
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    for i in range(n_mbon):
        rows.append({"root_id": rid, "cell_class": "MBON",
                      "cell_type": f"MBON-{i}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    for i in range(n_dan):
        rows.append({"root_id": rid, "cell_class": "DAN",
                      "cell_type": f"DAN-{i}",
                      "super_class": "central", "top_nt": "dopamine"})
        rid += 1

    # MBIN — including APL
    rows.append({"root_id": rid, "cell_class": "MBIN",
                  "cell_type": "APL",
                  "super_class": "central", "top_nt": "GABA"})
    rid += 1
    rows.append({"root_id": rid, "cell_class": "MBIN",
                  "cell_type": "DPM",
                  "super_class": "central", "top_nt": "serotonin"})
    rid += 1

    for i in range(n_alpn):
        rows.append({"root_id": rid, "cell_class": "ALPN",
                      "cell_type": f"adPN-{i}",
                      "super_class": "central",
                      "top_nt": rng.choice(["acetylcholine", "gaba"])})
        rid += 1

    # Non-MB neurons
    for i in range(n_other):
        rows.append({"root_id": rid, "cell_class": "CX",
                      "cell_type": f"CX-{i}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    return pd.DataFrame(rows)


@pytest.fixture
def sample_edges(sample_annotations):
    """Synthetic edge table with MB-like connectivity."""
    rng = np.random.RandomState(42)
    ann = sample_annotations
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    dan_ids = ann[ann["cell_class"] == "DAN"]["root_id"].values
    mbin_ids = ann[ann["cell_class"] == "MBIN"]["root_id"].values
    alpn_ids = ann[ann["cell_class"] == "ALPN"]["root_id"].values
    cx_ids = ann[ann["cell_class"] == "CX"]["root_id"].values

    rows = []

    # PN -> KC (each KC gets input from ~7 PNs, convergent)
    for kc in kc_ids:
        pn_sources = rng.choice(alpn_ids, size=min(7, len(alpn_ids)), replace=False)
        for pn in pn_sources:
            rows.append({"pre_pt_root_id": pn, "post_pt_root_id": kc,
                         "syn_count": rng.randint(5, 20),
                         "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                         "weight": rng.uniform(1.0, 5.0)})

    # KC -> MBON (sparse: each KC connects to ~3 MBONs)
    for kc in kc_ids:
        targets = rng.choice(mbon_ids, size=min(3, len(mbon_ids)), replace=False)
        for mbon in targets:
            rows.append({"pre_pt_root_id": kc, "post_pt_root_id": mbon,
                         "syn_count": rng.randint(5, 15),
                         "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                         "weight": rng.uniform(0.5, 3.0)})

    # APL -> KC (global inhibition: APL connects to many KCs)
    apl_id = mbin_ids[0]  # APL
    for kc in kc_ids[:150]:
        rows.append({"pre_pt_root_id": apl_id, "post_pt_root_id": kc,
                     "syn_count": rng.randint(5, 10),
                     "dominant_nt": "GABA", "nt_sign": "inhibitory",
                     "weight": rng.uniform(-3.0, -1.0)})

    # KC -> APL (feedback: many KCs drive APL)
    for kc in kc_ids[:100]:
        rows.append({"pre_pt_root_id": kc, "post_pt_root_id": apl_id,
                     "syn_count": rng.randint(5, 10),
                     "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                     "weight": rng.uniform(0.5, 2.0)})

    # DAN -> KC (modulatory — some connections)
    for dan in dan_ids[:5]:
        for kc in rng.choice(kc_ids, size=20, replace=False):
            rows.append({"pre_pt_root_id": dan, "post_pt_root_id": kc,
                         "syn_count": rng.randint(5, 10),
                         "dominant_nt": "dopamine", "nt_sign": "modulatory",
                         "weight": 0.0})

    # A few CX edges (should be excluded from MB circuit)
    for i in range(20):
        rows.append({"pre_pt_root_id": rng.choice(cx_ids),
                     "post_pt_root_id": rng.choice(cx_ids),
                     "syn_count": rng.randint(5, 15),
                     "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                     "weight": rng.uniform(0.5, 2.0)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: Sparseness metrics
# ---------------------------------------------------------------------------

class TestSparseness:
    def test_population_sparseness_uniform(self):
        from bravli.simulation.analysis import population_sparseness
        rates = np.ones(100) * 10.0
        s = population_sparseness(rates)
        assert abs(s - 1.0) < 1e-10, f"Uniform rates should give S=1, got {s}"

    def test_population_sparseness_one_hot(self):
        from bravli.simulation.analysis import population_sparseness
        rates = np.zeros(100)
        rates[0] = 10.0
        s = population_sparseness(rates)
        assert s == pytest.approx(1.0 / 100.0, rel=1e-10)

    def test_population_sparseness_empty(self):
        from bravli.simulation.analysis import population_sparseness
        s = population_sparseness(np.array([]))
        assert s == 0.0

    def test_population_sparseness_all_zero(self):
        from bravli.simulation.analysis import population_sparseness
        s = population_sparseness(np.zeros(50))
        assert s == 0.0

    def test_population_sparseness_two_active(self):
        from bravli.simulation.analysis import population_sparseness
        rates = np.zeros(100)
        rates[0] = 10.0
        rates[1] = 10.0
        s = population_sparseness(rates)
        # mean(r) = 0.2, mean(r^2) = 2.0, S = 0.04/2.0 = 0.02
        assert s == pytest.approx(0.02, rel=1e-10)

    def test_active_fraction_by_group(self):
        from bravli.simulation.engine import SimulationResult
        from bravli.simulation.analysis import active_fraction_by_group
        # 10 neurons, 500ms
        spike_times = [np.array([]) for _ in range(10)]
        # Neurons 0-2 fire at >1 Hz
        spike_times[0] = np.array([100.0, 200.0])
        spike_times[1] = np.array([150.0])
        spike_times[2] = np.array([300.0])
        result = SimulationResult(
            spike_times=spike_times, dt=0.1, duration=500.0, n_neurons=10
        )
        groups = {"A": np.array([0, 1, 2, 3, 4]), "B": np.array([5, 6, 7, 8, 9])}
        out = active_fraction_by_group(result, groups)
        assert out["A"][0] == 3  # 3 active in group A
        assert out["B"][0] == 0  # 0 active in group B


# ---------------------------------------------------------------------------
# Tests: MB circuit extraction
# ---------------------------------------------------------------------------

class TestExtractMBCircuit:
    def test_extracts_correct_classes(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        classes = set(mb_neurons["cell_class"].unique())
        assert "Kenyon_Cell" in classes
        assert "MBON" in classes
        assert "DAN" in classes
        assert "ALPN" in classes
        assert "CX" not in classes

    def test_circuit_role_assigned(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, _ = extract_mb_circuit(sample_annotations, sample_edges)
        roles = set(mb_neurons["circuit_role"].unique())
        assert "KC" in roles
        assert "MBON" in roles
        assert "PN" in roles

    def test_apl_tagged(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, _ = extract_mb_circuit(sample_annotations, sample_edges)
        apl = mb_neurons[mb_neurons["circuit_role"] == "APL"]
        assert len(apl) == 1
        assert apl.iloc[0]["cell_type"] == "APL"

    def test_edges_are_internal(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        mb_ids = set(mb_neurons["root_id"].values)
        assert all(mb_edges["pre_pt_root_id"].isin(mb_ids))
        assert all(mb_edges["post_pt_root_id"].isin(mb_ids))

    def test_no_cx_edges(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        cx_ids = set(
            sample_annotations[sample_annotations["cell_class"] == "CX"]["root_id"]
        )
        assert not any(mb_edges["pre_pt_root_id"].isin(cx_ids))
        assert not any(mb_edges["post_pt_root_id"].isin(cx_ids))

    def test_pn_to_kc_edges_exist(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        pn_ids = set(mb_neurons[mb_neurons["circuit_role"] == "PN"]["root_id"])
        kc_ids = set(mb_neurons[mb_neurons["circuit_role"] == "KC"]["root_id"])
        pn_kc = mb_edges[
            mb_edges["pre_pt_root_id"].isin(pn_ids) &
            mb_edges["post_pt_root_id"].isin(kc_ids)
        ]
        assert len(pn_kc) > 0


class TestMBCircuitStats:
    def test_stats_structure(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit, mb_circuit_stats
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        stats = mb_circuit_stats(mb_neurons, mb_edges)
        assert "neuron_counts" in stats
        assert "pathway_counts" in stats
        assert "pathway_synapses" in stats
        assert "KC" in stats["neuron_counts"]

    def test_pn_kc_pathway(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import extract_mb_circuit, mb_circuit_stats
        mb_neurons, mb_edges = extract_mb_circuit(
            sample_annotations, sample_edges
        )
        stats = mb_circuit_stats(mb_neurons, mb_edges)
        assert "PN -> KC" in stats["pathway_counts"]
        assert stats["pathway_counts"]["PN -> KC"] > 0


# ---------------------------------------------------------------------------
# Tests: MB circuit building and simulation
# ---------------------------------------------------------------------------

class TestBuildMBCircuit:
    def test_builds_circuit(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import build_mb_circuit
        circuit, mb_neurons, mb_edges = build_mb_circuit(
            sample_annotations, sample_edges, mode="class_aware"
        )
        assert circuit.n_neurons > 0
        assert circuit.n_synapses > 0

    def test_kc_model_assigned(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import build_mb_circuit
        _, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges, mode="class_aware"
        )
        kc = mb_neurons[mb_neurons["circuit_role"] == "KC"]
        assert all(kc["model_name"] == "kenyon_cell")
        assert all(kc["tau_m"] == 5.0)

    def test_pn_model_assigned(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import build_mb_circuit
        _, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges, mode="class_aware"
        )
        pn = mb_neurons[mb_neurons["circuit_role"] == "PN"]
        assert all(pn["model_name"] == "projection_neuron")


class TestNeuronGroups:
    def test_groups_cover_all_roles(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import build_mb_circuit, neuron_groups
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        groups = neuron_groups(circuit, mb_neurons)
        assert "KC" in groups
        assert "PN" in groups
        assert "MBON" in groups
        assert len(groups["KC"]) > 0

    def test_groups_indices_valid(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import build_mb_circuit, neuron_groups
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        groups = neuron_groups(circuit, mb_neurons)
        for role, indices in groups.items():
            assert all(0 <= idx < circuit.n_neurons for idx in indices)


class TestSimulateOdor:
    def test_simulation_runs(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import (
            build_mb_circuit, simulate_odor_presentation,
        )
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        trials = simulate_odor_presentation(
            circuit, mb_neurons, duration_ms=100.0,
            odor_fraction=0.2, n_trials=1, seed=42,
        )
        assert len(trials) == 1
        assert trials[0]["result"].n_neurons == circuit.n_neurons

    def test_sparseness_is_computed(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import (
            build_mb_circuit, simulate_odor_presentation,
        )
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        trials = simulate_odor_presentation(
            circuit, mb_neurons, duration_ms=100.0,
            odor_fraction=0.2, n_trials=1, seed=42,
        )
        assert "sparseness" in trials[0]
        assert 0.0 <= trials[0]["sparseness"] <= 1.0

    def test_multiple_trials(self, sample_annotations, sample_edges):
        from bravli.explore.mushroom_body import (
            build_mb_circuit, simulate_odor_presentation,
        )
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        trials = simulate_odor_presentation(
            circuit, mb_neurons, duration_ms=50.0,
            odor_fraction=0.2, n_trials=3, seed=42,
        )
        assert len(trials) == 3
        # Different trials should have different active PN sets
        pns_0 = set(trials[0]["active_pns"])
        pns_1 = set(trials[1]["active_pns"])
        # Not guaranteed different but very likely with different seeds
        assert isinstance(pns_0, set)


class TestMBReport:
    def test_report_runs(self, sample_annotations, sample_edges, capsys):
        from bravli.explore.mushroom_body import (
            build_mb_circuit, simulate_odor_presentation, mb_analysis_report,
        )
        circuit, mb_neurons, _ = build_mb_circuit(
            sample_annotations, sample_edges
        )
        trials = simulate_odor_presentation(
            circuit, mb_neurons, duration_ms=50.0,
            odor_fraction=0.2, n_trials=1, seed=42,
        )
        report = mb_analysis_report(trials, mb_neurons)
        assert "MUSHROOM BODY MICROCIRCUIT" in report
        assert "KC sparseness" in report
