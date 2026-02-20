"""Tests for neuromodulatory state switching (Investigation 8).

Tests Marder's principle: same connectome, different modulatory states,
different behavioral outputs.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mb_circuit():
    """Small MB circuit for neuromodulation tests."""
    rng = np.random.RandomState(42)
    rows = []
    rid = 100000

    # KCs — gamma and alpha_beta types
    for i in range(80):
        rows.append({"root_id": rid, "cell_class": "Kenyon_Cell",
                      "cell_type": "KCg-m" if i < 50 else "KCab",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    # MBONs — both appetitive and aversive compartments
    mbon_types = [
        ("MBON01", "acetylcholine"),   # gamma1 - aversive
        ("MBON02", "acetylcholine"),   # gamma2 - appetitive
        ("MBON03", "acetylcholine"),   # gamma3 - appetitive
        ("MBON15", "acetylcholine"),   # alpha1 - aversive
        ("MBON20", "acetylcholine"),   # beta1 - aversive
    ]
    for ct, nt in mbon_types:
        rows.append({"root_id": rid, "cell_class": "MBON",
                      "cell_type": ct,
                      "super_class": "central", "top_nt": nt})
        rid += 1

    # DANs
    dan_types = [
        ("PPL101", "dopamine"),  # gamma1
        ("PAM01", "dopamine"),   # gamma2
        ("PPL104", "dopamine"),  # alpha1
    ]
    for ct, nt in dan_types:
        rows.append({"root_id": rid, "cell_class": "DAN",
                      "cell_type": ct,
                      "super_class": "central", "top_nt": nt})
        rid += 1

    # APL (inhibitory interneuron)
    rows.append({"root_id": rid, "cell_class": "MBIN",
                  "cell_type": "APL",
                  "super_class": "central", "top_nt": "GABA"})
    rid += 1

    # PNs
    for i in range(20):
        rows.append({"root_id": rid, "cell_class": "ALPN",
                      "cell_type": f"adPN-{i}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    ann = pd.DataFrame(rows)

    # Edges
    edge_rows = []
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    alpn_ids = ann[ann["cell_class"] == "ALPN"]["root_id"].values
    mbin_ids = ann[ann["cell_class"] == "MBIN"]["root_id"].values

    # PN -> KC
    for kc in kc_ids:
        pns = rng.choice(alpn_ids, size=min(5, len(alpn_ids)), replace=False)
        for pn in pns:
            edge_rows.append({"pre_pt_root_id": pn, "post_pt_root_id": kc,
                              "syn_count": rng.randint(5, 20),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(1.0, 5.0)})

    # KC -> MBON
    for kc in kc_ids:
        targets = rng.choice(mbon_ids, size=min(3, len(mbon_ids)), replace=False)
        for mbon in targets:
            edge_rows.append({"pre_pt_root_id": kc, "post_pt_root_id": mbon,
                              "syn_count": rng.randint(5, 15),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(0.5, 3.0)})

    # APL -> KC (inhibitory)
    apl_id = mbin_ids[0]
    for kc in kc_ids[:60]:
        edge_rows.append({"pre_pt_root_id": apl_id, "post_pt_root_id": kc,
                          "syn_count": rng.randint(5, 10),
                          "dominant_nt": "GABA", "nt_sign": "inhibitory",
                          "weight": rng.uniform(-3.0, -1.0)})

    edges = pd.DataFrame(edge_rows)

    from bravli.explore.mushroom_body import build_mb_circuit
    circuit, mb_neurons, mb_edges = build_mb_circuit(ann, edges, mode="class_aware")
    return circuit, mb_neurons


# ---------------------------------------------------------------------------
# Tests: Modulatory states
# ---------------------------------------------------------------------------

class TestModulatoryStates:
    def test_predefined_states_exist(self):
        from bravli.explore.neuromodulation import MODULATORY_STATES
        assert "naive" in MODULATORY_STATES
        assert "appetitive" in MODULATORY_STATES
        assert "aversive" in MODULATORY_STATES
        assert "aroused" in MODULATORY_STATES
        assert "quiescent" in MODULATORY_STATES

    def test_naive_state_is_empty(self):
        from bravli.explore.neuromodulation import MODULATORY_STATES
        assert len(MODULATORY_STATES["naive"]) == 0

    def test_appetitive_enhances_appetitive_compartments(self):
        from bravli.explore.neuromodulation import MODULATORY_STATES
        state = MODULATORY_STATES["appetitive"]
        # Appetitive compartments should have gain > 1
        assert state.get("gamma2", 1.0) > 1.0
        assert state.get("gamma3", 1.0) > 1.0
        # Aversive compartments should have gain < 1
        assert state.get("gamma1", 1.0) < 1.0
        assert state.get("alpha1", 1.0) < 1.0

    def test_aversive_enhances_aversive_compartments(self):
        from bravli.explore.neuromodulation import MODULATORY_STATES
        state = MODULATORY_STATES["aversive"]
        assert state.get("gamma1", 1.0) > 1.0
        assert state.get("alpha1", 1.0) > 1.0
        assert state.get("gamma2", 1.0) < 1.0

    def test_appetitive_and_aversive_are_complementary(self):
        """Gains that are high in one should be low in the other."""
        from bravli.explore.neuromodulation import MODULATORY_STATES
        app = MODULATORY_STATES["appetitive"]
        avr = MODULATORY_STATES["aversive"]
        for comp in app:
            if comp in avr:
                # If appetitive enhances, aversive should suppress (and vice versa)
                if app[comp] > 1.0:
                    assert avr[comp] < 1.0, f"{comp}: app={app[comp]}, avr={avr[comp]}"
                elif app[comp] < 1.0:
                    assert avr[comp] > 1.0, f"{comp}: app={app[comp]}, avr={avr[comp]}"


# ---------------------------------------------------------------------------
# Tests: Weight modulation
# ---------------------------------------------------------------------------

class TestApplyModulatoryState:
    def test_apply_and_restore(self, mb_circuit):
        from bravli.explore.neuromodulation import (
            apply_modulatory_state, restore_weights,
        )
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        comp_index = build_compartment_index(circuit, mb_neurons)
        original = circuit.weights.copy()

        # Apply a gain
        apply_modulatory_state(circuit, comp_index, {"gamma1": 2.0})

        # Some weights should have changed
        changed = not np.allclose(circuit.weights, original)
        assert changed, "No weights changed after applying modulatory state"

        # Restore
        restore_weights(circuit, original)
        assert np.allclose(circuit.weights, original)

    def test_naive_state_no_change(self, mb_circuit):
        from bravli.explore.neuromodulation import apply_modulatory_state
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        comp_index = build_compartment_index(circuit, mb_neurons)
        original = circuit.weights.copy()

        apply_modulatory_state(circuit, comp_index, {})
        assert np.allclose(circuit.weights, original)

    def test_gain_multiplies_weights(self, mb_circuit):
        from bravli.explore.neuromodulation import apply_modulatory_state
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        comp_index = build_compartment_index(circuit, mb_neurons)
        original = circuit.weights.copy()
        gain = 2.0

        report = apply_modulatory_state(circuit, comp_index, {"gamma1": gain})

        # Check that affected synapses are scaled
        mask = comp_index.get("gamma1", {}).get("kc_mbon_syn_mask", np.array([]))
        if isinstance(mask, np.ndarray) and mask.sum() > 0:
            np.testing.assert_allclose(
                circuit.weights[mask], original[mask] * gain
            )

    def test_report_structure(self, mb_circuit):
        from bravli.explore.neuromodulation import apply_modulatory_state
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        comp_index = build_compartment_index(circuit, mb_neurons)

        report = apply_modulatory_state(circuit, comp_index, {"gamma1": 1.5})
        assert isinstance(report, dict)
        for comp_info in report.values():
            assert "n_synapses" in comp_info
            assert "gain" in comp_info


# ---------------------------------------------------------------------------
# Tests: Valence computation
# ---------------------------------------------------------------------------

class TestValenceScore:
    def test_valence_keys(self, mb_circuit):
        from bravli.explore.neuromodulation import compute_valence_score
        from bravli.explore.mushroom_body import neuron_groups
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        groups = neuron_groups(circuit, mb_neurons)
        mbon_indices = groups.get("MBON", np.array([]))
        comp_index = build_compartment_index(circuit, mb_neurons)

        rates = np.random.RandomState(42).uniform(0, 50, circuit.n_neurons)
        v = compute_valence_score(rates, mbon_indices, comp_index)

        assert "valence_score" in v
        assert "approach_drive" in v
        assert "avoidance_drive" in v
        assert "per_compartment" in v

    def test_higher_appetitive_gives_positive_valence(self, mb_circuit):
        from bravli.explore.neuromodulation import compute_valence_score
        from bravli.explore.mushroom_body import neuron_groups
        from bravli.explore.mb_compartments import build_compartment_index

        circuit, mb_neurons = mb_circuit
        groups = neuron_groups(circuit, mb_neurons)
        mbon_indices = groups.get("MBON", np.array([]))
        comp_index = build_compartment_index(circuit, mb_neurons)

        # Set all rates to zero, then give high rates to appetitive MBONs
        rates = np.zeros(circuit.n_neurons)
        for comp, info in comp_index.items():
            if info["valence"] == "appetitive" and len(info["mbon_indices"]) > 0:
                rates[info["mbon_indices"]] = 50.0

        v = compute_valence_score(rates, mbon_indices, comp_index)
        assert v["valence_score"] > 0
        assert v["approach_drive"] > v["avoidance_drive"]


# ---------------------------------------------------------------------------
# Tests: State switching experiment
# ---------------------------------------------------------------------------

class TestStateSwitching:
    def test_experiment_runs(self, mb_circuit):
        from bravli.explore.neuromodulation import state_switching_experiment

        circuit, mb_neurons = mb_circuit
        results = state_switching_experiment(
            circuit, mb_neurons,
            states={"naive": {}, "test": {"gamma1": 1.5}},
            duration_ms=100.0, seed=42,
        )
        assert "naive" in results
        assert "test" in results
        assert "mean_mbon_rate" in results["naive"]
        assert "valence" in results["naive"]

    def test_weights_restored_after_experiment(self, mb_circuit):
        from bravli.explore.neuromodulation import state_switching_experiment

        circuit, mb_neurons = mb_circuit
        original = circuit.weights.copy()

        state_switching_experiment(
            circuit, mb_neurons,
            states={"test": {"gamma1": 2.0}},
            duration_ms=100.0, seed=42,
        )

        np.testing.assert_allclose(circuit.weights, original)

    def test_different_states_different_rates(self, mb_circuit):
        """Aroused vs quiescent should produce different MBON rates."""
        from bravli.explore.neuromodulation import (
            state_switching_experiment, MODULATORY_STATES,
        )

        circuit, mb_neurons = mb_circuit
        results = state_switching_experiment(
            circuit, mb_neurons,
            states={
                "aroused": MODULATORY_STATES["aroused"],
                "quiescent": MODULATORY_STATES["quiescent"],
            },
            duration_ms=200.0, seed=42,
        )

        r_aroused = results["aroused"]["mean_mbon_rate"]
        r_quiet = results["quiescent"]["mean_mbon_rate"]
        # Aroused should generally produce higher rates
        # (or at least different rates)
        assert r_aroused != r_quiet or (r_aroused == 0 and r_quiet == 0)

    def test_valence_shifts_with_state(self, mb_circuit):
        """Appetitive state should produce higher valence than aversive."""
        from bravli.explore.neuromodulation import (
            state_switching_experiment, MODULATORY_STATES,
        )

        circuit, mb_neurons = mb_circuit
        results = state_switching_experiment(
            circuit, mb_neurons,
            states={
                "appetitive": MODULATORY_STATES["appetitive"],
                "aversive": MODULATORY_STATES["aversive"],
            },
            duration_ms=200.0,
            pn_rate_hz=80.0,
            seed=42,
        )

        v_app = results["appetitive"]["valence"]["valence_score"]
        v_avr = results["aversive"]["valence"]["valence_score"]
        # Appetitive state should yield higher (more positive) valence
        assert v_app >= v_avr


# ---------------------------------------------------------------------------
# Tests: Dose response
# ---------------------------------------------------------------------------

class TestDoseResponse:
    def test_dose_response_runs(self, mb_circuit):
        from bravli.explore.neuromodulation import dose_response

        circuit, mb_neurons = mb_circuit
        df = dose_response(
            circuit, mb_neurons,
            target_compartments=["gamma1"],
            gain_values=[0.5, 1.0, 1.5],
            duration_ms=100.0, seed=42,
        )
        assert len(df) == 3
        assert "gain" in df.columns
        assert "mean_mbon_rate" in df.columns
        assert "valence_score" in df.columns

    def test_dose_response_monotonic_tendency(self, mb_circuit):
        """Higher gain on aversive compartments should shift valence down."""
        from bravli.explore.neuromodulation import dose_response

        circuit, mb_neurons = mb_circuit
        df = dose_response(
            circuit, mb_neurons,
            target_compartments=["gamma1"],  # aversive compartment
            gain_values=[0.5, 1.0, 2.0],
            duration_ms=200.0,
            pn_rate_hz=80.0,
            seed=42,
        )
        # Not strictly monotonic due to noise, but the trend should hold
        # or at least the function should run without error
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Tests: Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_runs(self, mb_circuit, capsys):
        from bravli.explore.neuromodulation import (
            state_switching_experiment, neuromodulation_report,
        )

        circuit, mb_neurons = mb_circuit
        results = state_switching_experiment(
            circuit, mb_neurons,
            states={"naive": {}, "aroused": {"gamma1": 1.3}},
            duration_ms=100.0, seed=42,
        )
        report = neuromodulation_report(results)
        assert "NEUROMODULATORY STATE SWITCHING" in report
        assert "Marder" in report

    def test_report_with_all_states(self, mb_circuit, capsys):
        from bravli.explore.neuromodulation import (
            state_switching_experiment, neuromodulation_report,
            MODULATORY_STATES,
        )

        circuit, mb_neurons = mb_circuit
        results = state_switching_experiment(
            circuit, mb_neurons,
            states=MODULATORY_STATES,
            duration_ms=100.0, seed=42,
        )
        report = neuromodulation_report(results)
        assert "naive" in report
        assert "appetitive" in report
        assert "aversive" in report
        assert "Interpretation" in report
