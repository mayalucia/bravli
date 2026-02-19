"""Tests for three-factor STDP, plasticity hook, and olfactory conditioning.

Tests the Investigation 7 pipeline: plasticity hook in engine, three-factor
STDP rule, conditioning protocol, and learning metrics.
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit
from bravli.simulation.engine import simulate, SimulationResult
from bravli.simulation.stimulus import step_stimulus
from bravli.simulation.analysis import (
    firing_rates, weight_evolution, mbon_response_change, performance_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """Minimal MB annotation DataFrame for conditioning tests."""
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
    """Synthetic edges for conditioning tests."""
    rng = np.random.RandomState(42)
    ann = sample_annotations
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    dan_ids = ann[ann["cell_class"] == "DAN"]["root_id"].values
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
    # DAN -> KC (needed for compartment index)
    for dan in dan_ids[:3]:
        for kc in rng.choice(kc_ids, size=10, replace=False):
            rows.append({"pre_pt_root_id": dan, "post_pt_root_id": kc,
                         "syn_count": rng.randint(5, 10),
                         "dominant_nt": "dopamine", "nt_sign": "modulatory",
                         "weight": 0.0})

    return pd.DataFrame(rows)


@pytest.fixture
def mb_circuit(sample_annotations, sample_edges):
    """Build a small MB circuit for testing."""
    from bravli.explore.mushroom_body import build_mb_circuit
    circuit, mb_neurons, mb_edges = build_mb_circuit(
        sample_annotations, sample_edges, mode="class_aware"
    )
    return circuit, mb_neurons, mb_edges


@pytest.fixture
def two_neuron_circuit():
    """Minimal circuit: neuron 0 excites neuron 1."""
    return Circuit(
        n_neurons=2,
        v_rest=np.array([-52.0, -52.0]),
        v_thresh=np.array([-45.0, -45.0]),
        v_reset=np.array([-52.0, -52.0]),
        tau_m=np.array([20.0, 20.0]),
        t_ref=np.array([2.2, 2.2]),
        pre_idx=np.array([0], dtype=np.int32),
        post_idx=np.array([1], dtype=np.int32),
        weights=np.array([10.0]),
        tau_syn=5.0,
        delay_steps=18,
    )


# ---------------------------------------------------------------------------
# Tests: Plasticity Hook in Engine
# ---------------------------------------------------------------------------

class TestPlasticityHook:
    def test_plasticity_fn_none_unchanged(self, two_neuron_circuit):
        """simulate with plasticity_fn=None should behave identically."""
        n_steps = int(100.0 / 0.1)
        stim, _ = step_stimulus(2, n_steps, [0], amplitude=15.0,
                                start_ms=10.0, end_ms=90.0)
        r1 = simulate(two_neuron_circuit, duration=100.0, dt=0.1,
                      stimulus=stim, seed=42, plasticity_fn=None)
        # Should produce spikes (same as without the parameter)
        assert r1.n_spikes > 0

    def test_plasticity_fn_called(self, two_neuron_circuit):
        """A plasticity callback should be invoked each timestep."""
        call_count = [0]
        def counter(step, t, dt, spiked, v, g, circuit):
            call_count[0] += 1

        n_steps = int(50.0 / 0.1)
        stim, _ = step_stimulus(2, n_steps, [0], amplitude=15.0,
                                start_ms=5.0, end_ms=45.0)
        simulate(two_neuron_circuit, duration=50.0, dt=0.1,
                 stimulus=stim, plasticity_fn=counter)
        assert call_count[0] == n_steps

    def test_plasticity_fn_can_modify_weights(self, two_neuron_circuit):
        """Plasticity callback should be able to mutate weights."""
        original_w = two_neuron_circuit.weights[0]

        def halve_weights(step, t, dt, spiked, v, g, circuit):
            if step == 0:
                circuit.weights *= 0.5

        simulate(two_neuron_circuit, duration=10.0, dt=0.1,
                 plasticity_fn=halve_weights)
        assert two_neuron_circuit.weights[0] == pytest.approx(original_w * 0.5)


# ---------------------------------------------------------------------------
# Tests: Three-Factor STDP
# ---------------------------------------------------------------------------

class TestThreeFactorSTDP:
    def test_init(self, mb_circuit):
        from bravli.simulation.plasticity import ThreeFactorSTDP
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        rule = ThreeFactorSTDP(compartment_index=index)
        assert rule.n_plastic_synapses >= 0
        assert len(rule.weight_snapshots) == 0

    def test_eligibility_decay(self, mb_circuit):
        """Eligibility trace should decay with time constant."""
        from bravli.simulation.plasticity import ThreeFactorSTDP
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        rule = ThreeFactorSTDP(compartment_index=index, tau_eligibility=100.0)

        if rule.n_plastic_synapses == 0:
            pytest.skip("No plastic synapses in test circuit")

        # Manually set eligibility
        rule._eligibility[:] = 1.0
        dt = 0.1
        spiked = np.zeros(circuit.n_neurons, dtype=bool)
        v = circuit.v_rest.copy()
        g = np.zeros(circuit.n_neurons)

        # One step of decay
        rule(0, 0.0, dt, spiked, v, g, circuit)
        expected = 1.0 * (1.0 - dt / 100.0)
        assert rule._eligibility[0] == pytest.approx(expected, rel=1e-6)

    def test_three_factor_requires_both(self, mb_circuit):
        """No weight change without BOTH KC spikes AND DAN spikes."""
        from bravli.simulation.plasticity import ThreeFactorSTDP
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        rule = ThreeFactorSTDP(compartment_index=index, lr=0.1,
                               snapshot_interval_ms=0)

        if rule.n_plastic_synapses == 0:
            pytest.skip("No plastic synapses in test circuit")

        initial_w = circuit.weights.copy()
        dt = 0.1
        v = circuit.v_rest.copy()
        g = np.zeros(circuit.n_neurons)

        # No spikes at all
        spiked = np.zeros(circuit.n_neurons, dtype=bool)
        for step in range(100):
            rule(step, step * dt, dt, spiked, v, g, circuit)

        np.testing.assert_array_equal(circuit.weights, initial_w)

    def test_weight_clamp(self, mb_circuit):
        """Weights should not go below w_min."""
        from bravli.simulation.plasticity import ThreeFactorSTDP
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        w_min = 0.1
        rule = ThreeFactorSTDP(compartment_index=index, lr=100.0,
                               w_min=w_min, snapshot_interval_ms=0)

        if rule.n_plastic_synapses == 0:
            pytest.skip("No plastic synapses in test circuit")

        # Force large depression
        rule._eligibility[:] = 100.0
        for comp in rule._da_signal:
            rule._da_signal[comp] = 100.0

        dt = 0.1
        spiked = np.zeros(circuit.n_neurons, dtype=bool)
        v = circuit.v_rest.copy()
        g = np.zeros(circuit.n_neurons)

        rule(0, 0.0, dt, spiked, v, g, circuit)

        # Plastic synapses should be clamped at w_min
        plastic_w = circuit.weights[rule._kc_mbon_syn_indices]
        assert np.all(plastic_w >= w_min - 1e-10)

    def test_weight_snapshots(self, mb_circuit):
        """Snapshots should be recorded at the right interval."""
        from bravli.simulation.plasticity import ThreeFactorSTDP
        from bravli.explore.mb_compartments import build_compartment_index
        circuit, mb_neurons, _ = mb_circuit
        index = build_compartment_index(circuit, mb_neurons)
        rule = ThreeFactorSTDP(compartment_index=index, snapshot_interval_ms=10.0)

        dt = 0.1
        spiked = np.zeros(circuit.n_neurons, dtype=bool)
        v = circuit.v_rest.copy()
        g = np.zeros(circuit.n_neurons)

        # Run 200 steps (20 ms) -> should get snapshots at step 0 and 100
        for step in range(200):
            rule(step, step * dt, dt, spiked, v, g, circuit)

        assert len(rule.weight_snapshots) == 2
        assert rule.snapshot_times[0] == pytest.approx(0.0)
        assert rule.snapshot_times[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: Analysis Extensions
# ---------------------------------------------------------------------------

class TestAnalysisExtensions:
    def test_weight_evolution_empty(self):
        df = weight_evolution([], [])
        assert len(df) == 0
        assert "mean_weight" in df.columns

    def test_weight_evolution_basic(self):
        w0 = np.array([1.0, 1.0, 1.0])
        w1 = np.array([0.9, 1.0, 0.8])
        df = weight_evolution([w0, w1], [0.0, 100.0])
        assert len(df) == 2
        assert df.iloc[0]["frac_depressed"] == 0.0
        assert df.iloc[1]["frac_depressed"] == pytest.approx(2.0 / 3.0)

    def test_mbon_response_change(self):
        pre = np.array([10.0, 20.0, 0.0, 5.0])
        post = np.array([5.0, 20.0, 0.0, 10.0])
        mbon_idx = np.array([0, 1, 2, 3])
        li = mbon_response_change(pre, post, mbon_idx)
        # Neuron 0: (10-5)/(10+5) = 1/3
        assert li[0] == pytest.approx(1.0 / 3.0)
        # Neuron 1: (20-20)/40 = 0
        assert li[1] == pytest.approx(0.0)
        # Neuron 2: 0/0 = 0
        assert li[2] == pytest.approx(0.0)
        # Neuron 3: (5-10)/15 = -1/3 (potentiation)
        assert li[3] == pytest.approx(-1.0 / 3.0)

    def test_performance_index(self):
        cs_plus = np.array([5.0, 10.0, 0.0])
        cs_minus = np.array([10.0, 10.0, 0.0])
        mbon_idx = np.array([0, 1, 2])
        pi = performance_index(cs_plus, cs_minus, mbon_idx)
        # Neuron 0: (10-5)/(10+5) = 1/3 (discriminates)
        assert pi[0] == pytest.approx(1.0 / 3.0)
        # Neuron 1: (10-10)/20 = 0
        assert pi[1] == pytest.approx(0.0)
        # Neuron 2: 0/0 = 0
        assert pi[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Conditioning Protocol
# ---------------------------------------------------------------------------

class TestConditioningProtocol:
    def test_conditioning_runs(self, mb_circuit):
        """Full conditioning protocol should complete without error."""
        from bravli.explore.conditioning_experiment import aversive_conditioning
        circuit, mb_neurons, mb_edges = mb_circuit
        results = aversive_conditioning(
            circuit, mb_neurons, mb_edges,
            cs_odor_fraction=0.2,
            us_compartment="gamma1",
            n_training_trials=2,
            trial_duration_ms=100.0,
            iti_ms=50.0,
            test_duration_ms=100.0,
            pn_rate_hz=50.0,
            lr=0.01,
            seed=42,
        )
        assert "error" not in results
        assert "pre_test_rates" in results
        assert "post_test_rates" in results
        assert "learning_index" in results

    def test_conditioning_weight_change(self, mb_circuit):
        """Training should produce some weight change (if synapses exist)."""
        from bravli.explore.conditioning_experiment import aversive_conditioning
        circuit, mb_neurons, mb_edges = mb_circuit
        results = aversive_conditioning(
            circuit, mb_neurons, mb_edges,
            cs_odor_fraction=0.2,
            us_compartment="gamma1",
            n_training_trials=2,
            trial_duration_ms=100.0,
            iti_ms=50.0,
            test_duration_ms=100.0,
            pn_rate_hz=50.0,
            lr=0.01,
            seed=42,
        )
        ws = results["weight_change_summary"]
        # Global mean change should be <= 0 (depression or no change)
        assert ws["global_mean_change"] <= 0.01

    def test_conditioning_has_timing(self, mb_circuit):
        from bravli.explore.conditioning_experiment import aversive_conditioning
        circuit, mb_neurons, mb_edges = mb_circuit
        results = aversive_conditioning(
            circuit, mb_neurons, mb_edges,
            n_training_trials=2,
            trial_duration_ms=100.0,
            iti_ms=50.0,
            test_duration_ms=100.0,
            seed=42,
        )
        timing = results["timing"]
        assert "pre_test" in timing
        assert "training" in timing
        assert "post_test" in timing
        assert "control" in timing
        assert len(timing["training"]) == 2

    def test_conditioning_weight_evolution_recorded(self, mb_circuit):
        from bravli.explore.conditioning_experiment import aversive_conditioning
        circuit, mb_neurons, mb_edges = mb_circuit
        results = aversive_conditioning(
            circuit, mb_neurons, mb_edges,
            n_training_trials=2,
            trial_duration_ms=100.0,
            iti_ms=50.0,
            test_duration_ms=100.0,
            seed=42,
        )
        w_evo = results["weight_evolution"]
        assert isinstance(w_evo, pd.DataFrame)
        # Should have snapshots (total duration > 100ms)
        assert len(w_evo) > 0


class TestConditioningReport:
    def test_report_runs(self, mb_circuit, capsys):
        from bravli.explore.conditioning_experiment import (
            aversive_conditioning, conditioning_report,
        )
        circuit, mb_neurons, mb_edges = mb_circuit
        results = aversive_conditioning(
            circuit, mb_neurons, mb_edges,
            n_training_trials=2,
            trial_duration_ms=100.0,
            iti_ms=50.0,
            test_duration_ms=100.0,
            seed=42,
        )
        report = conditioning_report(results)
        assert "AVERSIVE OLFACTORY CONDITIONING" in report
        assert "Learning Metrics" in report

    def test_report_error(self):
        from bravli.explore.conditioning_experiment import conditioning_report
        report = conditioning_report({"error": "test error"})
        assert "test error" in report
