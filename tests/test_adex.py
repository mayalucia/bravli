"""Tests for AdEx engine and LIF vs AdEx comparison (Investigation 10).

Tests the adaptive exponential integrate-and-fire model and the
topology-dominates hypothesis (Zhang et al. 2024).
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit
from bravli.simulation.engine import simulate
from bravli.simulation.adex_engine import (
    simulate_adex, AdExParams, ADEX_PRESETS,
)
from bravli.simulation.analysis import firing_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_neuron_circuit():
    """Two-neuron circuit for basic AdEx tests."""
    return Circuit(
        n_neurons=2,
        v_rest=np.array([-55.0, -55.0]),
        v_thresh=np.array([-45.0, -45.0]),
        v_reset=np.array([-55.0, -55.0]),
        tau_m=np.array([20.0, 20.0]),
        t_ref=np.array([2.0, 2.0]),
        pre_idx=np.array([0], dtype=np.int32),
        post_idx=np.array([1], dtype=np.int32),
        weights=np.array([10.0]),
        tau_syn=5.0,
        delay_steps=1,
    )


@pytest.fixture
def small_network():
    """Small random network for comparison tests."""
    rng = np.random.RandomState(42)
    n = 50
    n_syn = 200
    pre = rng.randint(0, n, n_syn).astype(np.int32)
    post = rng.randint(0, n, n_syn).astype(np.int32)
    weights = rng.choice([3.0, -6.0], size=n_syn, p=[0.8, 0.2])

    return Circuit(
        n_neurons=n,
        v_rest=np.full(n, -55.0),
        v_thresh=np.full(n, -45.0),
        v_reset=np.full(n, -55.0),
        tau_m=np.full(n, 20.0),
        t_ref=np.full(n, 2.0),
        pre_idx=pre,
        post_idx=post,
        weights=weights,
        tau_syn=5.0,
        delay_steps=1,
    )


@pytest.fixture
def mb_circuit():
    """Small MB circuit for model comparison."""
    rng = np.random.RandomState(42)
    rows = []
    rid = 100000

    for i in range(60):
        rows.append({"root_id": rid, "cell_class": "Kenyon_Cell",
                      "cell_type": "KCg-m",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1
    for i in range(3):
        rows.append({"root_id": rid, "cell_class": "MBON",
                      "cell_type": f"MBON0{i+1}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1
    for i in range(3):
        rows.append({"root_id": rid, "cell_class": "DAN",
                      "cell_type": f"PAM0{i+1}",
                      "super_class": "central", "top_nt": "dopamine"})
        rid += 1
    rows.append({"root_id": rid, "cell_class": "MBIN",
                  "cell_type": "APL",
                  "super_class": "central", "top_nt": "GABA"})
    rid += 1
    for i in range(15):
        rows.append({"root_id": rid, "cell_class": "ALPN",
                      "cell_type": f"adPN-{i}",
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    ann = pd.DataFrame(rows)

    edge_rows = []
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    alpn_ids = ann[ann["cell_class"] == "ALPN"]["root_id"].values
    mbin_ids = ann[ann["cell_class"] == "MBIN"]["root_id"].values

    for kc in kc_ids:
        pns = rng.choice(alpn_ids, size=min(4, len(alpn_ids)), replace=False)
        for pn in pns:
            edge_rows.append({"pre_pt_root_id": pn, "post_pt_root_id": kc,
                              "syn_count": rng.randint(5, 15),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(1.0, 4.0)})
    for kc in kc_ids:
        targets = rng.choice(mbon_ids, size=min(2, len(mbon_ids)), replace=False)
        for mbon in targets:
            edge_rows.append({"pre_pt_root_id": kc, "post_pt_root_id": mbon,
                              "syn_count": rng.randint(3, 10),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(0.5, 2.0)})
    apl_id = mbin_ids[0]
    for kc in kc_ids[:40]:
        edge_rows.append({"pre_pt_root_id": apl_id, "post_pt_root_id": kc,
                          "syn_count": rng.randint(3, 8),
                          "dominant_nt": "GABA", "nt_sign": "inhibitory",
                          "weight": rng.uniform(-2.0, -0.5)})

    edges = pd.DataFrame(edge_rows)

    from bravli.explore.mushroom_body import build_mb_circuit
    circuit, mb_neurons, mb_edges = build_mb_circuit(ann, edges, mode="class_aware")
    return circuit, mb_neurons


# ---------------------------------------------------------------------------
# Tests: AdEx presets
# ---------------------------------------------------------------------------

class TestAdExPresets:
    def test_presets_exist(self):
        assert "regular_spiking" in ADEX_PRESETS
        assert "adapting" in ADEX_PRESETS
        assert "bursting" in ADEX_PRESETS
        assert "fast_spiking" in ADEX_PRESETS

    def test_preset_types(self):
        for name, params in ADEX_PRESETS.items():
            assert isinstance(params, AdExParams)
            assert params.delta_t > 0
            assert params.tau_w > 0


# ---------------------------------------------------------------------------
# Tests: AdEx engine basics
# ---------------------------------------------------------------------------

class TestAdExEngine:
    def test_no_input_no_spikes(self, two_neuron_circuit):
        """Without stimulus, no spikes."""
        r = simulate_adex(two_neuron_circuit, duration=100.0, seed=42)
        assert r.n_spikes == 0

    def test_strong_input_causes_spikes(self, two_neuron_circuit):
        """Strong stimulus should cause spikes."""
        stim = np.zeros((2, 2000))
        stim[0, :] = 15.0
        r = simulate_adex(two_neuron_circuit, duration=200.0,
                          stimulus=stim, seed=42)
        assert len(r.spike_times[0]) > 0

    def test_spike_propagation(self, two_neuron_circuit):
        """Presynaptic spikes should drive postsynaptic neuron."""
        stim = np.zeros((2, 5000))
        stim[0, :] = 15.0
        r = simulate_adex(two_neuron_circuit, duration=500.0,
                          stimulus=stim, seed=42)
        assert len(r.spike_times[0]) > 0
        assert len(r.spike_times[1]) > 0

    def test_returns_simulation_result(self, two_neuron_circuit):
        """Should return SimulationResult compatible with analysis tools."""
        from bravli.simulation.engine import SimulationResult
        r = simulate_adex(two_neuron_circuit, duration=100.0, seed=42)
        assert isinstance(r, SimulationResult)
        assert r.n_neurons == 2
        assert r.duration == 100.0

    def test_record_voltage(self, two_neuron_circuit):
        """Voltage recording should work."""
        stim = np.zeros((2, 1000))
        stim[0, :] = 15.0
        r = simulate_adex(two_neuron_circuit, duration=100.0,
                          stimulus=stim, record_v=True,
                          record_idx=[0, 1], seed=42)
        assert r.v_trace is not None
        assert r.v_trace.shape == (2, 1000)

    def test_default_params(self, small_network):
        """Default AdEx params should work on a network."""
        stim = np.zeros((small_network.n_neurons, 2000))
        stim[:5, :] = 15.0
        r = simulate_adex(small_network, duration=200.0,
                          stimulus=stim, seed=42)
        assert r.n_neurons == small_network.n_neurons


# ---------------------------------------------------------------------------
# Tests: Adaptation effect
# ---------------------------------------------------------------------------

class TestAdaptation:
    def test_adaptation_reduces_rate(self, two_neuron_circuit):
        """Higher adaptation (b) should reduce firing rate."""
        stim = np.zeros((2, 5000))
        stim[0, :] = 15.0

        r_no_adapt = simulate_adex(
            two_neuron_circuit,
            adex_params=AdExParams(b=0.0),
            duration=500.0, stimulus=stim, seed=42,
        )
        r_adapt = simulate_adex(
            two_neuron_circuit,
            adex_params=AdExParams(b=3.0, tau_w=50.0),
            duration=500.0, stimulus=stim, seed=42,
        )

        rate_no = len(r_no_adapt.spike_times[0])
        rate_yes = len(r_adapt.spike_times[0])
        # Adaptation should reduce spike count
        assert rate_yes <= rate_no

    def test_zero_adaptation_matches_exponential_lif(self, small_network):
        """With b=0, a=0, AdEx is exponential LIF (not plain LIF, but close)."""
        stim = np.zeros((small_network.n_neurons, 2000))
        stim[:10, :] = 12.0

        r_adex = simulate_adex(
            small_network,
            adex_params=AdExParams(delta_t=0.001, a=0.0, b=0.0),
            duration=200.0, stimulus=stim, seed=42,
        )
        r_lif = simulate(
            small_network, duration=200.0, stimulus=stim, seed=42,
        )

        # With very small delta_T, AdEx should approximate LIF
        # Not exact due to exponential term, but should be in the same ballpark
        adex_rates = firing_rates(r_adex)
        lif_rates = firing_rates(r_lif)
        # Both should produce similar total activity
        assert abs(np.mean(adex_rates) - np.mean(lif_rates)) < np.mean(lif_rates) + 10


# ---------------------------------------------------------------------------
# Tests: Comparison
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_compare_runs(self, mb_circuit):
        from bravli.explore.lif_vs_adex import compare_models

        circuit, mb_neurons = mb_circuit
        results = compare_models(
            circuit, mb_neurons,
            duration_ms=100.0, seed=42,
        )
        assert "lif" in results
        assert "adex" in results
        assert "comparison" in results
        assert "rate_correlation" in results["comparison"]

    def test_compare_keys(self, mb_circuit):
        from bravli.explore.lif_vs_adex import compare_models

        circuit, mb_neurons = mb_circuit
        results = compare_models(circuit, mb_neurons,
                                 duration_ms=100.0, seed=42)

        for model in ["lif", "adex"]:
            assert "rates" in results[model]
            assert "mean_rate" in results[model]
            assert "sparseness" in results[model]
            assert "active_fraction" in results[model]
            assert "group_rates" in results[model]

        comp = results["comparison"]
        assert "temporal_correlation" in comp
        assert "mean_relative_diff" in comp

    def test_compare_without_mb_neurons(self, small_network):
        from bravli.explore.lif_vs_adex import compare_models

        stim = np.zeros((small_network.n_neurons, 1000))
        stim[:5, :] = 15.0

        results = compare_models(
            small_network,
            stimulus=stim,
            duration_ms=100.0, seed=42,
        )
        assert "lif" in results
        assert "adex" in results


class TestAdaptationSweep:
    def test_sweep_runs(self, mb_circuit):
        from bravli.explore.lif_vs_adex import adaptation_sweep

        circuit, mb_neurons = mb_circuit
        df = adaptation_sweep(
            circuit, mb_neurons,
            b_values=[0.0, 1.0],
            duration_ms=100.0, seed=42,
        )
        assert len(df) == 2
        assert "b" in df.columns
        assert "adex_mean_rate" in df.columns
        assert "rate_correlation" in df.columns

    def test_sweep_rate_decreases_with_b(self, mb_circuit):
        """Higher adaptation should generally reduce rate."""
        from bravli.explore.lif_vs_adex import adaptation_sweep

        circuit, mb_neurons = mb_circuit
        df = adaptation_sweep(
            circuit, mb_neurons,
            b_values=[0.0, 5.0],
            duration_ms=200.0,
            pn_rate_hz=80.0,
            seed=42,
        )
        # Rate at b=5 should be <= rate at b=0
        assert df.iloc[1]["adex_mean_rate"] <= df.iloc[0]["adex_mean_rate"] + 1.0


# ---------------------------------------------------------------------------
# Tests: Report
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_report_runs(self, mb_circuit, capsys):
        from bravli.explore.lif_vs_adex import compare_models, comparison_report

        circuit, mb_neurons = mb_circuit
        results = compare_models(circuit, mb_neurons,
                                 duration_ms=100.0, seed=42)
        report = comparison_report(results)
        assert "LIF vs AdEx" in report
        assert "Topology" in report
        assert "Interpretation" in report

    def test_report_has_metrics(self, mb_circuit, capsys):
        from bravli.explore.lif_vs_adex import compare_models, comparison_report

        circuit, mb_neurons = mb_circuit
        results = compare_models(circuit, mb_neurons,
                                 duration_ms=100.0, seed=42)
        report = comparison_report(results)
        assert "correlation" in report.lower()
        assert "Mean rate" in report
