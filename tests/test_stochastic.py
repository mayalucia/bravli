"""Tests for stochastic synapses (Investigation 9).

Tests synaptic release failure, intrinsic noise, stochastic resonance,
and MB-specific noise effects.
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit
from bravli.simulation.engine import simulate
from bravli.simulation.stimulus import step_stimulus
from bravli.simulation.analysis import firing_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_neuron_circuit():
    """Simple two-neuron circuit: neuron 0 drives neuron 1."""
    return Circuit(
        n_neurons=2,
        v_rest=np.array([-55.0, -55.0]),
        v_thresh=np.array([-45.0, -45.0]),
        v_reset=np.array([-55.0, -55.0]),
        tau_m=np.array([20.0, 20.0]),
        t_ref=np.array([2.0, 2.0]),
        pre_idx=np.array([0], dtype=np.int32),
        post_idx=np.array([1], dtype=np.int32),
        weights=np.array([5.0]),
        tau_syn=5.0,
        delay_steps=1,
    )


@pytest.fixture
def small_network():
    """Small random network for noise tests."""
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
    """Small MB circuit for stochastic tests."""
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
# Tests: Engine noise parameters
# ---------------------------------------------------------------------------

class TestEngineNoise:
    def test_noise_sigma_zero_deterministic(self, two_neuron_circuit):
        """With noise_sigma=0, results should be deterministic."""
        stim = np.zeros((2, 1000))
        stim[0, :] = 15.0  # drive neuron 0

        r1 = simulate(two_neuron_circuit, duration=100.0, stimulus=stim,
                      noise_sigma=0.0, seed=42)
        r2 = simulate(two_neuron_circuit, duration=100.0, stimulus=stim,
                      noise_sigma=0.0, seed=42)

        assert r1.n_spikes == r2.n_spikes

    def test_noise_adds_variability(self, small_network):
        """Different seeds with noise should produce different results."""
        r1 = simulate(small_network, duration=200.0,
                      noise_sigma=5.0, seed=42)
        r2 = simulate(small_network, duration=200.0,
                      noise_sigma=5.0, seed=99)

        # Not exactly equal (stochastic)
        rates1 = firing_rates(r1)
        rates2 = firing_rates(r2)
        # At least some neurons should differ
        assert not np.allclose(rates1, rates2)

    def test_noise_can_cause_spikes(self, small_network):
        """Enough noise should cause spikes even without stimulus."""
        r = simulate(small_network, duration=500.0,
                     noise_sigma=15.0, seed=42)
        assert r.n_spikes > 0

    def test_no_noise_no_spontaneous(self, small_network):
        """Without noise or stimulus, no spikes."""
        r = simulate(small_network, duration=200.0,
                     noise_sigma=0.0, seed=42)
        assert r.n_spikes == 0


class TestEngineRelease:
    def test_release_prob_one_deterministic(self, two_neuron_circuit):
        """release_prob=1.0 should be identical to default."""
        stim = np.zeros((2, 2000))
        stim[0, :] = 15.0

        r_default = simulate(two_neuron_circuit, duration=200.0,
                             stimulus=stim, seed=42)
        r_prob1 = simulate(two_neuron_circuit, duration=200.0,
                           stimulus=stim, release_prob=1.0, seed=42)

        assert r_default.n_spikes == r_prob1.n_spikes

    def test_release_prob_zero_no_propagation(self, two_neuron_circuit):
        """release_prob=0 should prevent all synaptic transmission."""
        stim = np.zeros((2, 2000))
        stim[0, :] = 15.0  # drive only neuron 0

        r = simulate(two_neuron_circuit, duration=200.0,
                     stimulus=stim, release_prob=0.0, seed=42)

        # Neuron 0 should spike (directly driven)
        assert len(r.spike_times[0]) > 0
        # Neuron 1 should NOT spike (no transmission)
        assert len(r.spike_times[1]) == 0

    def test_low_release_fewer_spikes(self, two_neuron_circuit):
        """Lower release probability should reduce postsynaptic spikes."""
        stim = np.zeros((2, 5000))
        stim[0, :] = 15.0

        r_full = simulate(two_neuron_circuit, duration=500.0,
                          stimulus=stim, release_prob=1.0, seed=42)
        r_half = simulate(two_neuron_circuit, duration=500.0,
                          stimulus=stim, release_prob=0.3, seed=42)

        # Postsynaptic neuron should have fewer spikes at low release prob
        assert len(r_half.spike_times[1]) <= len(r_full.spike_times[1])

    def test_per_synapse_release_prob(self):
        """Per-synapse release probability array should work."""
        circuit = Circuit(
            n_neurons=3,
            v_rest=np.array([-55.0, -55.0, -55.0]),
            v_thresh=np.array([-45.0, -45.0, -45.0]),
            v_reset=np.array([-55.0, -55.0, -55.0]),
            tau_m=np.array([20.0, 20.0, 20.0]),
            t_ref=np.array([2.0, 2.0, 2.0]),
            pre_idx=np.array([0, 0], dtype=np.int32),
            post_idx=np.array([1, 2], dtype=np.int32),
            weights=np.array([15.0, 15.0]),
            tau_syn=5.0,
            delay_steps=1,
        )
        stim = np.zeros((3, 5000))
        stim[0, :] = 15.0

        # Synapse 0->1 has p=1.0, synapse 0->2 has p=0.0
        release_prob = np.array([1.0, 0.0])
        r = simulate(circuit, duration=500.0, stimulus=stim,
                     release_prob=release_prob, seed=42)

        assert len(r.spike_times[1]) > 0   # transmitted
        assert len(r.spike_times[2]) == 0   # blocked


# ---------------------------------------------------------------------------
# Tests: Noise sweep
# ---------------------------------------------------------------------------

class TestNoiseSweep:
    def test_sweep_runs(self, small_network):
        from bravli.explore.stochastic_synapses import noise_sweep

        df = noise_sweep(small_network,
                         noise_sigmas=[0.0, 5.0, 10.0],
                         duration_ms=100.0, seed=42)
        assert len(df) == 3
        assert "noise_sigma" in df.columns
        assert "mean_rate" in df.columns

    def test_rate_increases_with_noise(self, small_network):
        from bravli.explore.stochastic_synapses import noise_sweep

        df = noise_sweep(small_network,
                         noise_sigmas=[0.0, 20.0],
                         duration_ms=200.0, seed=42)
        # High noise should produce more spikes than no noise
        assert df.iloc[1]["mean_rate"] >= df.iloc[0]["mean_rate"]


class TestReleaseProbSweep:
    def test_sweep_runs(self, small_network):
        from bravli.explore.stochastic_synapses import release_prob_sweep

        stim = np.zeros((small_network.n_neurons, 2000))
        stim[:10, :] = 15.0  # drive first 10 neurons

        df = release_prob_sweep(small_network, stimulus=stim,
                                release_probs=[0.3, 1.0],
                                duration_ms=200.0, seed=42)
        assert len(df) == 2
        assert "release_prob" in df.columns


# ---------------------------------------------------------------------------
# Tests: Stochastic resonance
# ---------------------------------------------------------------------------

class TestStochasticResonance:
    def test_sr_runs(self, small_network):
        from bravli.explore.stochastic_synapses import stochastic_resonance_test

        df, info = stochastic_resonance_test(
            small_network,
            signal_indices=np.arange(5),
            signal_amplitude=3.0,
            noise_sigmas=[0.0, 5.0, 10.0],
            duration_ms=500.0,
            seed=42,
        )
        assert len(df) == 3
        assert "snr" in df.columns
        assert "signal_frequency_hz" in info
        assert "best_sigma" in info

    def test_sr_returns_info(self, small_network):
        from bravli.explore.stochastic_synapses import stochastic_resonance_test

        df, info = stochastic_resonance_test(
            small_network,
            signal_indices=np.arange(3),
            signal_amplitude=2.0,
            noise_sigmas=[0.0, 5.0],
            duration_ms=300.0,
            seed=42,
        )
        assert isinstance(info, dict)
        assert "has_resonance" in info


# ---------------------------------------------------------------------------
# Tests: MB stochastic experiment
# ---------------------------------------------------------------------------

class TestMBStochastic:
    def test_mb_experiment_runs(self, mb_circuit):
        from bravli.explore.stochastic_synapses import mb_stochastic_experiment

        circuit, mb_neurons = mb_circuit
        results = mb_stochastic_experiment(
            circuit, mb_neurons,
            noise_sigmas=[0.0, 5.0],
            release_probs=[0.5, 1.0],
            duration_ms=100.0, seed=42,
        )
        assert "noise_sweep" in results
        assert "release_sweep" in results
        assert len(results["noise_sweep"]) == 2
        assert len(results["release_sweep"]) == 2

    def test_mb_noise_sweep_has_kc_columns(self, mb_circuit):
        from bravli.explore.stochastic_synapses import mb_stochastic_experiment

        circuit, mb_neurons = mb_circuit
        results = mb_stochastic_experiment(
            circuit, mb_neurons,
            noise_sigmas=[0.0, 3.0],
            release_probs=[1.0],
            duration_ms=100.0, seed=42,
        )
        df = results["noise_sweep"]
        assert "kc_mean_rate" in df.columns
        assert "kc_active_frac" in df.columns


# ---------------------------------------------------------------------------
# Tests: Report
# ---------------------------------------------------------------------------

class TestStochasticReport:
    def test_report_with_noise(self, small_network):
        from bravli.explore.stochastic_synapses import noise_sweep, stochastic_report

        df = noise_sweep(small_network,
                         noise_sigmas=[0.0, 5.0],
                         duration_ms=100.0, seed=42)
        report = stochastic_report(noise_df=df)
        assert "STOCHASTIC SYNAPSES" in report
        assert "Intrinsic noise" in report

    def test_report_with_sr(self, small_network):
        from bravli.explore.stochastic_synapses import (
            stochastic_resonance_test, stochastic_report,
        )

        df, info = stochastic_resonance_test(
            small_network,
            signal_indices=np.arange(3),
            signal_amplitude=3.0,
            noise_sigmas=[0.0, 5.0],
            duration_ms=300.0, seed=42,
        )
        report = stochastic_report(sr_df=df, sr_info=info)
        assert "resonance" in report.lower()

    def test_report_with_mb(self, mb_circuit):
        from bravli.explore.stochastic_synapses import (
            mb_stochastic_experiment, stochastic_report,
        )

        circuit, mb_neurons = mb_circuit
        results = mb_stochastic_experiment(
            circuit, mb_neurons,
            noise_sigmas=[0.0, 3.0],
            release_probs=[0.5, 1.0],
            duration_ms=100.0, seed=42,
        )
        report = stochastic_report(mb_results=results)
        assert "MB circuit" in report
