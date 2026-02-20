"""Tests for Brunel phase diagram and network regime classification.

Tests Investigation 4: random LIF network construction, regime
classification metrics, phase sweep, and FlyWire regime comparison.
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.engine import simulate, SimulationResult
from bravli.simulation.analysis import firing_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_brunel():
    """Build a small Brunel network for fast tests."""
    from bravli.explore.brunel_network import build_brunel_network
    circuit, params = build_brunel_network(
        n_excitatory=200, g=4.0, eta=2.0, seed=42
    )
    return circuit, params


@pytest.fixture
def mb_circuit():
    """Small MB circuit for FlyWire regime test."""
    rng = np.random.RandomState(42)
    rows = []
    rid = 100000
    for i in range(100):
        rows.append({"root_id": rid, "cell_class": "Kenyon_Cell",
                      "cell_type": rng.choice(["KCg-m", "KCab"]),
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
                      "super_class": "central", "top_nt": "acetylcholine"})
        rid += 1

    ann = pd.DataFrame(rows)

    edge_rows = []
    kc_ids = ann[ann["cell_class"] == "Kenyon_Cell"]["root_id"].values
    mbon_ids = ann[ann["cell_class"] == "MBON"]["root_id"].values
    alpn_ids = ann[ann["cell_class"] == "ALPN"]["root_id"].values
    mbin_ids = ann[ann["cell_class"] == "MBIN"]["root_id"].values

    for kc in kc_ids:
        pns = rng.choice(alpn_ids, size=min(5, len(alpn_ids)), replace=False)
        for pn in pns:
            edge_rows.append({"pre_pt_root_id": pn, "post_pt_root_id": kc,
                              "syn_count": rng.randint(5, 20),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(1.0, 5.0)})
    for kc in kc_ids:
        targets = rng.choice(mbon_ids, size=min(2, len(mbon_ids)), replace=False)
        for mbon in targets:
            edge_rows.append({"pre_pt_root_id": kc, "post_pt_root_id": mbon,
                              "syn_count": rng.randint(5, 15),
                              "dominant_nt": "acetylcholine", "nt_sign": "excitatory",
                              "weight": rng.uniform(0.5, 3.0)})
    apl_id = mbin_ids[0]
    for kc in kc_ids[:80]:
        edge_rows.append({"pre_pt_root_id": apl_id, "post_pt_root_id": kc,
                          "syn_count": rng.randint(5, 10),
                          "dominant_nt": "GABA", "nt_sign": "inhibitory",
                          "weight": rng.uniform(-3.0, -1.0)})

    edges = pd.DataFrame(edge_rows)

    from bravli.explore.mushroom_body import build_mb_circuit
    circuit, mb_neurons, mb_edges = build_mb_circuit(ann, edges, mode="class_aware")
    return circuit, mb_neurons


# ---------------------------------------------------------------------------
# Tests: Network construction
# ---------------------------------------------------------------------------

class TestBrunelNetwork:
    def test_build_network(self, small_brunel):
        circuit, params = small_brunel
        assert circuit.n_neurons == 250  # 200 E + 50 I
        assert circuit.n_synapses > 0
        assert params["n_excitatory"] == 200
        assert params["n_inhibitory"] == 50

    def test_ei_ratio(self, small_brunel):
        circuit, params = small_brunel
        n_e = params["n_excitatory"]
        n_i = params["n_inhibitory"]
        assert n_i / n_e == pytest.approx(0.25)

    def test_weight_signs(self, small_brunel):
        circuit, params = small_brunel
        # Excitatory weights should be positive
        assert np.any(circuit.weights > 0)
        # Inhibitory weights should be negative
        assert np.any(circuit.weights < 0)

    def test_weight_ratio(self, small_brunel):
        circuit, params = small_brunel
        g = params["g"]
        j_eff = params["j_eff"]
        exc_w = circuit.weights[circuit.weights > 0]
        inh_w = circuit.weights[circuit.weights < 0]
        assert np.all(np.isclose(exc_w, j_eff))
        assert np.all(np.isclose(inh_w, -g * j_eff))

    def test_connection_count(self, small_brunel):
        circuit, params = small_brunel
        c_e = params["c_e"]
        c_i = params["c_i"]
        # Each neuron should have c_e + c_i incoming connections
        expected = circuit.n_neurons * (c_e + c_i)
        assert circuit.n_synapses == expected

    def test_nu_thr_formula(self, small_brunel):
        _, params = small_brunel
        expected = (params["v_thresh"] - params["v_rest"]) / (
            params["j"] * params["c_e"] * params["tau_m"]
        )
        assert params["nu_thr"] == pytest.approx(expected)

    def test_nu_ext_formula(self, small_brunel):
        _, params = small_brunel
        assert params["nu_ext"] == pytest.approx(
            params["eta"] * params["nu_thr"]
        )


class TestBrunelStimulus:
    def test_stimulus_shape(self, small_brunel):
        from bravli.explore.brunel_network import build_brunel_stimulus
        circuit, params = small_brunel
        stim = build_brunel_stimulus(circuit, params, duration_ms=100.0, dt=0.1)
        assert stim.shape == (250, 1000)

    def test_stimulus_nonnegative(self, small_brunel):
        from bravli.explore.brunel_network import build_brunel_stimulus
        circuit, params = small_brunel
        stim = build_brunel_stimulus(circuit, params, duration_ms=100.0)
        assert np.all(stim >= 0)

    def test_stimulus_has_nonzero(self, small_brunel):
        from bravli.explore.brunel_network import build_brunel_stimulus
        circuit, params = small_brunel
        stim = build_brunel_stimulus(circuit, params, duration_ms=100.0)
        assert np.any(stim > 0)


# ---------------------------------------------------------------------------
# Tests: Simulation produces spikes
# ---------------------------------------------------------------------------

class TestBrunelSimulation:
    def test_produces_spikes(self, small_brunel):
        from bravli.explore.brunel_network import build_brunel_stimulus
        circuit, params = small_brunel
        stim = build_brunel_stimulus(circuit, params, duration_ms=200.0, seed=42)
        result = simulate(circuit, duration=200.0, dt=0.1, stimulus=stim, seed=42)
        assert result.n_spikes > 0

    def test_rate_depends_on_eta(self):
        from bravli.explore.brunel_network import (
            build_brunel_network, build_brunel_stimulus,
        )
        rates = []
        for eta in [1.0, 2.0, 3.0]:
            circuit, params = build_brunel_network(
                n_excitatory=200, g=4.0, eta=eta, seed=42
            )
            stim = build_brunel_stimulus(circuit, params, duration_ms=200.0, seed=42)
            result = simulate(circuit, duration=200.0, dt=0.1, stimulus=stim, seed=42)
            rates.append(result.mean_rate())
        # Higher eta should give higher rate
        assert rates[2] > rates[0]


# ---------------------------------------------------------------------------
# Tests: Regime classification
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_classification_keys(self, small_brunel):
        from bravli.explore.brunel_network import build_brunel_stimulus, classify_regime
        circuit, params = small_brunel
        stim = build_brunel_stimulus(circuit, params, duration_ms=200.0, seed=42)
        result = simulate(circuit, duration=200.0, dt=0.1, stimulus=stim, seed=42)
        c = classify_regime(result)
        assert "regime" in c
        assert "cv_isi" in c
        assert "synchrony" in c
        assert "mean_rate" in c
        assert c["regime"] in ("SR", "SI", "AR", "AI", "quiescent")

    def test_cv_increases_with_g(self):
        """Stronger inhibition should increase CV (more irregular firing)."""
        from bravli.explore.brunel_network import (
            build_brunel_network, build_brunel_stimulus, classify_regime,
        )
        cvs = []
        for g in [3.0, 6.0]:
            circuit, params = build_brunel_network(
                n_excitatory=500, g=g, eta=2.0, seed=42
            )
            stim = build_brunel_stimulus(circuit, params, duration_ms=300.0, seed=42)
            result = simulate(circuit, duration=300.0, dt=0.1, stimulus=stim, seed=42)
            c = classify_regime(result)
            cvs.append(c["cv_isi"])
        # g=6 should have higher CV than g=3
        assert cvs[1] > cvs[0]

    def test_quiescent_regime(self):
        """Below-threshold drive should produce quiescent state."""
        from bravli.explore.brunel_network import (
            build_brunel_network, build_brunel_stimulus, classify_regime,
        )
        circuit, params = build_brunel_network(
            n_excitatory=200, g=8.0, eta=0.5, seed=42
        )
        stim = build_brunel_stimulus(circuit, params, duration_ms=200.0, seed=42)
        result = simulate(circuit, duration=200.0, dt=0.1, stimulus=stim, seed=42)
        c = classify_regime(result)
        assert c["mean_rate"] < 5.0  # very low rate


# ---------------------------------------------------------------------------
# Tests: Phase sweep
# ---------------------------------------------------------------------------

class TestPhaseSweep:
    def test_sweep_runs(self):
        from bravli.explore.brunel_network import brunel_phase_sweep
        df = brunel_phase_sweep(
            g_values=[3.0, 5.0],
            eta_values=[1.5, 2.5],
            n_excitatory=200,
            duration_ms=200.0,
            seed=42,
        )
        assert len(df) == 4
        assert "regime" in df.columns
        assert "g" in df.columns
        assert "eta" in df.columns

    def test_sweep_has_all_points(self):
        from bravli.explore.brunel_network import brunel_phase_sweep
        df = brunel_phase_sweep(
            g_values=[3.0, 6.0],
            eta_values=[1.0, 2.0, 3.0],
            n_excitatory=200,
            duration_ms=200.0,
            seed=42,
        )
        assert len(df) == 6
        assert set(df["g"]) == {3.0, 6.0}
        assert set(df["eta"]) == {1.0, 2.0, 3.0}


# ---------------------------------------------------------------------------
# Tests: FlyWire regime classification
# ---------------------------------------------------------------------------

class TestFlyWireRegime:
    def test_classify_flywire(self, mb_circuit):
        from bravli.explore.brunel_network import classify_flywire_regime
        circuit, mb_neurons = mb_circuit
        result = classify_flywire_regime(
            circuit, mb_neurons,
            duration_ms=200.0, pn_rate_hz=50.0, seed=42,
        )
        assert "regime" in result
        assert "effective_g" in result
        assert result["effective_g"] >= 0
        assert result["regime"] in ("SR", "SI", "AR", "AI", "quiescent")


# ---------------------------------------------------------------------------
# Tests: Report
# ---------------------------------------------------------------------------

class TestBrunelReport:
    def test_report_runs(self, capsys):
        from bravli.explore.brunel_network import brunel_phase_sweep, brunel_report
        df = brunel_phase_sweep(
            g_values=[3.0, 5.0],
            eta_values=[2.0],
            n_excitatory=200,
            duration_ms=200.0,
            seed=42,
        )
        report = brunel_report(df)
        assert "BRUNEL PHASE DIAGRAM" in report

    def test_report_with_flywire(self, mb_circuit, capsys):
        from bravli.explore.brunel_network import (
            brunel_phase_sweep, classify_flywire_regime, brunel_report,
        )
        circuit, mb_neurons = mb_circuit
        df = brunel_phase_sweep(
            g_values=[4.0],
            eta_values=[2.0],
            n_excitatory=200,
            duration_ms=200.0,
            seed=42,
        )
        fw = classify_flywire_regime(
            circuit, mb_neurons, duration_ms=200.0, seed=42,
        )
        report = brunel_report(df, flywire_result=fw)
        assert "FlyWire" in report
        assert "Interpretation" in report
