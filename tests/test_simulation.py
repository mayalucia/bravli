"""Tests for the simulation module.

Uses small synthetic circuits to verify correctness of the LIF engine,
circuit assembly, stimulus generation, and analysis tools.
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit, build_circuit, build_circuit_from_edges
from bravli.simulation.engine import simulate, SimulationResult
from bravli.simulation.stimulus import (
    poisson_stimulus, step_stimulus, pulse_stimulus, combine_stimuli,
)
from bravli.simulation.analysis import (
    firing_rates, spike_raster, ei_balance, active_fraction, population_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        weights=np.array([10.0]),  # strong excitatory
        tau_syn=5.0,
        delay_steps=18,
    )


@pytest.fixture
def ten_neuron_edges():
    """Synthetic edge table for 10 neurons."""
    rng = np.random.RandomState(42)
    n_edges = 30
    pre = rng.randint(0, 10, n_edges)
    post = rng.randint(0, 10, n_edges)
    # Remove self-connections
    mask = pre != post
    pre, post = pre[mask], post[mask]
    weights = rng.uniform(-1.0, 1.0, len(pre))

    return pd.DataFrame({
        "pre_pt_root_id": pre,
        "post_pt_root_id": post,
        "weight": weights,
    })


@pytest.fixture
def ten_neuron_annotations():
    """Neuron annotations for 10 neurons."""
    return pd.DataFrame({
        "root_id": range(10),
        "v_rest": -52.0,
        "v_thresh": -45.0,
        "v_reset": -52.0,
        "tau_m": 20.0,
        "t_ref": 2.2,
        "model_mode": "spiking",
    })


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

class TestCircuit:
    def test_two_neuron_properties(self, two_neuron_circuit):
        c = two_neuron_circuit
        assert c.n_neurons == 2
        assert c.n_synapses == 1
        assert not c.is_heterogeneous

    def test_summary(self, two_neuron_circuit):
        s = two_neuron_circuit.summary()
        assert "2 neurons" in s
        assert "1 synapses" in s

    def test_build_from_edges(self, ten_neuron_edges):
        c = build_circuit_from_edges(ten_neuron_edges)
        assert c.n_neurons == 10
        assert c.n_synapses == len(ten_neuron_edges)

    def test_build_from_neurons_and_edges(self, ten_neuron_annotations, ten_neuron_edges):
        c = build_circuit(ten_neuron_annotations, ten_neuron_edges)
        assert c.n_neurons == 10
        assert len(c.id_to_idx) == 10


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestEngine:
    def test_no_input_no_spikes(self, two_neuron_circuit):
        """Without stimulus, neurons at rest should not spike."""
        result = simulate(two_neuron_circuit, duration=100.0, dt=0.1)
        assert result.n_spikes == 0
        assert result.n_neurons == 2
        assert result.duration == 100.0

    def test_strong_step_causes_spikes(self, two_neuron_circuit):
        """A strong step current should drive neuron 0 to spike."""
        n = two_neuron_circuit.n_neurons
        n_steps = int(100.0 / 0.1)
        stim, _ = step_stimulus(n, n_steps, target_indices=[0],
                                amplitude=15.0, start_ms=10.0, end_ms=90.0)
        result = simulate(two_neuron_circuit, duration=100.0, dt=0.1,
                         stimulus=stim)
        # Neuron 0 should spike
        assert len(result.spike_times[0]) > 0

    def test_spike_propagation(self, two_neuron_circuit):
        """Spikes from neuron 0 should propagate to neuron 1."""
        n = two_neuron_circuit.n_neurons
        n_steps = int(200.0 / 0.1)
        stim, _ = step_stimulus(n, n_steps, target_indices=[0],
                                amplitude=15.0, start_ms=10.0, end_ms=190.0)
        result = simulate(two_neuron_circuit, duration=200.0, dt=0.1,
                         stimulus=stim)
        # Both neurons should spike
        assert len(result.spike_times[0]) > 0
        assert len(result.spike_times[1]) > 0
        # Neuron 1 should spike after neuron 0 (with delay)
        if len(result.spike_times[1]) > 0:
            assert result.spike_times[1][0] > result.spike_times[0][0]

    def test_record_voltage(self, two_neuron_circuit):
        """Voltage recording should return traces."""
        result = simulate(two_neuron_circuit, duration=50.0, dt=0.1,
                         record_v=True, record_idx=[0, 1])
        assert result.v_trace is not None
        assert result.v_trace.shape == (2, 500)
        assert result.g_trace is not None
        # At rest, voltage should be near v_rest
        assert abs(result.v_trace[0, 0] - (-52.0)) < 0.1

    def test_refractory_period(self):
        """After a spike, neuron should not fire during refractory."""
        # Single neuron, no connections
        c = Circuit(
            n_neurons=1,
            v_rest=np.array([-52.0]),
            v_thresh=np.array([-45.0]),
            v_reset=np.array([-52.0]),
            tau_m=np.array([20.0]),
            t_ref=np.array([10.0]),  # long refractory
            pre_idx=np.array([], dtype=np.int32),
            post_idx=np.array([], dtype=np.int32),
            weights=np.array([]),
            tau_syn=5.0,
            delay_steps=18,
        )
        n_steps = int(100.0 / 0.1)
        stim, _ = step_stimulus(1, n_steps, [0], amplitude=20.0,
                                start_ms=0.0, end_ms=100.0)
        result = simulate(c, duration=100.0, dt=0.1, stimulus=stim)
        # Should spike but with ~10ms gap between spikes
        if len(result.spike_times[0]) >= 2:
            isi = np.diff(result.spike_times[0])
            assert np.all(isi >= 9.9)  # refractory enforced

    def test_mean_rate(self, two_neuron_circuit):
        result = simulate(two_neuron_circuit, duration=100.0, dt=0.1)
        assert result.mean_rate() == 0.0  # no stimulus, no spikes

    def test_heterogeneous_params(self):
        """Neurons with different tau_m should fire at different rates."""
        c = Circuit(
            n_neurons=2,
            v_rest=np.array([-52.0, -52.0]),
            v_thresh=np.array([-45.0, -45.0]),
            v_reset=np.array([-52.0, -52.0]),
            tau_m=np.array([5.0, 40.0]),  # fast vs slow
            t_ref=np.array([2.0, 2.0]),
            pre_idx=np.array([], dtype=np.int32),
            post_idx=np.array([], dtype=np.int32),
            weights=np.array([]),
            tau_syn=5.0,
            delay_steps=18,
        )
        n_steps = int(500.0 / 0.1)
        stim, _ = step_stimulus(2, n_steps, [0, 1], amplitude=15.0,
                                start_ms=10.0, end_ms=490.0)
        result = simulate(c, duration=500.0, dt=0.1, stimulus=stim)
        # Fast neuron (tau_m=5) should fire more than slow (tau_m=40)
        assert len(result.spike_times[0]) > len(result.spike_times[1])


# ---------------------------------------------------------------------------
# Stimulus tests
# ---------------------------------------------------------------------------

class TestStimulus:
    def test_poisson_shape(self):
        stim, prot = poisson_stimulus(10, 1000, [0, 1, 2], rate_hz=100.0, seed=42)
        assert stim.shape == (10, 1000)
        assert prot.name == "poisson"
        # Non-target neurons should be zero
        assert np.all(stim[5] == 0.0)
        # Target neurons should have some non-zero entries
        assert np.any(stim[0] > 0)

    def test_step_shape(self):
        stim, prot = step_stimulus(5, 1000, [1], amplitude=10.0,
                                   start_ms=10.0, end_ms=50.0, dt=0.1)
        assert stim.shape == (5, 1000)
        assert prot.name == "step"
        # Before start: zero
        assert stim[1, 0] == 0.0
        # During step: amplitude
        assert stim[1, 200] == 10.0
        # After end: zero
        assert stim[1, 600] == 0.0

    def test_pulse_shape(self):
        stim, prot = pulse_stimulus(3, 1000, [0], amplitude=20.0,
                                    pulse_ms=1.0, time_ms=50.0, dt=0.1)
        assert stim.shape == (3, 1000)
        assert prot.name == "pulse"
        # At pulse time
        assert stim[0, 500] == 20.0
        # After pulse
        assert stim[0, 520] == 0.0

    def test_combine(self):
        s1 = np.ones((3, 100))
        s2 = np.ones((3, 100)) * 2
        combined = combine_stimuli(s1, s2)
        assert np.allclose(combined, 3.0)

    def test_poisson_rate_approximate(self):
        """Poisson stimulus should produce approximately the right rate."""
        stim, _ = poisson_stimulus(1, 100000, [0], rate_hz=100.0,
                                   weight=1.0, dt=0.1, seed=123)
        # Expected spikes: 100 Hz * 10 s = 1000 spikes
        n_spikes = np.sum(stim[0] > 0)
        assert 800 < n_spikes < 1200


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestAnalysis:
    def _make_result(self):
        """Create a synthetic result with known spikes."""
        spike_times = [
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # 5 spikes in 100ms = 50Hz
            np.array([15.0, 45.0]),  # 2 spikes = 20Hz
            np.array([]),  # silent
        ]
        return SimulationResult(
            spike_times=spike_times,
            dt=0.1,
            duration=100.0,
            n_neurons=3,
        )

    def test_firing_rates(self):
        result = self._make_result()
        rates = firing_rates(result)
        assert abs(rates[0] - 50.0) < 0.1
        assert abs(rates[1] - 20.0) < 0.1
        assert rates[2] == 0.0

    def test_firing_rates_windowed(self):
        result = self._make_result()
        rates = firing_rates(result, time_window=(0.0, 50.0))
        # Neuron 0: 4 spikes in 50ms = 80 Hz
        assert abs(rates[0] - 80.0) < 0.1

    def test_spike_raster(self):
        result = self._make_result()
        times, neurons = spike_raster(result)
        assert len(times) == 7  # 5 + 2 + 0
        assert len(neurons) == 7

    def test_spike_raster_subset(self):
        result = self._make_result()
        times, neurons = spike_raster(result, neuron_indices=[0])
        assert len(times) == 5

    def test_active_fraction(self):
        result = self._make_result()
        frac = active_fraction(result, threshold_hz=1.0)
        assert abs(frac - 2.0 / 3.0) < 0.01

    def test_population_rate(self):
        result = self._make_result()
        times, rates = population_rate(result, bin_ms=50.0)
        assert len(times) == 2
        assert np.all(rates >= 0)

    def test_ei_balance(self, two_neuron_circuit):
        """EI balance on a silent network should be all zeros."""
        result = simulate(two_neuron_circuit, duration=50.0, dt=0.1)
        balance = ei_balance(result, two_neuron_circuit)
        assert "exc_input" in balance.columns
        assert "inh_input" in balance.columns
        assert all(balance["firing_rate_hz"] == 0.0)
