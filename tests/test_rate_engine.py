"""Tests for the rate-based dynamics engine."""

import numpy as np
import pytest

from bravli.simulation.rate_engine import (
    RateCircuit, RateResult, simulate_rate,
    phi, phi_pv, phi_power,
)


# ---------------------------------------------------------------------------
# Transfer function tests
# ---------------------------------------------------------------------------

class TestTransferFunctions:

    def test_phi_rectification(self):
        assert phi(-5.0) == 0.0
        assert phi(0.0) == 0.0
        assert phi(3.0) == 3.0

    def test_phi_clipping(self):
        assert phi(25.0) == 20.0
        assert phi(100.0) == 20.0

    def test_phi_array(self):
        x = np.array([-2.0, 0.0, 5.0, 30.0])
        result = phi(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 5.0, 20.0])

    def test_phi_pv_quadratic(self):
        assert phi_pv(-1.0) == 0.0
        assert phi_pv(0.0) == 0.0
        assert phi_pv(2.0) == 4.0
        assert phi_pv(3.0) == 9.0

    def test_phi_pv_clipping(self):
        assert phi_pv(5.0) == 20.0  # 5^2 = 25, clipped to 20
        assert phi_pv(10.0) == 20.0

    def test_phi_power_with_exponent(self):
        assert phi_power(3.0, k=1.0) == 3.0
        assert phi_power(3.0, k=2.0) == 9.0
        assert phi_power(-1.0, k=2.0) == 0.0


# ---------------------------------------------------------------------------
# RateCircuit construction tests
# ---------------------------------------------------------------------------

class TestRateCircuit:

    def test_single_population(self):
        c = RateCircuit(
            n_populations=1,
            labels=["E"],
            tau=np.array([10.0]),
            transfer_fn=[phi],
            W=np.array([[0.0]]),
            bias=np.array([5.0]),
        )
        assert c.n_populations == 1
        assert c.population_index("E") == 0

    def test_two_population(self):
        c = RateCircuit(
            n_populations=2,
            labels=["E", "I"],
            tau=np.array([10.0, 5.0]),
            transfer_fn=[phi, phi],
            W=np.array([[0.0, -1.0],
                         [1.0, 0.0]]),
            bias=np.array([5.0, 0.0]),
        )
        assert c.population_index("I") == 1

    def test_mismatched_dimensions_raises(self):
        with pytest.raises(AssertionError):
            RateCircuit(
                n_populations=2,
                labels=["E"],  # wrong length
                tau=np.array([10.0, 5.0]),
                transfer_fn=[phi, phi],
                W=np.zeros((2, 2)),
                bias=np.zeros(2),
            )


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulateRate:

    def test_single_population_decay_to_bias(self):
        """A single population with constant bias should decay to φ(bias)."""
        c = RateCircuit(
            n_populations=1,
            labels=["E"],
            tau=np.array([10.0]),
            transfer_fn=[phi],
            W=np.array([[0.0]]),
            bias=np.array([5.0]),
        )
        result = simulate_rate(c, duration=200.0, dt=0.1)

        # Should converge to phi(5.0) = 5.0
        final_rate = result.rates[0, -1]
        assert abs(final_rate - 5.0) < 0.01

    def test_single_population_with_self_inhibition(self):
        """Self-connection changes steady state: r* = φ(W*r* + bias)."""
        # r* = phi(-0.5 * r* + 8) => r* = -0.5*r* + 8 => 1.5*r* = 8 => r* ≈ 5.33
        c = RateCircuit(
            n_populations=1,
            labels=["E"],
            tau=np.array([10.0]),
            transfer_fn=[phi],
            W=np.array([[-0.5]]),
            bias=np.array([8.0]),
        )
        result = simulate_rate(c, duration=500.0, dt=0.1)
        final = result.rates[0, -1]
        expected = 8.0 / 1.5
        assert abs(final - expected) < 0.05

    def test_two_population_ei_balance(self):
        """E-I pair reaches steady state."""
        c = RateCircuit(
            n_populations=2,
            labels=["E", "I"],
            tau=np.array([10.0, 10.0]),
            transfer_fn=[phi, phi],
            W=np.array([[0.0, -2.0],
                         [1.0, 0.0]]),
            bias=np.array([10.0, 0.0]),
        )
        result = simulate_rate(c, duration=500.0, dt=0.1)

        # Both should reach positive steady state
        r_e = result.rates[0, -1]
        r_i = result.rates[1, -1]
        assert r_e > 0
        assert r_i > 0

        # At steady state: r_E = phi(-2*r_I + 10), r_I = phi(r_E)
        # => r_I = r_E, r_E = -2*r_E + 10 => 3*r_E = 10 => r_E ≈ 3.33
        assert abs(r_e - 10.0 / 3.0) < 0.05

    def test_divisive_normalization(self):
        """Divisive connection reduces output proportionally."""
        # Population 0 receives additive bias=10, divisively normalized by pop 1
        c = RateCircuit(
            n_populations=2,
            labels=["target", "divisor"],
            tau=np.array([10.0, 10.0]),
            transfer_fn=[phi, phi],
            W=np.array([[0.0, 0.0],
                         [0.0, 0.0]]),
            bias=np.array([10.0, 4.0]),
            divisive={0: (1, 1.0, 1.0)},  # target / (1.0 + 1.0 * r_divisor)
        )
        result = simulate_rate(c, duration=500.0, dt=0.1)

        # divisor -> phi(4) = 4, so target -> phi(10 / (1 + 4)) = phi(2) = 2
        assert abs(result.rates[0, -1] - 2.0) < 0.05
        assert abs(result.rates[1, -1] - 4.0) < 0.05

    def test_stimulus_injection(self):
        """External stimulus shifts the steady state."""
        c = RateCircuit(
            n_populations=1,
            labels=["E"],
            tau=np.array([10.0]),
            transfer_fn=[phi],
            W=np.array([[0.0]]),
            bias=np.array([0.0]),
        )
        n_steps = int(200.0 / 0.1)
        stim = np.full((1, n_steps), 7.0)
        result = simulate_rate(c, duration=200.0, dt=0.1, stimulus=stim)
        assert abs(result.rates[0, -1] - 7.0) < 0.05

    def test_result_accessors(self):
        """RateResult label-based accessors work."""
        c = RateCircuit(
            n_populations=2,
            labels=["E", "I"],
            tau=np.array([10.0, 10.0]),
            transfer_fn=[phi, phi],
            W=np.zeros((2, 2)),
            bias=np.array([3.0, 5.0]),
        )
        result = simulate_rate(c, duration=200.0, dt=0.1)

        assert result.n_populations == 2
        assert len(result.rate("E")) == result.n_steps
        assert abs(result.mean_rate("I", start=150.0) - 5.0) < 0.1

    def test_euler_small_dt_matches_large_dt(self):
        """Smaller dt should give similar steady state."""
        def make_circuit():
            return RateCircuit(
                n_populations=1,
                labels=["E"],
                tau=np.array([10.0]),
                transfer_fn=[phi],
                W=np.array([[0.0]]),
                bias=np.array([5.0]),
            )
        r1 = simulate_rate(make_circuit(), duration=200.0, dt=1.0)
        r2 = simulate_rate(make_circuit(), duration=200.0, dt=0.01)
        assert abs(r1.rates[0, -1] - r2.rates[0, -1]) < 0.1

    def test_plasticity_hook_called(self):
        """Plasticity function is called each step and can mutate weights."""
        calls = []

        def mock_plasticity(step, t, dt, rates, circuit):
            calls.append(step)
            # Slowly increase bias (as a proxy for weight change)
            circuit.bias[0] += 0.001

        c = RateCircuit(
            n_populations=1,
            labels=["E"],
            tau=np.array([10.0]),
            transfer_fn=[phi],
            W=np.array([[0.0]]),
            bias=np.array([1.0]),
        )
        result = simulate_rate(c, duration=10.0, dt=0.1, plasticity_fn=mock_plasticity)
        assert len(calls) == 100  # 10ms / 0.1ms
        # Bias should have increased
        assert c.bias[0] > 1.0
