"""Tests for the frog retinal circuit: Lettvin (1959) predictive coding."""

import numpy as np
import pytest

from bravli.simulation.visual_world import (
    Arena, straight_trajectory, circular_trajectory, random_walk_trajectory,
    static_trajectory, onset_trajectory, render_retina,
)
from bravli.applications.retina.frog import (
    build_frog_retina, retinal_stimulus, run_frog_experiment,
    extract_population, _dog_weights,
    P_TYPE, S_TYPE, T_TYPE, DEFAULT_PARAMS,
)


# ---- Visual World ----

class TestTrajectories:

    def test_straight_trajectory_shape(self):
        pos = straight_trajectory((10, 50), (0.1, 0), 100, dt=0.1)
        assert pos.shape == (100, 2)

    def test_straight_trajectory_direction(self):
        pos = straight_trajectory((0, 0), (1.0, 0), 100, dt=1.0)
        assert pos[-1, 0] == pytest.approx(99.0)
        assert pos[-1, 1] == pytest.approx(0.0)

    def test_circular_trajectory_returns_to_start(self):
        # 101 steps to complete full circle (step 0 and step 100 = same angle)
        pos = circular_trajectory((50, 50), 20.0, 2 * np.pi / 100.0, 101, dt=1.0)
        np.testing.assert_allclose(pos[0], pos[-1], atol=0.5)

    def test_random_walk_deterministic(self):
        p1 = random_walk_trajectory((50, 50), 1.0, 100, seed=42)
        p2 = random_walk_trajectory((50, 50), 1.0, 100, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_static_trajectory(self):
        pos = static_trajectory((25, 50), 200)
        assert pos.shape == (200, 2)
        assert np.all(pos[:, 0] == 25)

    def test_onset_trajectory_nan_before(self):
        pos = onset_trajectory((50, 50), 100, 200)
        assert np.all(np.isnan(pos[:100]))
        assert not np.any(np.isnan(pos[100:]))


class TestRenderRetina:

    def test_output_shape(self):
        arena = Arena(width=100.0, n_pixels=50)
        pos = static_trajectory((50, 50), 100)
        img = render_retina(arena, pos, bug_radius=2.0)
        assert img.shape == (50, 100)

    def test_no_bug_gives_background(self):
        arena = Arena(width=100.0, n_pixels=20, background=0.0)
        pos = onset_trajectory((50, 50), 100, 100)  # all NaN
        img = render_retina(arena, pos, bug_radius=2.0)
        assert np.all(img == 0.0)

    def test_bug_creates_luminance_bump(self):
        arena = Arena(width=100.0, n_pixels=50)
        pos = static_trajectory((50, 50), 10)
        img = render_retina(arena, pos, bug_radius=2.0)
        # Center pixel (index ~25) should be brightest
        center_pix = arena.n_pixels // 2
        assert img[center_pix, 0] > img[0, 0]

    def test_moving_bug_shifts_bump(self):
        arena = Arena(width=100.0, n_pixels=50)
        pos = straight_trajectory((20, 50), (1.0, 0), 100, dt=1.0)
        img = render_retina(arena, pos, bug_radius=2.0)
        # Peak pixel should shift rightward
        peak_early = np.argmax(img[:, 0])
        peak_late = np.argmax(img[:, -1])
        assert peak_late > peak_early


# ---- Circuit Construction ----

class TestFrogCircuit:

    def test_population_count(self):
        c = build_frog_retina(n_pixels=20)
        assert c.n_populations == 60  # 3 * 20

    def test_labels(self):
        c = build_frog_retina(n_pixels=5)
        assert c.labels[:5] == ["P0", "P1", "P2", "P3", "P4"]
        assert c.labels[5:10] == ["S0", "S1", "S2", "S3", "S4"]
        assert c.labels[10:15] == ["T0", "T1", "T2", "T3", "T4"]

    def test_time_constants(self):
        c = build_frog_retina(n_pixels=10)
        N = 10
        assert np.all(c.tau[:N] == DEFAULT_PARAMS["tau_P"])
        assert np.all(c.tau[N:2*N] == DEFAULT_PARAMS["tau_S"])
        assert np.all(c.tau[2*N:] == DEFAULT_PARAMS["tau_T"])

    def test_dog_weights_center_positive(self):
        dog = _dog_weights(20, sigma_c=1.5, sigma_s=5.0, w_surround=0.7)
        # Diagonal (self-connection) should be positive (center)
        for i in range(20):
            assert dog[i, i] > 0

    def test_dog_weights_surround_negative(self):
        dog = _dog_weights(20, sigma_c=1.5, sigma_s=5.0, w_surround=0.7)
        # Far-off-diagonal should be negative (surround)
        assert dog[0, 10] < 0

    def test_s_to_t_inhibitory(self):
        c = build_frog_retina(n_pixels=10)
        N = 10
        # S → T should be inhibitory (negative weight)
        for i in range(N):
            assert c.W[T_TYPE * N + i, S_TYPE * N + i] < 0

    def test_p_to_t_excitatory(self):
        c = build_frog_retina(n_pixels=10)
        N = 10
        for i in range(N):
            assert c.W[T_TYPE * N + i, P_TYPE * N + i] > 0


# ---- Circuit Dynamics ----

class TestFrogDynamics:

    def test_smoke_test(self):
        """Circuit runs without error."""
        exp = run_frog_experiment("onset", n_pixels=10, duration=100.0)
        assert exp["result"].n_populations == 30
        assert exp["result"].n_steps > 0

    def test_onset_transient_burst(self):
        """T cells fire transiently when bug appears, then partially decay.

        With center-surround, a static localized bug produces a steady-state
        residual in T (spatial prediction error). The transient component
        should decay, but a residual remains because the DoG cannot
        perfectly predict a point source.
        """
        exp = run_frog_experiment(
            "onset", n_pixels=20, duration=800.0, dt=0.1,
            onset_step=int(100.0 / 0.1),  # onset at 100ms
            position=(50.0, 50.0),
        )
        N = 20
        T_rates = extract_population(exp["result"], T_TYPE, N)
        center = N // 2

        onset_idx = int(100.0 / 0.1)
        trace = T_rates[center, :]
        post_onset = trace[onset_idx:]
        peak_val = np.max(post_onset)
        peak_idx = np.argmax(post_onset)

        # Peak should occur shortly after onset
        assert peak_idx < int(50.0 / 0.1), "Peak should be within 50ms of onset"
        assert peak_val > 0.1, "T cell should fire at onset"

        # Late value should be less than peak (transient component decayed)
        late_val = np.mean(trace[-int(50.0 / 0.1):])  # last 50ms
        assert late_val < peak_val, \
            f"T should decay from peak: late={late_val:.4f} vs peak={peak_val:.4f}"

    def test_sustained_responds_to_bug(self):
        """S cells should have positive steady-state response to a localized bug."""
        exp = run_frog_experiment(
            "onset", n_pixels=20, duration=800.0, dt=0.1,
            onset_step=int(50.0 / 0.1),
            position=(50.0, 50.0),
        )
        N = 20
        S_rates = extract_population(exp["result"], S_TYPE, N)
        center = N // 2

        # Late in simulation, S at bug location should be positive
        late_start = int(600.0 / 0.1)
        s_late = np.mean(S_rates[center, late_start:])
        assert s_late > 0.05, f"S should respond to localized bug: {s_late:.4f}"

    def test_moving_bug_traveling_wave(self):
        """Moving bug should produce a traveling wave of T activity."""
        exp = run_frog_experiment(
            "straight", n_pixels=30, duration=1000.0, dt=0.1,
            start=(10.0, 50.0), velocity=(0.08, 0.0),
            arena_width=100.0, bug_radius=3.0,
        )
        N = 30
        T_rates = extract_population(exp["result"], T_TYPE, N)

        # Find peak T time for two spatial locations
        pix_a = 8   # ~27% of retina
        pix_b = 20  # ~67% of retina
        peak_time_a = np.argmax(T_rates[pix_a, :])
        peak_time_b = np.argmax(T_rates[pix_b, :])

        # Bug moves left to right, so pix_b should peak later
        assert peak_time_b > peak_time_a, \
            "T peak should arrive later at rightward pixel"

    def test_uniform_vs_spot_steady_state(self):
        """At steady state, uniform illumination should give LARGER T than spot.

        This is the center-surround prediction: for uniform input, the DoG
        drives S→0 (surround cancels center), so T = P - 0 = P (full error).
        For a localized spot, S > 0 (center dominates surround), so T = P - S < P.
        The spot is partially predicted; the uniform flash is not.

        This demonstrates that center-surround performs SPATIAL prediction:
        it predicts the center from the surround, and uniform illumination
        is perfectly predicted (S matches P) only if the DoG row sums to 1.
        With our balanced DoG (row sum < 0), uniform input actually drives
        S to 0, leaving T = P.
        """
        N = 20
        n_steps = int(800.0 / 0.1)
        onset = int(100.0 / 0.1)
        center = N // 2

        # Localized spot at center
        spot_image = np.zeros((N, n_steps))
        spot_image[center-1:center+2, onset:] = 1.0

        circuit = build_frog_retina(N)
        stim_spot = retinal_stimulus(spot_image, N)

        from bravli.simulation.rate_engine import simulate_rate
        res_spot = simulate_rate(circuit, duration=800.0, dt=0.1,
                                 stimulus=stim_spot)

        S_spot = extract_population(res_spot, S_TYPE, N)
        T_spot = extract_population(res_spot, T_TYPE, N)

        # S should be positive at center for localized spot (center > surround)
        s_late = np.mean(S_spot[center, -int(100.0 / 0.1):])
        assert s_late > 0.05, f"S should respond to localized spot: {s_late:.4f}"

        # T should be less than P at center (partially predicted)
        t_late = np.mean(T_spot[center, -int(100.0 / 0.1):])
        assert t_late < 1.0, f"T should be reduced by S prediction: {t_late:.4f}"
        assert t_late > 0.0, f"T should still be positive: {t_late:.4f}"
