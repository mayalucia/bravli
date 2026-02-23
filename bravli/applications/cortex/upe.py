"""Uncertainty-modulated prediction error (UPE) circuit.

Implements the Wilmes & Senn (2025) 5-population rate model:

    E+   — positive prediction error (L2/3 pyramidal)
    E-   — negative prediction error (L2/3 pyramidal)
    SST+ — Martinotti cells, learn stimulus mean (subtractive inhibition)
    PV+  — basket cells, learn stimulus variance (divisive inhibition)
    R    — representation neuron, integrates prediction errors

The circuit computes UPE = (s - μ) / σ², the optimal Bayesian update
signal for a Gaussian generative model.

Reference:
    Wilmes KA, Petrovici MA, Sachidhanandam S, Senn W.
    "Uncertainty-modulated prediction errors in cortical microcircuits."
    eLife 2025, 14:e95127.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from bravli.simulation.rate_engine import (
    RateCircuit, RateResult, simulate_rate,
    phi, phi_pv, phi_power,
)
from bravli.simulation.rate_plasticity import UPEPlasticity
from bravli.utils import get_logger

LOG = get_logger("cortex.upe")

# Population indices (convention)
E_PLUS = 0
E_MINUS = 1
SST = 2
PV = 3
R = 4

POPULATION_LABELS = ["E+", "E-", "SST+", "PV+", "R"]

# Default parameters (following Wilmes & Senn 2025)
DEFAULT_PARAMS = {
    # Time constants (ms)
    "tau_E": 10.0,       # excitatory populations
    "tau_I": 2.0,        # inhibitory populations

    # Nudging factor: how much direct stimulus reaches interneurons
    "beta": 0.1,

    # Connection weights
    "w_Es": 1.0,        # stimulus → E+/E- (sensory drive)
    "w_ESST": 1.0,      # SST → E+/E- (subtractive, sign applied in W)
    "w_EPV": 6.0,        # PV → E+/E- (divisive normalization strength)
    "w_PVs": 1.0,        # stimulus → PV (via beta nudging)
    "w_PVSST": 1.0,      # SST → PV (subtraction before squaring)
    "w_Eplus_R": 1.0,    # E+ → R
    "w_Eminus_R": 1.0,   # E- → R (negative, error correction)
    "I_0": 1.0,          # divisive offset (prevents division by zero)

    # Dendritic nonlinearity exponent for E+/E-
    "k": 1.0,

    # Initial plastic weights (SST and PV from R)
    "w_SST_R_init": 0.1,
    "w_PV_R_init": 0.01,

    # Learning rates
    "eta_sst": 0.001,
    "eta_pv": 0.001,
}


def build_upe_circuit(params=None):
    """Construct the 5-population UPE circuit.

    Parameters
    ----------
    params : dict, optional
        Override default parameters. Keys match DEFAULT_PARAMS.

    Returns
    -------
    RateCircuit
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    n = 5
    tau = np.array([
        p["tau_E"],   # E+
        p["tau_E"],   # E-
        p["tau_I"],   # SST+
        p["tau_I"],   # PV+
        p["tau_E"],   # R
    ])

    # Weight matrix: W[i, j] = weight from j to i
    W = np.zeros((n, n))

    # SST subtractive inhibition on E+ and E-
    # (sign is negative — SST subtracts from the sensory input)
    W[E_PLUS, SST] = -p["w_ESST"]
    W[E_MINUS, SST] = -p["w_ESST"]

    # E+/E- drive R (positive and negative prediction errors)
    W[R, E_PLUS] = p["w_Eplus_R"]
    W[R, E_MINUS] = -p["w_Eminus_R"]

    # Plastic weights: SST and PV receive from R (representation)
    # NOTE: These are raw weights. The (1-β) scaling is handled by the
    # plasticity rule, which updates these in-place. At steady state,
    # W[SST, R] ≈ μ and W[PV, R]² ≈ σ².
    W[SST, R] = p["w_SST_R_init"]
    W[PV, R] = p["w_PV_R_init"]

    # PV receives SST subtraction scaled by β (teaching signal):
    #   β * (w_Ps * s - w_PSST * r_SST)
    # Since stimulus injects β*s into PV, the SST→PV weight is β-scaled too
    W[PV, SST] = -p["beta"] * p["w_PVSST"]

    bias = np.zeros(n)

    # Transfer functions
    k = p["k"]
    transfer_fn = [
        (lambda x, _k=k: phi_power(x, _k)) if k != 1.0 else phi,  # E+
        (lambda x, _k=k: phi_power(x, _k)) if k != 1.0 else phi,  # E-
        phi,       # SST+
        phi_pv,    # PV+ (QUADRATIC — critical)
        phi,       # R
    ]

    # Divisive normalization: PV divides E+ and E- outputs
    divisive = {
        E_PLUS: (PV, p["w_EPV"], p["I_0"]),
        E_MINUS: (PV, p["w_EPV"], p["I_0"]),
    }

    circuit = RateCircuit(
        n_populations=n,
        labels=list(POPULATION_LABELS),
        tau=tau,
        transfer_fn=transfer_fn,
        W=W,
        bias=bias,
        divisive=divisive,
    )

    LOG.info("Built UPE circuit: %d populations, beta=%.2f, w_EPV=%.1f, I_0=%.1f",
             n, p["beta"], p["w_EPV"], p["I_0"])
    return circuit


def make_stimulus(distribution_sequence, dt=0.1, trial_duration=10.0):
    """Create a stimulus array from a sequence of Gaussian blocks.

    Parameters
    ----------
    distribution_sequence : list of (mean, std, n_trials)
        Each block draws n_trials samples from N(mean, std²).
    dt : float
        Timestep (ms).
    trial_duration : float
        Duration of each trial (ms). The stimulus is constant within a trial.

    Returns
    -------
    stimulus : np.ndarray
        Shape (5, n_total_steps). Stimulus injected into E+, E-, SST, PV
        populations with appropriate beta-scaling.
    samples : np.ndarray
        The raw stimulus samples (one per trial).
    block_boundaries : list of int
        Step indices where distribution blocks change.
    """
    p = DEFAULT_PARAMS
    beta = p["beta"]

    steps_per_trial = int(trial_duration / dt)
    all_samples = []
    block_boundaries = [0]

    for mean, std, n_trials in distribution_sequence:
        samples = np.random.normal(mean, std, n_trials)
        samples = np.maximum(samples, 0)  # rectify: rates can't be negative
        all_samples.append(samples)
        block_boundaries.append(block_boundaries[-1] + n_trials * steps_per_trial)

    all_samples = np.concatenate(all_samples)
    n_total_steps = len(all_samples) * steps_per_trial

    # Build stimulus array: 5 populations × n_steps
    stim = np.zeros((5, n_total_steps))

    for trial_idx, s in enumerate(all_samples):
        t_start = trial_idx * steps_per_trial
        t_end = t_start + steps_per_trial

        # Sensory input to E+ and E- (full strength)
        stim[E_PLUS, t_start:t_end] = s
        stim[E_MINUS, t_start:t_end] = -s  # E- gets negated sensory input

        # Nudging: direct stimulus to SST and PV (scaled by beta)
        stim[SST, t_start:t_end] = beta * s
        stim[PV, t_start:t_end] = beta * s

    return stim, all_samples, block_boundaries


def run_upe_experiment(distribution_sequence, params=None, dt=0.1,
                       trial_duration=10.0, seed=42):
    """Run a complete UPE learning experiment.

    Parameters
    ----------
    distribution_sequence : list of (mean, std, n_trials)
        Blocks of stimulus distributions.
    params : dict, optional
        Override circuit parameters.
    dt : float
        Timestep (ms).
    trial_duration : float
        Duration per trial (ms).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        result : RateResult
        plasticity : UPEPlasticity
        samples : np.ndarray (raw stimulus values)
        block_boundaries : list of int
        circuit : RateCircuit (final state with learned weights)
    """
    np.random.seed(seed)

    p = {**DEFAULT_PARAMS, **(params or {})}
    circuit = build_upe_circuit(p)
    stim, samples, boundaries = make_stimulus(
        distribution_sequence, dt=dt, trial_duration=trial_duration,
    )
    duration = stim.shape[1] * dt

    plasticity = UPEPlasticity(
        eta_sst=p["eta_sst"],
        eta_pv=p["eta_pv"],
        sst_idx=SST,
        pv_idx=PV,
        repr_idx=R,
        snapshot_interval=int(trial_duration / dt),  # snapshot every trial
    )

    result = simulate_rate(
        circuit, duration=duration, dt=dt,
        stimulus=stim, plasticity_fn=plasticity,
    )

    return {
        "result": result,
        "plasticity": plasticity,
        "samples": samples,
        "block_boundaries": boundaries,
        "circuit": circuit,
    }


def analyze_upe(experiment):
    """Analyze a UPE experiment for convergence and accuracy.

    Parameters
    ----------
    experiment : dict
        Output of run_upe_experiment().

    Returns
    -------
    dict with analysis results.
    """
    plasticity = experiment["plasticity"]
    samples = experiment["samples"]

    # Overall stimulus statistics
    mean_true = float(np.mean(samples))
    var_true = float(np.var(samples))

    # Learned values
    learned_mean = plasticity.learned_mean()
    learned_var = plasticity.learned_variance()

    analysis = {
        "true_mean": mean_true,
        "true_variance": var_true,
        "learned_mean": learned_mean,
        "learned_variance": learned_var,
    }

    if learned_mean is not None:
        analysis["mean_error"] = abs(learned_mean - mean_true)
        analysis["mean_relative_error"] = (
            abs(learned_mean - mean_true) / max(abs(mean_true), 1e-6)
        )
    if learned_var is not None and var_true > 0:
        analysis["variance_error"] = abs(learned_var - var_true)
        analysis["variance_relative_error"] = (
            abs(learned_var - var_true) / var_true
        )

    return analysis
