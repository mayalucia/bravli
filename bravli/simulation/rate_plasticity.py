"""Local learning rules for rate-based circuits.

Implements the plasticity rules from Wilmes & Senn (2025) for the
uncertainty-modulated prediction error circuit:

    SST learning (mean estimation):
        Δw_SST,a = η (r_SST - φ(w_SST,a · r_a)) · r_a

    PV learning (variance estimation):
        Δw_PV,a = η (r_PV - φ_PV(w_PV,a · r_a)) · r_a

Both are anti-Hebbian prediction-error rules: the weight changes to
minimize the discrepancy between the interneuron's current firing rate
and what it would produce from the representation neuron alone (without
direct stimulus nudging). At convergence:
    - SST weight → stimulus mean (because φ is linear in the active regime)
    - PV weight² → stimulus variance (because φ_PV is quadratic)

Reference:
    Wilmes KA et al. (2025). eLife 14:e95127, Eqs. 10, 13.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from bravli.simulation.rate_engine import phi, phi_pv
from bravli.utils import get_logger

LOG = get_logger("simulation.rate_plasticity")


@dataclass
class UPEPlasticity:
    """Learning rules for the Wilmes & Senn UPE circuit.

    Updates SST and PV weights to track stimulus mean and variance.

    Parameters
    ----------
    eta_sst : float
        SST learning rate.
    eta_pv : float
        PV learning rate.
    sst_idx : int
        Population index of SST+ in the RateCircuit.
    pv_idx : int
        Population index of PV+ in the RateCircuit.
    repr_idx : int
        Population index of R (representation neuron).
    sst_transfer : callable
        Transfer function for SST (default: phi, rectified linear).
    pv_transfer : callable
        Transfer function for PV (default: phi_pv, quadratic).
    snapshot_interval : int
        Record weight snapshots every N steps. 0 to disable.
    """
    eta_sst: float = 0.001
    eta_pv: float = 0.001
    sst_idx: int = 2
    pv_idx: int = 3
    repr_idx: int = 4
    sst_transfer: Callable = phi
    pv_transfer: Callable = phi_pv
    snapshot_interval: int = 100

    # Recorded during simulation
    weight_snapshots: List = field(default_factory=list, init=False)
    snapshot_times: List = field(default_factory=list, init=False)

    def __call__(self, step, t, dt, rates, circuit):
        """Called each timestep by simulate_rate().

        Mutates circuit.W in-place to update SST and PV weights
        from the representation neuron.
        """
        r_sst = rates[self.sst_idx]
        r_pv = rates[self.pv_idx]
        r_repr = rates[self.repr_idx]

        # SST learning: Δw = η (r_SST - φ(w · r_R)) · r_R
        w_sst = circuit.W[self.sst_idx, self.repr_idx]
        predicted_sst = self.sst_transfer(w_sst * r_repr)
        dw_sst = self.eta_sst * (r_sst - predicted_sst) * r_repr
        circuit.W[self.sst_idx, self.repr_idx] += dw_sst * dt

        # PV learning: Δw = η (r_PV - φ_PV(w · r_R)) · r_R
        w_pv = circuit.W[self.pv_idx, self.repr_idx]
        predicted_pv = self.pv_transfer(w_pv * r_repr)
        dw_pv = self.eta_pv * (r_pv - predicted_pv) * r_repr
        circuit.W[self.pv_idx, self.repr_idx] += dw_pv * dt

        # Record
        if self.snapshot_interval > 0 and step % self.snapshot_interval == 0:
            self.weight_snapshots.append({
                "w_sst": circuit.W[self.sst_idx, self.repr_idx],
                "w_pv": circuit.W[self.pv_idx, self.repr_idx],
            })
            self.snapshot_times.append(t)

    def learned_mean(self):
        """The SST weight, which converges to the stimulus mean."""
        if self.weight_snapshots:
            return self.weight_snapshots[-1]["w_sst"]
        return None

    def learned_variance(self):
        """The PV weight squared, which converges to stimulus variance."""
        if self.weight_snapshots:
            w = self.weight_snapshots[-1]["w_pv"]
            return w ** 2
        return None

    def weight_trajectory(self, key="w_sst"):
        """Extract a weight trajectory from snapshots."""
        return np.array([s[key] for s in self.weight_snapshots])
