"""Plasticity rules for the LIF simulation engine.

Each rule is a callable compatible with the engine's plasticity_fn interface:
    rule(step, t, dt, spiked, v, g, circuit)

The rule may mutate circuit.weights in-place.

References:
    Izhikevich EM (2007). Cerebral Cortex 17(10):2443-2452.
    Hige T et al. (2015). Neuron 88(5):985-998.
    Handler A et al. (2019). Cell 178(1):60-75.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from bravli.utils import get_logger

LOG = get_logger("simulation.plasticity")


@dataclass
class ThreeFactorSTDP:
    """Dopamine-gated KC->MBON depression (three-factor learning rule).

    The rule implements the core mechanism of Drosophila olfactory conditioning:

    1. When a KC spikes, an eligibility trace is incremented at all its
       outgoing KC->MBON synapses. The trace decays with tau_eligibility.
    2. When a DAN spikes (in a specific compartment), a dopamine signal
       is generated for that compartment. It decays with tau_dopamine.
    3. Weight update: dw = -lr * eligibility * da_signal * dt
       (depression only — KC->MBON synapses weaken when KC activity
       coincides with DAN activity in the same compartment).

    This is the simplified Hige et al. 2015 rule. The full Handler et al.
    2019 model would require receptor-specific timing windows per compartment.

    Parameters
    ----------
    compartment_index : dict
        From build_compartment_index(). Maps compartment names to
        {kc_indices, mbon_indices, dan_indices, kc_mbon_syn_mask}.
    tau_eligibility : float
        Eligibility trace decay time constant (ms).
    tau_dopamine : float
        Dopamine signal decay time constant (ms).
    lr : float
        Learning rate (weight change per eligibility * da_signal * dt).
    w_min : float
        Minimum weight — prevents sign reversal.
    snapshot_interval_ms : float
        How often to record weight snapshots (ms). 0 to disable.
    """
    compartment_index: dict
    tau_eligibility: float = 1000.0
    tau_dopamine: float = 500.0
    lr: float = 0.001
    w_min: float = 0.0
    snapshot_interval_ms: float = 100.0

    # Internal state (populated by __post_init__)
    _kc_mbon_global_mask: np.ndarray = field(init=False, repr=False)
    _kc_mbon_syn_indices: np.ndarray = field(init=False, repr=False)
    _eligibility: np.ndarray = field(init=False, repr=False)
    _da_signal: dict = field(init=False, repr=False)
    _comp_local_masks: dict = field(init=False, repr=False)
    weight_snapshots: list = field(init=False, repr=False)
    snapshot_times: list = field(init=False, repr=False)
    da_history: dict = field(init=False, repr=False)

    def __post_init__(self):
        # Global mask: union of all compartment KC->MBON synapses
        first_comp = next(iter(self.compartment_index.values()))
        n_syn = len(first_comp["kc_mbon_syn_mask"])
        self._kc_mbon_global_mask = np.zeros(n_syn, dtype=bool)

        for info in self.compartment_index.values():
            self._kc_mbon_global_mask |= info["kc_mbon_syn_mask"]

        # Dense indices of KC->MBON synapses in the full weight array
        self._kc_mbon_syn_indices = np.where(self._kc_mbon_global_mask)[0]
        n_plastic = len(self._kc_mbon_syn_indices)

        # Eligibility trace: one per plastic synapse
        self._eligibility = np.zeros(n_plastic, dtype=np.float64)

        # Dopamine signal: one per compartment
        self._da_signal = {comp: 0.0 for comp in self.compartment_index}

        # Per-compartment local mask: which indices within _kc_mbon_syn_indices
        # belong to each compartment
        self._comp_local_masks = {}
        for comp, info in self.compartment_index.items():
            comp_global = info["kc_mbon_syn_mask"]
            # Local indices: positions in _kc_mbon_syn_indices that match this compartment
            local = np.array([
                i for i, global_idx in enumerate(self._kc_mbon_syn_indices)
                if comp_global[global_idx]
            ], dtype=np.int32)
            self._comp_local_masks[comp] = local

        # Recording
        self.weight_snapshots = []
        self.snapshot_times = []
        self.da_history = {comp: [] for comp in self.compartment_index}

        LOG.info("ThreeFactorSTDP: %d plastic synapses across %d compartments, "
                 "tau_elig=%.0f ms, tau_da=%.0f ms, lr=%.4f",
                 n_plastic, len(self.compartment_index),
                 self.tau_eligibility, self.tau_dopamine, self.lr)

    def __call__(self, step, t, dt, spiked, v, g, circuit):
        """Called by the engine each timestep."""
        # 1. Decay eligibility traces
        self._eligibility *= (1.0 - dt / self.tau_eligibility)

        # 2. KC spikes -> increment eligibility
        if np.any(spiked):
            pre_neurons = circuit.pre_idx[self._kc_mbon_syn_indices]
            spiked_mask = spiked[pre_neurons]
            self._eligibility[spiked_mask] += 1.0

        # 3. Decay and update dopamine signals per compartment
        for comp, info in self.compartment_index.items():
            self._da_signal[comp] *= (1.0 - dt / self.tau_dopamine)
            dan_indices = info["dan_indices"]
            if len(dan_indices) > 0 and np.any(spiked[dan_indices]):
                self._da_signal[comp] += 1.0

        # 4. Weight update: depress where eligibility AND dopamine are nonzero
        for comp, local_mask in self._comp_local_masks.items():
            da = self._da_signal[comp]
            if da <= 0 or len(local_mask) == 0:
                continue
            delta = self.lr * self._eligibility[local_mask] * da * dt
            global_indices = self._kc_mbon_syn_indices[local_mask]
            circuit.weights[global_indices] -= delta

        # 5. Clamp weights
        circuit.weights[self._kc_mbon_syn_indices] = np.maximum(
            circuit.weights[self._kc_mbon_syn_indices], self.w_min
        )

        # 6. Periodic snapshots
        if self.snapshot_interval_ms > 0:
            snapshot_steps = int(self.snapshot_interval_ms / dt)
            if snapshot_steps > 0 and step % snapshot_steps == 0:
                self.weight_snapshots.append(
                    circuit.weights[self._kc_mbon_syn_indices].copy()
                )
                self.snapshot_times.append(t)
                for comp in self.compartment_index:
                    self.da_history[comp].append(self._da_signal[comp])

    @property
    def n_plastic_synapses(self):
        """Number of plastic KC->MBON synapses."""
        return len(self._kc_mbon_syn_indices)

    def mean_weight(self, circuit):
        """Current mean weight of plastic synapses."""
        return float(np.mean(circuit.weights[self._kc_mbon_syn_indices]))

    def weight_change_summary(self, circuit, initial_weights):
        """Summarize weight changes relative to initial values.

        Parameters
        ----------
        circuit : Circuit
            Current circuit state.
        initial_weights : np.ndarray
            Weights at start of training.

        Returns
        -------
        dict
            Per-compartment and global weight change statistics.
        """
        current = circuit.weights[self._kc_mbon_syn_indices]
        initial = initial_weights[self._kc_mbon_syn_indices]
        delta = current - initial

        summary = {
            "global_mean_change": float(np.mean(delta)),
            "global_mean_pct_change": float(
                np.mean(delta / np.where(initial != 0, initial, 1.0)) * 100
            ),
            "n_depressed": int(np.sum(delta < -1e-10)),
            "n_unchanged": int(np.sum(np.abs(delta) < 1e-10)),
            "compartments": {},
        }

        for comp, local_mask in self._comp_local_masks.items():
            if len(local_mask) == 0:
                continue
            comp_delta = delta[local_mask]
            summary["compartments"][comp] = {
                "n_synapses": len(local_mask),
                "mean_change": float(np.mean(comp_delta)),
                "n_depressed": int(np.sum(comp_delta < -1e-10)),
            }

        return summary
