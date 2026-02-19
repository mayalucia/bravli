"""Assemble a simulation-ready circuit from bravli data pipeline outputs.

A Circuit packages neuron parameters and connectivity into numpy arrays
suitable for the LIF engine.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from bravli.utils import get_logger

LOG = get_logger("simulation.circuit")


@dataclass
class Circuit:
    """A simulation-ready neural circuit.

    All arrays are indexed by a dense neuron index [0, n_neurons).
    The mapping from FlyWire root_id to index is stored in id_to_idx.

    Attributes
    ----------
    n_neurons : int
        Number of neurons.
    v_rest : np.ndarray
        Resting potential per neuron (mV).
    v_thresh : np.ndarray
        Spike threshold per neuron (mV).
    v_reset : np.ndarray
        Reset potential per neuron (mV).
    tau_m : np.ndarray
        Membrane time constant per neuron (ms).
    t_ref : np.ndarray
        Refractory period per neuron (ms).
    pre_idx : np.ndarray
        Presynaptic neuron indices (int, length = n_synapses).
    post_idx : np.ndarray
        Postsynaptic neuron indices (int, length = n_synapses).
    weights : np.ndarray
        Synaptic weights (mV, length = n_synapses).
    tau_syn : np.ndarray or float
        Synaptic time constant (ms). Scalar or per-synapse.
    delay_steps : np.ndarray or int
        Transmission delay in timesteps. Scalar or per-synapse.
    id_to_idx : dict
        Mapping from FlyWire root_id to dense index.
    idx_to_id : np.ndarray
        Mapping from dense index to root_id.
    neuron_labels : Optional[pd.DataFrame]
        Neuron metadata (model_name, model_mode, super_class, etc.).
    """
    n_neurons: int
    v_rest: np.ndarray
    v_thresh: np.ndarray
    v_reset: np.ndarray
    tau_m: np.ndarray
    t_ref: np.ndarray
    pre_idx: np.ndarray
    post_idx: np.ndarray
    weights: np.ndarray
    tau_syn: float = 5.0
    delay_steps: int = 18  # 1.8 ms at dt=0.1ms
    id_to_idx: dict = field(default_factory=dict)
    idx_to_id: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_labels: Optional[pd.DataFrame] = None

    @property
    def n_synapses(self):
        return len(self.weights)

    @property
    def is_heterogeneous(self):
        """True if neurons have different parameters."""
        return not (np.all(self.v_rest == self.v_rest[0])
                    and np.all(self.tau_m == self.tau_m[0]))

    def neuron_ids(self, indices):
        """Convert dense indices back to root_ids."""
        return self.idx_to_id[indices]

    def neuron_indices(self, root_ids):
        """Convert root_ids to dense indices."""
        return np.array([self.id_to_idx[rid] for rid in root_ids])

    def summary(self):
        """Return a summary string."""
        lines = [
            f"Circuit: {self.n_neurons:,} neurons, {self.n_synapses:,} synapses",
            f"  Heterogeneous: {self.is_heterogeneous}",
            f"  tau_syn: {self.tau_syn} ms",
            f"  delay: {self.delay_steps} steps",
            f"  weight range: [{self.weights.min():.3f}, {self.weights.max():.3f}] mV" if len(self.weights) > 0 else "  weight range: (no synapses)",
        ]
        if self.neuron_labels is not None and "model_mode" in self.neuron_labels.columns:
            modes = self.neuron_labels["model_mode"].value_counts()
            lines.append(f"  modes: {dict(modes)}")
        return "\n".join(lines)


def build_circuit(neurons, edges, dt=0.1):
    """Build a Circuit from annotated neuron and edge DataFrames.

    Parameters
    ----------
    neurons : pd.DataFrame
        Output of assign_cell_models(). Must have: root_id, v_rest,
        v_thresh, v_reset, tau_m, t_ref.
    edges : pd.DataFrame
        Output of compute_synaptic_weights(). Must have:
        pre_pt_root_id, post_pt_root_id, weight.
    dt : float
        Simulation timestep (ms). Used to compute delay_steps.

    Returns
    -------
    Circuit
    """
    # Build dense index
    root_ids = neurons["root_id"].values
    n = len(root_ids)
    id_to_idx = {rid: i for i, rid in enumerate(root_ids)}

    # Filter edges to neurons present in the circuit
    valid = (edges["pre_pt_root_id"].isin(id_to_idx) &
             edges["post_pt_root_id"].isin(id_to_idx))
    edges_valid = edges[valid]

    if len(edges_valid) < len(edges):
        LOG.warning("Dropped %d edges with neurons not in circuit",
                    len(edges) - len(edges_valid))

    pre_idx = edges_valid["pre_pt_root_id"].map(id_to_idx).values.astype(np.int32)
    post_idx = edges_valid["post_pt_root_id"].map(id_to_idx).values.astype(np.int32)
    weights = edges_valid["weight"].values.astype(np.float64)

    # Synaptic time constant: per-synapse if available, else default
    tau_syn = 5.0
    if "tau_decay" in edges_valid.columns:
        tau_vals = edges_valid["tau_decay"].values
        if not np.all(pd.isna(tau_vals)):
            tau_syn = np.where(pd.isna(tau_vals), 5.0, tau_vals).astype(np.float64)

    # Delay: 1.8 ms default
    delay_steps = max(1, int(round(1.8 / dt)))

    circuit = Circuit(
        n_neurons=n,
        v_rest=neurons["v_rest"].values.astype(np.float64),
        v_thresh=neurons["v_thresh"].values.astype(np.float64),
        v_reset=neurons["v_reset"].values.astype(np.float64),
        tau_m=neurons["tau_m"].values.astype(np.float64),
        t_ref=neurons["t_ref"].values.astype(np.float64),
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        tau_syn=tau_syn,
        delay_steps=delay_steps,
        id_to_idx=id_to_idx,
        idx_to_id=root_ids,
        neuron_labels=neurons,
    )

    LOG.info("Built circuit: %d neurons, %d synapses", n, len(weights))
    return circuit


def build_circuit_from_edges(edges, dt=0.1, v_rest=-52.0, v_thresh=-45.0,
                              v_reset=-52.0, tau_m=20.0, t_ref=2.2):
    """Build a uniform (Shiu-style) circuit directly from an edge DataFrame.

    Convenience function when you don't need class-aware cell models.
    Neurons are inferred from unique pre/post IDs in the edge table.

    Parameters
    ----------
    edges : pd.DataFrame
        Must have: pre_pt_root_id, post_pt_root_id, weight.
    dt : float
        Timestep (ms).
    v_rest, v_thresh, v_reset, tau_m, t_ref : float
        Uniform parameters for all neurons.

    Returns
    -------
    Circuit
    """
    all_ids = np.union1d(
        edges["pre_pt_root_id"].unique(),
        edges["post_pt_root_id"].unique(),
    )
    n = len(all_ids)

    neurons = pd.DataFrame({
        "root_id": all_ids,
        "v_rest": v_rest,
        "v_thresh": v_thresh,
        "v_reset": v_reset,
        "tau_m": tau_m,
        "t_ref": t_ref,
    })

    return build_circuit(neurons, edges, dt=dt)
