"""Assign cell electrical models to FlyWire neurons.

Uses the annotation table's super_class and cell_class columns to
resolve the best available cell model for each neuron.
"""

import pandas as pd

from bravli.bench.dataset import evaluate_datasets
from bravli.models.cell_models import CELL_MODEL_DB
from bravli.utils import get_logger

LOG = get_logger("models.assign")


@evaluate_datasets
def assign_cell_models(annotations, mode="class_aware"):
    """Assign cell model parameters to each neuron.

    Parameters
    ----------
    annotations : pd.DataFrame
        Neuron annotation table with 'root_id', 'super_class',
        and optionally 'cell_class' columns.
    mode : str
        "uniform" — all neurons get shiu_uniform params.
        "class_aware" — resolve by cell_class > super_class > default.

    Returns
    -------
    pd.DataFrame
        Input with added columns: model_name, model_mode,
        v_rest, v_thresh, v_reset, tau_m, t_ref, c_m, r_input,
        model_confidence.
    """
    result = annotations.copy()

    if mode == "uniform":
        model = CELL_MODEL_DB.get("shiu_uniform")
        result["model_name"] = model.name
        result["model_mode"] = model.mode
        result["v_rest"] = model.v_rest
        result["v_thresh"] = model.v_thresh
        result["v_reset"] = model.v_reset
        result["tau_m"] = model.tau_m
        result["t_ref"] = model.t_ref
        result["c_m"] = model.c_m
        result["r_input"] = model.r_input
        result["model_confidence"] = model.confidence
        LOG.info("Assigned uniform Shiu model to %d neurons", len(result))
        return result

    # Class-aware mode
    names = []
    modes = []
    v_rests = []
    v_threshes = []
    v_resets = []
    tau_ms = []
    t_refs = []
    c_ms = []
    r_inputs = []
    confs = []

    has_cell_class = "cell_class" in annotations.columns
    has_super_class = "super_class" in annotations.columns

    for _, row in annotations.iterrows():
        cc = row.get("cell_class") if has_cell_class else None
        sc = row.get("super_class") if has_super_class else None
        model = CELL_MODEL_DB.resolve(cell_class=cc, super_class=sc)

        if model is None:
            model = CELL_MODEL_DB.get("default_spiking")

        names.append(model.name)
        modes.append(model.mode)
        v_rests.append(model.v_rest)
        v_threshes.append(model.v_thresh)
        v_resets.append(model.v_reset)
        tau_ms.append(model.tau_m)
        t_refs.append(getattr(model, "t_ref", 0.0))
        c_ms.append(model.c_m)
        r_inputs.append(model.r_input)
        confs.append(model.confidence)

    result["model_name"] = names
    result["model_mode"] = modes
    result["v_rest"] = v_rests
    result["v_thresh"] = v_threshes
    result["v_reset"] = v_resets
    result["tau_m"] = tau_ms
    result["t_ref"] = t_refs
    result["c_m"] = c_ms
    result["r_input"] = r_inputs
    result["model_confidence"] = confs

    mode_counts = result["model_mode"].value_counts()
    LOG.info("Assigned cell models to %d neurons: %s",
             len(result), dict(mode_counts))
    return result


@evaluate_datasets
def population_summary(annotated_neurons):
    """Summarize cell model assignments across the population.

    Parameters
    ----------
    annotated_neurons : pd.DataFrame
        Output of assign_cell_models.

    Returns
    -------
    pd.DataFrame
        Per-model summary: count, mode, key parameters.
    """
    if "model_name" not in annotated_neurons.columns:
        raise ValueError("No 'model_name' column — run assign_cell_models first")

    rows = []
    for name, group in annotated_neurons.groupby("model_name"):
        row = {
            "model_name": name,
            "n_neurons": len(group),
            "mode": group["model_mode"].iloc[0],
            "v_rest_mV": group["v_rest"].iloc[0],
            "v_thresh_mV": group["v_thresh"].iloc[0],
            "tau_m_ms": group["tau_m"].iloc[0],
            "c_m_pF": group["c_m"].iloc[0],
            "confidence": group["model_confidence"].iloc[0],
        }
        rows.append(row)

    return (pd.DataFrame(rows)
            .set_index("model_name")
            .sort_values("n_neurons", ascending=False))
