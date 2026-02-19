"""physiology — Synaptic physiology models for Drosophila brain simulation.

Maps neurotransmitter types to biophysical synapse parameters: kinetics,
reversal potentials, peak conductances, and short-term plasticity. Provides
both simple (sign-only) and biophysical (per-NT kinetics) model levels.

Data sources: Su & O'Dowd 2003 (nAChR), Lee et al. 2003 (GABA_A),
Han et al. 2024 (NMJ iGluR), Shiu et al. 2024 (whole-brain LIF).
"""

from .synapse_models import (
    SynapseModel,
    STPParams,
    SYNAPSE_DB,
    get_synapse_model,
    list_models,
    simple_sign,
)
from .assign import (
    assign_synapse_models,
    compute_synaptic_weights,
    physiology_summary,
)
