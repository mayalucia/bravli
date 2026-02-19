"""models — Point neuron models for Drosophila brain simulation.

Provides LIF and graded-potential neuron models with a parameter database
spanning uniform (Shiu), class-aware, and graded-aware configurations.

References:
    Shiu et al. 2024 — Whole-brain LIF (Nature 634:210-219)
    Gouwens & Wilson 2009 — PN passive properties (J Neurosci 29:6239)
    Su & O'Dowd 2003 — KC properties
    Azevedo et al. 2020 — Motoneuron properties
"""

from .cell_models import (
    LIFParams,
    GradedParams,
    CellModelDB,
    CELL_MODEL_DB,
    get_cell_params,
    list_cell_models,
)
from .assign import (
    assign_cell_models,
    population_summary,
)
