"""simulation — Whole-brain LIF simulation engine.

Pure-numpy implementation of the Shiu et al. (2024) leaky integrate-and-fire
model, extended with class-aware heterogeneous parameters and graded-potential
neuron support.

References:
    Shiu et al. 2024 — Nature 634:210-219
    Zhang et al. 2024 — iScience (arXiv:2404.17128)
"""

from .circuit import (
    Circuit,
    build_circuit,
    build_circuit_from_edges,
)
from .engine import (
    SimulationResult,
    simulate,
)
from .stimulus import (
    StimulusProtocol,
    poisson_stimulus,
    step_stimulus,
    pulse_stimulus,
    combine_stimuli,
)
from .analysis import (
    firing_rates,
    spike_raster,
    ei_balance,
    active_fraction,
    population_sparseness,
    lifetime_sparseness,
    active_fraction_by_group,
    population_rate,
    weight_evolution,
    mbon_response_change,
    performance_index,
)
from .plasticity import ThreeFactorSTDP
