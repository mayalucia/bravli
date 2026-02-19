"""connectivity — FlyWire connectome edge list analysis.

Load the aggregated edge list, filter by synapse count threshold,
build neuropil connectivity matrices, and analyze pathway statistics.

Data source: Zenodo record 10676866
  proofread_connections_783.feather (852 MB)
"""

from .edges import (
    load_edges,
    threshold_edges,
    assign_dominant_nt,
    aggregate_by_pair,
)
from .matrix import (
    neuropil_synapse_counts,
    neuropil_connectivity_matrix,
    neuropil_nt_matrices,
)
from .pathways import (
    pathway_stats,
    top_pathways,
    convergence_divergence,
    nt_pathway_breakdown,
)
