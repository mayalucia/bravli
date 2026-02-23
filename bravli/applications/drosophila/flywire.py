"""FlyWire connectome loading and edge processing.

Re-exports from bravli.parcellation.load_flywire and
bravli.connectivity.edges for convenient access.
"""

from bravli.parcellation.load_flywire import (
    build_neuropil_hierarchy,
    load_flywire_annotations,
    load_parcellation,
)
from bravli.connectivity.edges import (
    load_edges,
    assign_dominant_nt,
    threshold_edges,
    aggregate_by_pair,
)

__all__ = [
    "build_neuropil_hierarchy",
    "load_flywire_annotations",
    "load_parcellation",
    "load_edges",
    "assign_dominant_nt",
    "threshold_edges",
    "aggregate_by_pair",
]
