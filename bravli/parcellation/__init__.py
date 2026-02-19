"""parcellation — Neuropil region hierarchy and FlyWire data loading.

The fly brain is divided into ~78 neuropils. This module provides:
  - NeuropilRegion: a recursive tree node for the region hierarchy
  - FlyBrainParcellation: the top-level parcellation object
  - load_flywire_annotations: load the public FlyWire annotation TSV
"""

from .parcellation import NeuropilRegion, FlyBrainParcellation
from .load_flywire import load_flywire_annotations, build_neuropil_hierarchy
