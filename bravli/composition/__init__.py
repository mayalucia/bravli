"""composition — Cell type distributions and neurotransmitter profiles.

Pure functions that summarize the cellular composition of brain regions.
All functions accept either raw DataFrames or Dataset objects via the
@evaluate_datasets decorator.
"""

from .composition import (
    count_by,
    cell_type_distribution,
    neurotransmitter_profile,
    compare_divisions,
    top_types,
)
