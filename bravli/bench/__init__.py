"""bench — Dataset management for scientific data.

The central abstraction: a Dataset is a typed, named, lazy-loading object
that knows how to materialize itself from disc. The @evaluate_datasets
decorator lets domain functions accept either raw data or Dataset objects
transparently.
"""

from .dataset import (
    Dataset,
    LocalDataset,
    CuratedDataset,
    GeneratedDataset,
    evaluate_datasets,
)
