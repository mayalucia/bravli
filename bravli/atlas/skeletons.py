"""Load neuron skeletons from the Zenodo FlyWire parquet file.

Data source: sk_lod1_783_healed_ds2.parquet (5.1 GB)
  DOI: 10.5281/zenodo.10877326
  Contains: TreeNeuron skeletons for all 139,244 FlyWire 783 neurons
  Coordinates: nanometres in FAFB14.1 space
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from bravli.utils import get_logger

LOG = get_logger("atlas.skeletons")

try:
    import navis
    HAS_NAVIS = True
except ImportError:
    HAS_NAVIS = False

# Default path (relative to repo root)
DEFAULT_PARQUET = Path("data/zenodo/sk_lod1_783_healed_ds2.parquet")


def load_skeletons(
    root_ids: List[int],
    parquet_path: Optional[Union[str, Path]] = None,
) -> "navis.NeuronList":
    """Load neuron skeletons by root ID from the Zenodo parquet.

    Parameters
    ----------
    root_ids : list of int
        FlyWire root IDs of neurons to load.
    parquet_path : str or Path, optional
        Path to the skeleton parquet file.  Defaults to
        data/zenodo/sk_lod1_783_healed_ds2.parquet.

    Returns
    -------
    navis.NeuronList
        Loaded TreeNeuron skeletons.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.
    ImportError
        If navis is not installed.
    """
    if not HAS_NAVIS:
        raise ImportError("navis is required. Install with: pip install navis")

    path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"Skeleton parquet not found at {path}. "
            "Download from: https://doi.org/10.5281/zenodo.10877326"
        )

    LOG.info("Loading %d skeletons from %s", len(root_ids), path.name)
    neurons = navis.read_parquet(str(path), subset=root_ids)
    LOG.info("Loaded %d neurons (total %d nodes)",
             len(neurons),
             sum(n.n_nodes for n in neurons))
    return neurons


def sample_neuron_ids(
    annotations: pd.DataFrame,
    cell_class: Optional[str] = None,
    cell_type: Optional[str] = None,
    side: str = "right",
    n: int = 5,
) -> List[int]:
    """Sample root IDs from the annotation table.

    Filters annotations by cell_class, cell_type, and hemisphere,
    then returns the first `n` root IDs.  Deterministic (no randomness)
    so results are reproducible.

    Parameters
    ----------
    annotations : pd.DataFrame
        FlyWire annotation table with 'root_id', 'cell_class',
        'cell_type', 'side' columns.
    cell_class : str, optional
        Filter by cell_class (e.g., 'Kenyon_Cell', 'MBON', 'DAN').
    cell_type : str, optional
        Filter by cell_type (e.g., 'KCg-m', 'PAM01').
    side : str
        Hemisphere filter: 'right', 'left', or 'both'.
    n : int
        Number of IDs to return.

    Returns
    -------
    list of int
        Root IDs suitable for `load_skeletons()`.
    """
    mask = pd.Series(True, index=annotations.index)

    if cell_class is not None:
        mask &= annotations["cell_class"] == cell_class
    if cell_type is not None:
        mask &= annotations["cell_type"] == cell_type
    if side != "both" and "side" in annotations.columns:
        mask &= annotations["side"] == side

    filtered = annotations[mask]
    ids = filtered["root_id"].head(n).tolist()
    LOG.debug("Sampled %d IDs (class=%s, type=%s, side=%s)",
              len(ids), cell_class, cell_type, side)
    return ids
