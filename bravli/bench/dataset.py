"""Dataset hierarchy and the @evaluate_datasets decorator.

Heritage: ported from bravlibpy (Blue Brain Project), stripped of NEXUS,
Slurm, and connsense dependencies. The core patterns — lazy .value,
@evaluate_datasets, and serializable .define() — are preserved intact.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from copy import deepcopy

import pandas as pd
import numpy as np

from bravli.utils import get_logger

LOG = get_logger("dataset")


# ---------------------------------------------------------------------------
# Default loaders and saviors
# ---------------------------------------------------------------------------

def _default_loaders():
    """File-type → loader function mapping."""
    import json
    import yaml

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def load_yaml(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    return {
        "csv": pd.read_csv,
        "tsv": lambda p: pd.read_csv(p, sep="\t"),
        "json": load_json,
        "yaml": load_yaml,
        "feather": pd.read_feather,
    }


def _default_saviors():
    """File-type → save function mapping."""
    import json
    import yaml

    def write_json(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def write_yaml(data, path):
        with open(path, "w") as f:
            yaml.dump(data, f)

    return {
        "csv": lambda df, p: df.to_csv(p, index=False),
        "tsv": lambda df, p: df.to_csv(p, sep="\t", index=False),
        "json": write_json,
        "yaml": write_yaml,
    }


# ---------------------------------------------------------------------------
# Base Dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """A typed, named, lazy-loading scientific dataset.

    The minimum viable description of data: what is it called, what file
    format is it in, how do we load it, and how do we save it. The .value
    property lazy-loads the data on first access.

    Parameters
    ----------
    name : str
        Human-readable name.
    ftype : str
        File extension / format key (e.g., "csv", "tsv", "json").
    loader : callable, optional
        Function path → data. If None, resolved from ftype defaults.
    savior : callable, optional
        Function (data, path) → None. If None, resolved from ftype defaults.
    description : str, optional
        What this dataset contains.
    """

    name: str
    ftype: str
    loader: Callable = None
    savior: Callable = None
    description: str = None

    def __post_init__(self):
        self._loader_config = self.loader
        self._savior_config = self.savior

        if self.loader is None:
            self.loader = _default_loaders().get(self.ftype)
        if self.savior is None:
            self.savior = _default_saviors().get(self.ftype)

    def define(self):
        """Serializable definition of this dataset — for provenance."""
        return {
            "class": self.__class__.__qualname__,
            "name": self.name,
            "ftype": self.ftype,
            "description": self.description or "Not provided",
        }

    def load(self, path):
        """Load data from a path using this dataset's loader."""
        if self.loader is None:
            raise ValueError(f"No loader for dataset '{self.name}' (ftype={self.ftype})")
        LOG.info("Loading dataset '%s' from %s", self.name, path)
        self._value = self.loader(Path(path))
        self._path = Path(path)
        return self._value

    def save(self, data, path):
        """Save data to a path using this dataset's savior."""
        if self.savior is None:
            raise ValueError(f"No savior for dataset '{self.name}' (ftype={self.ftype})")
        LOG.info("Saving dataset '%s' to %s", self.name, path)
        return self.savior(data, Path(path))

    @property
    def value(self):
        """Lazy access to the dataset's data.

        On first access, loads from self._path if available.
        Subsequent accesses return the cached value.
        """
        try:
            return self._value
        except AttributeError:
            pass
        try:
            return self.load(self._path)
        except AttributeError:
            raise RuntimeError(
                f"Dataset '{self.name}' has no data and no path to load from. "
                "Call .load(path) first, or create a LocalDataset with an origin."
            )

    def with_data(self, data):
        """Attach in-memory data to this dataset. Returns self for chaining."""
        self._value = data
        return self

# ---------------------------------------------------------------------------
# LocalDataset
# ---------------------------------------------------------------------------

@dataclass
class LocalDataset(Dataset):
    """A dataset residing at a known local path.

    Parameters
    ----------
    origin : Path or str
        Where the data lives on disc.
    """

    origin: Any = None  # Path | str — using Any for Python 3.10 compat

    def __post_init__(self):
        if isinstance(self.origin, str):
            self.origin = Path(self.origin)
        if self.origin is not None:
            self._path = self.origin
        super().__post_init__()

    def define(self):
        definition = super().define()
        definition["origin"] = str(self.origin) if self.origin else None
        return definition

# ---------------------------------------------------------------------------
# CuratedDataset
# ---------------------------------------------------------------------------

@dataclass
class CuratedDataset(LocalDataset):
    """A dataset curated from scientific literature.

    Parameters
    ----------
    author : str
        Who assembled this dataset.
    source : str
        Where the data came from (paper DOI, database name, etc.).
    """

    author: str = None
    source: str = None

    def define(self):
        definition = super().define()
        definition["author"] = self.author
        definition["source"] = self.source
        return definition

# ---------------------------------------------------------------------------
# GeneratedDataset
# ---------------------------------------------------------------------------

@dataclass
class GeneratedDataset(Dataset):
    """A dataset derived from computation on other datasets.

    Parameters
    ----------
    inputs : list of Dataset
        The datasets this computation depends on.
    computation : callable
        A function that takes the input values and returns the output.
    params : dict, optional
        Additional keyword arguments to pass to the computation.
    """

    inputs: list = field(default_factory=list)
    computation: Callable = None
    params: dict = field(default_factory=dict)

    def generate(self):
        """Run the computation to produce this dataset's value."""
        if self.computation is None:
            raise ValueError(f"No computation defined for dataset '{self.name}'")

        input_values = []
        for inp in self.inputs:
            if isinstance(inp, Dataset):
                input_values.append(inp.value)
            else:
                input_values.append(inp)

        LOG.info("Generating dataset '%s' from %d inputs", self.name, len(input_values))
        self._value = self.computation(*input_values, **self.params)
        return self._value

    @property
    def value(self):
        """Lazy: generate on first access if not already computed."""
        try:
            return self._value
        except AttributeError:
            return self.generate()

    def define(self):
        definition = super().define()
        definition["inputs"] = [
            inp.define() if isinstance(inp, Dataset) else str(inp)
            for inp in self.inputs
        ]
        definition["computation"] = (
            f"{self.computation.__module__}.{self.computation.__qualname__}"
            if self.computation and hasattr(self.computation, "__qualname__")
            else str(self.computation)
        )
        definition["params"] = self.params
        return definition

# ---------------------------------------------------------------------------
# @evaluate_datasets — the transparent unwrapping decorator
# ---------------------------------------------------------------------------

def evaluate_datasets(method):
    """Decorator: unwrap Dataset arguments to their .value before calling.

    Any positional or keyword argument that has a .value attribute (i.e., is
    a Dataset) is replaced by its .value. All other arguments pass through
    unchanged. This lets domain functions accept either raw data or Dataset
    objects transparently.
    """

    def _unwrap(arg):
        try:
            return arg.value
        except AttributeError:
            return arg

    def wrapper(*args, **kwargs):
        unwrapped_args = tuple(_unwrap(a) for a in args)
        unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return method(*unwrapped_args, **unwrapped_kwargs)

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    wrapper.__wrapped__ = method
    return wrapper
