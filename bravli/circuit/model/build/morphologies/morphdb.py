"""
Help to access morphologies used to build the circuit.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from bluepy import Cell

from dmt.tk.field import Field, WithFields, lazyfield, field, NA


class CellQueryDB(WithFields):
    """
    Handle queries for a pandas DataFrame that has some columns containing
    cell properties.
    """

    def __init__(self, dataframe=None, *args, **kwargs):
        """..."""
        if dataframe is not None:
            self._raw = dataframe
        super().__init__(*args, **kwargs)

    @lazyfield
    def raw(self):
        """Raw data"""
        return NA

    @lazyfield
    def dataframe(self):
        """pandas DataFrame to hold the data."""
        return NA

    def as_key(self, query):
        """Convert `cell` to a key in the index"""
        def key(p):
            return (query[p] if query is not None and p in query and query[p]
                    else slice(None))
        return [key(p) for p in self.index]

    def loc(self, cell):
        """Positions in the index."""
        key = self.as_key(cell)

        try:
            loc = self.dataframe.index.get_locs(key)
        except KeyError as err:
            raise KeyError(f"No entries found for queried cell properties {cell}")

        return (key, loc)

    def get(self, cell):
        """
        Get morphology for `cell`.
        """
        try:
            key, locs = self.loc(cell)
        except KeyError:
            return None

        try:
            applicable = self.dataframe.iloc[locs]
        except KeyError:
            return None

        queried = [i for i, k in zip(self.index, key) if k != slice(None)]
        return applicable.reset_index()#.drop(columns=queried)

    def count(self, cell=None):
        """
        Number of morphologies that apply to a `cell`.
        """
        return len(self.dataframe.index.get_locs(self.as_key(cell)))



class MorphDB(CellQueryDB):
    """
    A class to load the `MorphDB` file that contains morphologies used to
    build a circuit.
    And some code to analyze morphologies in a circuit.
    """

    path_file = Field(
        """
        Path to the file to load data from.
        """,
        __required__=False)
    separator = Field(
        """
        Separator between the columns in the MorphDB file.
        """,
        __default_value__=r"\s+")
    header = Field(
        """
        True / False / None that indicates if the `MorphDB` contains a header.
        """,
        __default_value__=False)
    names = Field(
        """
        A list of names for the columns in the file that contains this MorphDB.
        Note that order of the names is important, and should match the order
        in the `MorphDB` file.
        """,
        __default_value__=["morphology", "layer", "mtype", "etype", "me_combo"])
                           #"electro_morphology"])
    index = Field(
        """
        A list of column names in the `MorphDB` file that will form the data's
        index. These names are attributes of a `Cell`, and should be a subset of
        `self.names`
        """,
        __default_value__=["layer", "mtype", "etype", "morphology"])

    @lazyfield
    def raw(self):
        """..."""
        assert self.path_file,\
            "Attribute Not Set: No path_file to load data from"

        read = pd.read_csv(self.path_file,
                           sep=self.separator,
                           header=self.header if self.header else None,
                           names=self.names)
        assert read.me_combo.nunique() == read.shape[0],\
            "Duplicate values for me_combo in data."
        return read


    @lazyfield
    def dataframe(self):
        """MEcombo Data"""
        return self.raw.set_index(self.index)

    def save(self, path):
        """
        Save as a TSV, without a header --- to confirm with what can be loaded.
        """
        self.raw.to_csv(path, sep=self.separator, header=False, index=False)


class MorphElectroComboes(CellQueryDB):
    """
    Query morpho-electro-physiological types.
    """
    path_file = Field(
        """
        that contains the mecombo --> memodels as a table.
        """)
    separator = Field(
        "Separator between the columns.",
        __default_value__=r"\s+")
    names = Field(
        """
        A list of names for the columns in the file that contains this MorphDB.
        Note that order of the names is important, and should match the order
        in the `MorphDB` file.
        """,
        __default_value__=["morph_name", "layer", "fullmtype", "etype",
                           "emodel", "combo_name",
                           "threshold_current", "holding_current"])
    renames = Field(
        """
        Column names to use.
        """,
        __default_value__=["morphology", "layer", "mtype", "etype",
                           "emodel", "me_combo",
                           "threshold_current", "holding_current"])
    index = Field(
        """
        A list of cell properties that can be used as a query.
        """,
        __default_value__=["layer", "mtype", "etype", "morphology", "me_combo"])
    emodel_variables = Field(
        """
        Columns in dataframe that provide emodel info.
        """,
        __default_value__=["emodel", "threshold_current", "holding_current"])

    @lazyfield
    def raw(self):
        """..."""
        read = (pd.read_csv(self.path_file, sep=self.separator, names=self.names,
                            header=0)
                .rename(columns=dict(zip(self.names, self.renames))))

        assert read.me_combo.nunique() == read.shape[0],\
            "Duplicate values for me_combo in data."
        return read


    @lazyfield
    def dataframe(self):
        """MEcombo Data"""
        return self.raw.set_index(self.index)

    def get(self, query):
        """
        Arugments
        ---------
        query : Either a string for me_combo, or Mapping(cell-property -> value).
        """
        if isinstance(query, Mapping):
            return super().get(query)

        emodels = self.get({"me_combo": query})

        if isinstance(query, str):
            assert emodels.shape[0] == 1
            return emodels.iloc[0]

        return emodels

    @lazyfield
    def map_mecombo_emodel(self):
        return self.raw.set_index("me_combo")[self.emodel_variables]

    def to_morphdb_like(self, this_one):
        return MorphDB(self.raw[this_one.names])

    def validate(self, morphdb):
        """Validate the entries of this MorphElectroComboes instance
        for a MorphDB instance.
        """
        raise NotImplementedError("TODO")
    
