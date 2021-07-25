"""
Help to access morphologies used to build the circuit.
"""

import numpy as np
import pandas as pd

from bluepy import Cell

from dmt.tk.field import Field, WithFields, lazyfield


class MorphFarm(WithFields):
    """
    Grows morphologies for a cell.
    """
    pass
    

class MorphDB(MorphFarm):
    """
    A class to load the `MorphDB` file that contains morphologies used to
    build a circuit.
    And some code to analyze morphologies in a circuit.
    """

    path_file = Field(
        """
        Path to the file that contains the data to be loaded.
        """)
    separator = Field(
        """
        Separator between the columns in the MorphDB file.
        """,
        __default_value__=' ')
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
        __default_value__=["morphology", "layer", "mtype", "etype",
                           "electro_morphology"])
    index = Field(
        """
        A list of column names in the `MorphDB` file that will form the data's
        index. These names are attributes of a `Cell`, and should be a subset of
        `self.names`
        """,
        __default_value__=["layer", "mtype", "etype", "morphology"])


    @lazyfield
    def dataframe(self):
        """Data associated with this MorphDB"""
        return pd.read_csv(self.path_file,
                           sep=self.separator,
                           header=self.header if self.header else None,
                           names=self.names)\
                 .set_index(self.index)

    def as_key(self, cell):
        """Convert `cell` to a key in the index"""
        def key(p):
            return (cell[p] if cell is not None and p in cell and cell[p]
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
        return applicable.reset_index().drop(columns=queried)

    def count(self, cell=None):
        """
        Number of morphologies that apply to a `cell`.
        """
        return len(self.dataframe.index.get_locs(self.as_key(cell)))
