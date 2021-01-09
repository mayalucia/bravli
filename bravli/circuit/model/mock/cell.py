"""
Mock the behavior of a cell.
"""

from abc import\
    abstractmethod,\
    abstractproperty,\
    abstractclassmethod,\
    abstractstaticmethod
from collections.abc import Mapping
from collections import OrderedDict
import numpy as np
import pandas as pd
from bluepysnap.bbp import Cell as CellProperty
from dmt.tk.field import NA, Record, Field, LambdaField, lazyfield, Property\
    WithFields, ABCWithFields

cell_properties = [CellProperty.ID,
                   CellProperty.LAYER,
                   CellProperty.MTYPE,
                   CellProperty.ETYPE,
                   CellProperty.MORPHOLOGY,
                   CellProperty.ME_COMBO,
                   CellProperty.REGION,
                   CellProperty.X, CellProperty.Y, CellProperty.Z,
                   CellProperty.SYNAPSE_CLASS]


class Cell(ABCWithFields):
    raise NotImplementedError(
        """
        What should a `Cell` object provide?
        Should it be defined through it's `Fields`?
        Is it even possible to define through it's `Fields`?
        It will come down to it's behavior --- i.e the method calls it supports.

        The other option is to define `Cell` as a data loader.
        But the fact is that a cell's data needs to be gathered from several
        sources.

        The details of these data can be hidden behind the interface of
        a `Cell`, just like a `Circuit`.
        """)
