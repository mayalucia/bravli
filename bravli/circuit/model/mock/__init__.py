"""
Mock circuit models that are based on a brain atlas.
"""
import numpy as np
import pandas as pd
from bluepy.v2.enums import Cell as CellProperty
from dmt.tk.journal import Logger
from dmt.tk.field import Field, lazyfield, WithFields, Record
from neuro_dmt.utils.geometry import Cuboid
from neuro_dmt.analysis.reporting import CircuitProvenance
from ..model import BlueBrainCircuitModel
from ...adapter import BlueBrainModelAdapter
