"""
Initial Human Neocortical Microcircuit. (2021)
"""
from collections import OrderedDict
from pathlib import Path
from bravli.circuit.analysis.configuration import CircuitAnalysisConfiguration

PROJ = Path("/gpfs/bbp.cscs.ch/project/proj71")

CIRCUITS = PROJ / "circuits" / "O1"

CONFIG = CircuitAnalysisConfiguration()

CONFIG.available_circuits = OrderedDict([("P0.v1.0", CIRCUITS / "20200422"),
                                         ("P1.rc1.0", CIRCUITS / "phase-1" / "20201224")])
