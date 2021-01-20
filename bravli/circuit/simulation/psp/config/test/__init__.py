"""
Test develop PSP configurations.
"""
from collections.abc import Iterable
from pathlib import Path
import yaml
from dmt.tk.journal import Logger
from bravli.circuit.simulation import psp

LOGGER = Logger(client=__file__)

def generate(circuit, pathways, path_output=None,
             connection_constraints=None, simulation_protocol=None):
    """..."""
    templates = Path(__file__).parent.parent/"templates"
    protocols = templates / "protocols"
    constraints = templates / "constraints"

    def read(filepath):
        with open(filepath, 'r') as fptr:
            return yaml.load(fptr, Loader=yaml.FullLoader)

    protocol = read(protocols / (simulation_protocol or "single-spike.yaml"))
    LOGGER.debug("PSP protocol {}".format(protocol))

    constraint = read(constraints / (connection_constraints or "no-distance.yaml"))
    LOGGER.debug("PSP constraint {}".format(constraint))

    return psp.config.generate(circuit, constraint, protocol,
                               target_pathways=pathways,
                               path_output=path_output)

    


