"""
Configure PSP simulations on a circuit.
"""
from itertools import islice, product
from pathlib import Path
import yaml
from bluepy.v2.enums import Cell
from dmt.tk.journal import Logger
from .config import Configuration

LOGGER = Logger(client=__file__)

def sample_connections(circuit, pathway, n, shuffle=False):
    LOGGER.debug("sample connections for {}".format(pathway))
    pre, post = pathway
    conn = circuit.connectome
    iterc = conn.iter_connections(pre=pre, post=post, shuffle=shuffle)
    return list(islice(iterc, 0, n))


def is_viable(circuit, pathway, min_number_connections):
    """..."""
    connections = sample_connections(circuit, pathway, min_number_connections)
    LOGGER.info("\t with {} connections".format(len(connections)))
    return False if len(connections) < min_number_connections else True


def generate(circuit, 
             connection_constraints,
             simulation_protocol,
             min_number_connections=None,
             target_pathways=None,
             target=None,
             path_output=None):
    """
    Generate configurations for PSP simulations.

    Arguments
    --------------
    circuit: BluePyCircuit TODO: Implement with adapter pattern
    target: A named cell target TODO: Generalize target to be a Mapping.
    """
    from .config import Configuration

    min_number_connections = min_number_connections or 100

    LOGGER.status("Generate PSP configurations",
                  "connection-constraints: {}".format(connection_constraints),
                  "simulation protocol: {}".format(simulation_protocol))

    if target_pathways is None:
        if target:
            target_cells = circuit.cells.get({"$target": target})
            cell_group = lambda mtype: {"$target": target, "mtype": mtype}
        else:
            target_cells = circuit.cells.get()
            cell_group = lambda mtype: {"mtype": mtype}

        target_mtypes = target_cells.mtype.unique()
        target_pathways = [(cell_group(pre), cell_group(post))
                           for pre, post in product(target_mtypes, target_mtypes)]
    else:
        assert target is None

    def get(pathway):
        pre, post = pathway
        if not is_viable(circuit, pathway, min_number_connections):
            LOGGER.info("\t\t Pathway is not viable.")
            return None
        LOGGER.info("\t\t Pathway is viable.")
        config = Configuration(pathway={"pre": pre, "post": post,
                                        "constraints": connection_constraints},
                               protocol=simulation_protocol)
        if path_output:
            config.dump_yaml(path_output)
        return config

    def label(pathway):
        pre, post = pathway
        return (pre["mtype"], post["mtype"])

    yamls = {}
    for n, pathway in enumerate(target_pathways):
        LOGGER.info("({}). Generate configuration: {}->{}".format(n, *pathway))
        yamls[label(pathway)] = get(pathway)

    return yamls
