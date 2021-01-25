"""
Configure PSP simulations on a circuit.
"""
from itertools import islice, product
from enum import Enum
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



class TargetSpecifiction(Enum):
    """
    A cell target can be placed in the pathway configuration or in targets.
    """
    PATHWAY = 0
    TARGETS = 1


def _generate_target_in_pathway(circuit,
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


def get_cell_groups(pathways, target=None, path_output=None):
    """..."""
    if isinstance(pathways, Path):
        def parse(path):
            return tuple(path.name.split('.')[0].split('-'))

        return get_cell_groups([parse(p) for p in pathways.glob("*.yaml")],
                               target, path_output)

    mtypes = {mtype for pathway in pathways for mtype in pathway}

    def add_target(mtype):
        query =  {"mtype": mtype}
        if target:
            query["$target"] = target
        return query

    with_target = {mtype: add_target(mtype) for mtype in mtypes}

    if path_output:
        with open(path_output/"targets.yaml", 'w') as fptr:
            yaml.dump(with_target, fptr, allow_unicode=True)
    return with_target


def _generate_target_in_target(circuit,
                               connection_constraints,
                               simulation_protocol,
                               min_number_connections=None,
                               target=None,
                               path_output=None):
    """
    Generate configurations such that a cell target is placed in the targets configuration file.
    """
    from .config import Configuration

    min_number_connections = min_number_connections or 100

    LOGGER.status("Generate PSP configurations",
                  "connection-constraints: {}".format(connection_constraints),
                  "simulation protocol: {}".format(simulation_protocol))

    if target:
        target_cells = circuit.cells.get({"$target": target})
    else:
        target_cells = circuit.cells.get()

    target_mtypes = target_cells.mtype.unique()
    target_pathways = [(pre, post) for pre, post in product(target_mtypes,
                                                                target_mtypes)]
    def get(pathway):
        pre, post = pathway

        def _with_target(mtype):
            query = {"mtype": mtype}
            if target:
                query["$target"] = target
            return query

        if not is_viable(circuit, (_with_target(pre), _with_target(post)),
                         min_number_connections):
            LOGGER.info("\t\t Pathway is not viable.")
            return None

        config = Configuration(pathway={"pre": pre, "post": post,
                                        "constraints": connection_constraints},
                               protocol=simulation_protocol)

        if path_output:
            config.dump_yaml(path_output)
        return config

    yamls = {}
    for n, pathway in enumerate(target_pathways):
        LOGGER.info("({}). Generate configuration: {}->{}".format(n, *pathway))
        yamls[pathway] = get(pathway)

    yamls["targets"] = get_cell_groups(target_pathways, target, path_output)

    return yamls

def generate(circuit,
             connection_constraints,
             simulation_protocol,
             min_number_connections=None,
             target_pathways=None,
             target=None,
             path_output=None,
             target_in=TargetSpecifiction.PATHWAY):
    """..."""
    if target_in ==  TargetSpecification.PATHWAY:
        return _generate_target_in_pathway(circuit, connection_constraints, simulation_protocol,
                                           min_number_connections=min_number_connections,
                                           target_pathways=target_pathways,
                                           target=target,
                                           path_output=path_output)
    if target_in == TargetSpecifiction.TARGETS:
        if target_pathways is not None:
            raise NotImplementedError("the case of target in target")

        return _generate_target_in_target(circuit, connection_constraints, simulation_protocol,
                                          min_number_connections=min_number_connections,
                                          target=target,
                                          path_output=path_output)

    raise ValueError("Unknown target specification {}", target_in)
