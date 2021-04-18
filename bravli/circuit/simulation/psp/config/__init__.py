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


class TargetSpecification(Enum):
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


def get_viable_pathways(circuit, target, min_number_connections):
    target_mtypes = circuit.cells.get({"$target": target}, "mtype")
    target_pathways = product(target_mtypes, target_mtypes)

    def is_viable(pathway):
        pre, post = pathway
        with_target = lambda mtype: {"$target": target, "mtype": mtype}
        pathway_with_target = (with_target(pre), with_target(post))
        connections = sample_connections(circuit, pathway_with_target,
                                         min_number_connections)

        LOGGER.info("\t with {} connections".format(len(connections)))

        return False if len(connections) < min_number_connections else True

    return [pathway for pathway in target_pathways if is_viable(pathway)]


def generate_configs(pathways,
                     connection_constraints,
                     simulation_protocol,
                     min_number_connections=None,
                     path_output=None):
    """
    Generate pathway configurations
    """
    from .config import Configuration

    min_number_connections = min_number_connections or 100

    LOGGER.status("Generate PSP  pathway configurations",
                  "connection-constraints: {}".format(connection_constraints),
                  "simulation protocol: {}".format(simulation_protocol))

    def get(pathway):
        pre, post = pathway
        config = Configuration(pathway={"pre": pre, "post": post,
                                        "constraints": connection_constraints},
                               protocol=simulation_protocol)

        if path_output:
            config.dump_yaml(path_output)

        return config

    configs = {}
    for n, pathway in enumerate(pathways):
        LOGGER.info("({}) Generate configuration: {}->{}".format(n, *pathway))
        configs[pathway] = get(pathway)

    return configs


def generate_psp_targets(pathways, cell_group=None, path_output=None):
    """..."""
    def parse(path):
        return tuple(path.name.split('.')[0].split('-'))

    if isinstance(pathways, Path):
        return generate_psp_targets([parse(p) for p in pathways.glob("*.yaml")],
                                    cell_group, path_output)

    mtypes = {mtype for pathway in pathways for mtype in pathway}

    cell_group = cell_group or {}

    def add_target(mtype):
        query = {k: v for k, v in cell_group.items()}
        query["mtype"] = mtype
        return query

    with_target = {mtype: add_target(mtype) for mtype in mtypes}

    if path_output:
        with open(path_output / "targets.yaml", 'w') as fptr:
            yaml.dump(with_target, fptr, allow_unicode=True)

    return with_target


def _generate_target_in_target(circuit,
                               connection_constraints,
                               simulation_protocol,
                               min_number_connections=None,
                               viable_pathways=None,
                               target=None,
                               path_output=None):
    """
    Generate configurations such that a cell target is placed in the targets configuration file.

    Arguments
    --------------
    target : named cell target.
    ~        if provided then viable pathways will be those that have at least
    ~        'min_number_connections' among target cells.
    ~        Thus, pathways will be viable for this target.
    ~        if `None` then viable pathways will be defined for the entire circuit.
    get_viable_pathways : A call-back to compute viable pathways.
    """

    min_number_connections = min_number_connections or 100

    LOGGER.status("Generate PSP configurations",
                  "connection-constraints: {}".format(connection_constraints),
                  "simulation protocol: {}".format(simulation_protocol))

    if not viable_pathways:
        assert target
        viable_pathways = get_viable_pathways(circuit, target,
                                              min_number_connections)

    LOGGER.status("Generate pathway configs")

    configs = generate_configs(viable_pathways,
                               connection_constraints,
                               simulation_protocol,
                               min_number_connections,
                               path_output)

    LOGGER.status("Generate targets")

    configs["targets"] = generate_targets(viable_pathways, target, path_output)

    return configs


def generate(circuit,
             connection_constraints,
             simulation_protocol,
             min_number_connections=None,
             viable_pathways=None,
             target=None,
             path_output=None,
             target_in=TargetSpecification.TARGETS):
    """..."""

    if target_in==TargetSpecification.PATHWAY:
        return _generate_target_in_pathway(circuit,
                                           connection_constraints,
                                           simulation_protocol,
                                           min_number_connections=min_number_connections,
                                           target_pathways=viable_pathways,
                                           target=target,
                                           path_output=path_output)

    if target_in==TargetSpecification.TARGETS:

        return _generate_target_in_target(circuit,
                                          connection_constraints,
                                          simulation_protocol,
                                          min_number_connections=min_number_connections,
                                          viable_pathways=viable_pathways,
                                          target=target,
                                          path_output=path_output)

    raise ValueError("Unknown target specification {}", target_in)
