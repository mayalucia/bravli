import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Cell
from dmt.tk.journal import Logger
from bravli.circuit.simulation import psp
from nmc_portal_resources_cli.utils import helper

LOGGER = Logger(client=__file__, level=Logger.Level.INFO)


def main(args):
    circuit = helper.load_circuit(args.path_mconfig)

    LOGGER.status("Generate pathways:\n {}".format(args))

    templates = args.templates or Path(__file__).parent / "templates"

    LOGGER.status("LOCATION {}".format(Path(__file__).parent))

    default_constraints = templates / "constraints" / "no-distance.yaml"
    default_protocol = templates / "protocols" / "single-spike.yaml"

    path_constraints = args.connection_constraints or default_constraints
    with open(path_constraints, "r") as fptr:
        connection_constraints = yaml.load(fptr, Loader=yaml.FullLoader)

    path_protocols = args.simulation_protocol or default_protocol
    with open(path_protocols, 'r') as f:
        simulation_protocol = yaml.load(f, Loader=yaml.FullLoader)


    path_output = Path(args.output) or Path.cwd()
    path_output.mkdir(exist_ok=True, parents=False)

    protocol = path_protocols.name.split('.')[0]

    LOGGER.status("Generate pathways",
                  "circuit mconfig {}".format(args.path_mconfig),
                  "regions {}".format(args.regions),
                  "central column {}".format(args.central_column),
                  "connection constraints {}".format(connection_constraints),
                  "simulation protocol {}".format(simulation_protocol),
                  "target {}".format(args.target),
                  "output {}".format(path_output))

    min_strength = np.int(args.min_number_connections or 100)
    viable_pathways = helper.connectome.get_viable_pathways(circuit,
                                                            region=args.regions,
                                                            central_column=args.central_column,
                                                            min_strength=min_strength,
                                                            as_tuples=True)
    LOGGER.status("number of viable pathways {}".format(len(viable_pathways)))

    LOGGER.status("Generate pathway configs")

    path_pathways = path_output / "pathways" / protocol
    path_pathways.mkdir(parents=True, exist_ok=True)
    psp.config.generate_configs(viable_pathways,
                                connection_constraints,
                                simulation_protocol,
                                min_strength,
                                path_pathways)
    if args.regions:
        targets = ["{}_Column".format(region) if args.central_column else region
                   for region in args.regions]
        LOGGER.status("Create PSP targets {}".format(targets))

        for target in targets:
            path_target = path_output / target
            path_target.mkdir(parents=False, exist_ok=True)
            psp.config.generate_targets(viable_pathways, target, path_target)


if __name__ == "__main__":

    LOGGER.status(format(sys.argv))

    parser = argparse.ArgumentParser(description="Generate pathway PSP configs")

    parser.add_argument("path_mconfig",
                        help="Path to the circuit's  mconfig that provides paths for circuit's assets.")

    parser.add_argument("-m", "--templates",
                        required=False, default=None,
                        help="Path to a dict that contains connection constraints and simulation protocols")
    parser.add_argument("-r", "--regions", nargs='+',
                        required=False, default=None,
                        help="Regions in the circuit where pathways should be viable.")

    parser.add_argument("--central-column", dest="central_column",
                        action="store_true",
                        help="Specify to consider only central columns in regions." )
    parser.set_defaults(central_column=False)

    parser.add_argument("-n", "--min-number-connections",
                        required=False, default=None,
                        help="Min number of connections in viable pathway.")

    parser.add_argument("-c", "--connection-constraints",
                        required=False, default=None,
                        help="YAML specifying constraints on each connection.")

    parser.add_argument("-p", "--simulation-protocol",
                        required=False, default=None)

    parser.add_argument("-t", "--target",
                        required=False, default=None,
                        help="String that represents a named cell target.")

    parser.add_argument("-o", "--output",
                        required=False, default=None,
                        help="Path to output")

    args = parser.parse_args()
    LOGGER.status(str(args))
    main(args)

