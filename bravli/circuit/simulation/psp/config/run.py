import sys
import argparse
from pathlib import Path
import yaml
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Cell
from dmt.tk.journal import Logger
from bravli.circuit.simulation import psp

LOGGER = Logger(client=__file__, level=Logger.Level.INFO)


def main(args):
    circuit = Circuit(str(Path(args.path_circuit) / "CircuitConfig"))
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


    path_target = Path(args.output or Path.cwd())
    if args.target:
        path_target = path_target / args.target
    else:
        path_target = path_target / "All"

    path_output = path_target / path_protocols.name.split('.')[0] / "pathways"
                   
    LOGGER.status("Generate pathways",
                  args.path_circuit,
                  str(connection_constraints),
                  str(simulation_protocol),
                  args.target,
                  str(path_output))
    psp.config.generate(circuit,
                        connection_constraints,
                        simulation_protocol, 
                        args.min_number_connections or 100,
                        target=args.target,
                        path_output=path_output)


if __name__ == "__main__":
    LOGGER.status(format(sys.argv))
    parser = argparse.ArgumentParser(description="Generate pathway PSP configs")
    parser.add_argument("path_circuit",
                        help="Path to the circuit to simulate.")
    parser.add_argument("-m", "--templates",
                        required=False, default=None,
                        help="Path to a dict that contains connection constraints and simulation protocols")
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

