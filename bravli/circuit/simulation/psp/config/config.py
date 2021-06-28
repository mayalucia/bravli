"""
Configure PSP simulations.
"""
from pathlib import Path
import yaml
import pandas as pd
from dmt.tk.field import NA, Field, LambdaField, WithFields


class Configuration(WithFields):
    """
    Configuration of a simulation to measure PSP traces for a pathway.
    A configuration can be loaded as a YAML.
    Or a configuration may be defined as a mapping or an instance of this class.

    """
    class Pathway(WithFields):
        """
        A single PSP simulation involves a current input from the pre-synaptic
        cell that evokes a voltage potential in the post-synaptic cell.
        We analyze properties of PSP for given pathways.
        """

        class Constraints(WithFields):
            """
            Each pathway will have it's own constraints
            """
            unique_gids = Field(
                """
                Boolean, `True` if unique gids should be used in sampling
                cell pairs.
                """)
            max_dist_x = Field(
                """
                Maximum distance between pair cells along the x-axis.
                """,
                __required__=False)
            max_dist_y = Field(
                """
                Maximum distance between pair cells along the y-axis.
                """,
                __required__=False)
            max_dist_z = Field(
                """
                Maximum distance between pair cells along the z-axis.
                """,
                __required__=False)


        pre = Field(
            """
            The pre-synaptic type in this `Pathway` instance.
            """)
        post = Field(
            """
            The post-synaptic type in this `Pathway` instance.
            """)
        constraints = Field(
            """
            Constraints associated with this `Pathway` instance.
            """,
            __as__=Constraints)
        label = LambdaField(
            """
            Label used for this `Pathway` instance.
            """,
            lambda self: "{}-{}".format(self.pre, self.post))


    class Protocol(WithFields):
        """
        ...to follow to run the simulation.
        """
        record_dt = Field(
            """
            Time step for recording the voltage.
            """)
        hold_V = Field(
            """
            Holding voltage.
            """)
        t_stim = Field(
            """
            Start time of stimulation.
            """)
        t_stop = Field(
            """
            Stop time of stimulation.
            """)
        post_ttx = Field(
            """
            Boolean indicating if simulation should be run with a TTX treatment
            on the post-synaptic side.
            """)


    class Reference(WithFields):
        """
        A simulation configuration may contain reference data.
        """
        author = Field(
            """
            Author(s) of the reference data.
            """)
        psp_amplitude = Field(
            """
            Statistical summary (mean and std), as a mapping or `pandas.Series`.
            """,
            __as__=pd.Series)
        synapse_count = Field(
            """
            Statistical summary (mean and std), as a mapping or `pandas.Series`.
            """,
            __as__=pd.Series,
            __default_value__=NA)


    pathway = Field(
        """
        The mtype-->mtype pathway to run the simulation for.
        """,
        __as__=Pathway)
    protocol = Field(
        """
        The protocol for the simulation to be run.
        """,
        __as__=Protocol)
    reference = Field(
        """
        Experimental reference providing expected measurements.
        """,
        __as__=Reference,
        __default_value__=NA)

    def dump_yaml(self, path, file_name=None):
        """
        Dump a YAML.

        Arguments
        --------------
        path: Directory to dump in.
        file_name: Name of the file to dump in.
        """
        def get_mtype(group):
            try:
                return group["mtype"]
            except TypeError:
                try:
                    return group.mtype
                except AttributeError:
                    pass
            return group

        if not file_name:
            file_name = "{}-{}.yaml".format(get_mtype(self.pathway.pre),
                                            get_mtype(self.pathway.post))

        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)

        path_file =  dirpath / file_name
        with open(path_file, 'w') as f:
            yaml.dump(self.field_dict, f, allow_unicode=True)
        return path_file


