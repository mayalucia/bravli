"""
Adapter of connectome methods
"""

import pandas as pd
from dmt.tk.field import Field, lazyfield, WithFields, Record, NA
from neuro_dmt.utils import ConnectomeType


class CircuitConnectomeAdapter(WithFields):
    """
    Extend an `CircuitModelAdapter` to handle connectome methods.
    """

    def get_connectome(self, circuit_model,
                       of_type=ConnectomeType.FUNCTIONAL):
        """..."""
        return (circuit_model.connectome if of_type==ConnectomeType.FUNCTIONAL
                else circuit_model.connectome_structural)

    def iter_connections(self, circuit_model,
                         of_type=ConnectomeType.FUNCTIONAL,
                         source_population=None,
                         target_population=None,
                         with_edge_ids=False,
                         with_edge_count=True,
                         shuffled=True):
        """
        Arguments
        --------------
        source_population :: Either a `pandas.Series` representing a cell
        ~                    or a `pandas.DataFrame` containing cells as rows
        ~                    or a `numpy.array` of cell gids.
        """
        return self.get_connectome(circuit_model, of_type=of_type)\
                   .iter_connections(pre=source_population,
                                     post=target_population,
                                     return_synapse_count=with_edge_count,
                                     return_synapse_ids=with_edge_ids,
                                     shuffle=shuffled)

    def iter_connections_structural(self, circuit_model, **query):
        """
        iterate through touches between two populations of nodes.
        """
        return self.iter_connections(circuit_model,
                                     of_type=ConnectomeType.STRUCTURAL,
                                     **query)
    def iter_connections_functional(self, circuit_model, **query):
        """..."""
        return self.iter_connections(self, circuit_model,
                                     of_type=ConnectomeType.FUNCTIONAL,
                                     **query,)

    def get_afferent_connections(self,circuit_model,
                                 post_synaptic,
                                 with_synapse_ids=False,
                                 with_synapse_count=True,
                                 of_type=ConnectomeType.FUNCTIONAL):
        """
        Arguments
        --------------
        post_synaptic :: Either a `pandas.Series` representing a cell
        ~                or a `pandas.DataFrame` containing cells as rows
        ~                or a `numpy.array` of cell gids.
        """
        iter_connections = self.iter_connections(circuit_model,
                                                 of_type=of_type,
                                                 target_population=post_synaptic,
                                                 with_edge_ids=with_synapse_ids,
                                                 with_edge_count=with_synapse_count)
        connections = np.array([connection for connection in iter_connections])
        if not with_synapse_count:
            return pd.DataFrame(connections[:, 0],
                                columns=["pre_gid"])
        return pd.DataFrame(connections,
                            columns=["pre_gid", "post_gid", "strength"])

    def get_efferent_connections(self, circuit_model,
                                 pre_synaptic,
                                 with_synapse_ids=False,
                                 with_synapse_count=True,
                                 of_type=ConnectomeType.FUNCTIONAL):
        """
        Arguments
        --------------
        pre_synaptic :: Either a `pandas.Series` representing a cell
        ~               or a `pandas.DataFrame` containing cells as rows
        ~               or a `numpy.array` of cell gids.
        """
        iter_connections = self.iter_connections(circuit_model,
                                                 of_type=of_type,
                                                 source_population=pre_synaptic,
                                                 with_edge_ids=with_synapse_ids,
                                                 with_edge_count=with_synapse_count)
        connections = np.array([connection for connection in iter_connections])
        if not with_synapse_count:
            return pd.DataFrame(connections[:, 1],
                                columns=["post_gid"])
        return pd.DataFrame(connections,
                            columns=["pre_gid", "post_gid", "strength"])

    def get_efferent_connection_strengths(self, circuit_model,
                                          pre_synaptic,
                                          of_type=ConnectomeType.FUNCTIONAL):
        """
        Arguments
        --------------
        pre_synaptic :: Either a `pandas.Series` representing a cell
        ~               or a `pandas.DataFrame` containing cells as rows
        ~               or a `numpy.array` of cell gids.
        """
        return self.get_efferent_connections(circuit_model, pre_synaptic,
                                             with_synapse_count=True,
                                             of_type=of_type)

    def count_axon_appositions(self, circuit_model, cell):
        """..."""
        connectome = self.get_connectome(circuit_model,
                                         of_type=ConnectomeType.STRUCTURAL)
        return len(connectome.efferent_synapses(self._resolve_one_gid(cell)))

    def count_efferent_synapses(self, circuit_model, cell):
        """..."""
        connectome = self.get_connectome(circuit_model,
                                         of_type=ConnectomeType.FUNCTIONAL)
        return len(connectome.efferent_synapses(self._resolve_one_gid(cell)))

    def get_pathways(self, circuit_model,
                     pre_synaptic=None,
                     post_synaptic=None):
        """
        Arguments
        -------------
        pre_synaptic :: Either an iterable of unique cell type specifiers
        ~               Or a mapping `cell_type-->value`
        post_synaptic :: Either an iterable of of unique cell type specifiers
        ~               Or a mapping `cell_type-->value`
        """
        if pre_synaptic is None:
            if post_synaptic is None:
                raise TypeError("""Missing arguments.
                Pass either `pre_synaptic`, or `post_synaptic` or both.""")
            try:
                pre_synaptic = post_synaptic.keys()
            except AttributeError:
                pre_synaptic = post_synaptic
        else:
            if post_synaptic is None:
                try:
                    post_synaptic = pre_synaptic.keys()
                except AttributeError:
                    post_synaptic = pre_synaptic

        def _at(synaptic_location, cell_type):
            return pd.concat([cell_type], axis=1,
                             keys=["{}_synaptic".format(synaptic_location)])

        pre_synaptic_cell_types = _at("pre",
                                      self.get_cell_types(circuit_model,
                                                          pre_synaptic))
        post_synaptic_cell_types = _at("post",
                                       self.get_cell_types(circuit_model,
                                                           post_synaptic))
        return pd.DataFrame([pre.append(post)
                             for _, pre in pre_synaptic_cell_types.iterrows()
                             for _, post in post_synaptic_cell_types.iterrows()])\
                 .reset_index(drop=True)
