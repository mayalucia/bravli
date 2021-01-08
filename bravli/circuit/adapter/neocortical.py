"""
Adapt BBP's neocortical models.
"""

from collections import OrderedDict
import numpy as np
import pandas as pd
from dmt.tk.field import Field, lazyfield, WithFields, Record, NA
from dmt.tk.collections import get_list
from neuro_dmt.utils.geometry import Cuboid
from neuro_dmt.utils import ConnectomeType
from neuro_dmt import terminology
from .query import SpatialQueryData, QueryDB# disable pylint=relative-beyond-top-level

X = terminology.bluebrain.cell.x
Y = terminology.bluebrain.cell.y
Z = terminology.bluebrain.cell.z
XYZ = [X, Y, Z]

class BlueBrainModelAdapter(WithFields):
    """
    Adapt a circuit / simulation from the Blue Brain Project.
    """
    random_position_generator = Field(
        """
        A (nested) dict mapping circuit, and a spatial query to their
        random position generator. 
        """,
        __default_value__={})

    @staticmethod
    def _resolve_circuit(model):
        try:
            return model.circuit
        except AttributeError:
            return model
        return None

    def get_namespace(self, model):
        """
        A namespace providing values for circuit properties required by analyses.
        """
        circuit_model = self._resolve_circuit(model)
        return {"layer-values": self.get_layers(circuit_model),
                "layer-type": self.get_layer_type(circuit_model),
                "region": self.get_brain_region(circuit_model),
                "sub-regions": self.get_sub_regions(circuit_model),
                "animal": self.get_animal(circuit_model)}

    def get_provenance(self, model):
        """
        Mapping providing information about the circuit model's provenance.
        """
        return self._resolve_circuit(model).provenance.field_dict

    def get_label(self, circuit_model):
        """..."""
        try:
            return circuit_model.label
        except AttributeError:
            return circuit_model.__class__.__name__
        return NA

    def get_animal(self, circuit_model):
        """
        The animal modeled in the circuit.
        """
        provenance = self.get_provenance(circuit_model)
        try:
            return provenance["animal"]
        except KeyError:
            try:
                return circuit_model.animal
            except AttributeError:
                pass
            pass
        return NA

    def get_brain_area(self, circuit_model):
        """
        Label for the area of the brain modeled.
        For example, "SSCx"
        """
        provenance = self.get_provenance(circuit_model)
        try:
            return provenance["brain_area"]
        except KeyError:
            try:
                return provenance["brain_region"]
            except KeyError:
                pass
            pass
        return NA
 
    def get_brain_region(self, circuit_model):
        """
        Label for the brain region modeled.
        """
        return circuit_model.provenance.brain_region

    def get_sub_regions(self, circuit_model):
        """..."""
        return (list(circuit_model.cells.region.unique())
                if self.model_has_subregions else
                [self.get_brain_region(circuit_model)])
    @staticmethod
    def _prefix_L(layer):
        if isinstance(layer, (int, np.int16, np.int32, np.int64, np.integer)):
            return "L{}".format(layer)
        if isinstance(layer, str):
            if layer[0] != "L":
                return "L{}".format(layer)
            return layer
        raise TypeError(
            "Bad type {} of layer value {} ".format(type(layer), layer))

    def get_layers(self, circuit_model):
        return sorted(self._prefix_L(layer)
                       for layer in circuit_model.cells.layer.unique())

    @lazyfield
    def visible_voxels(self):
        """
        Voxels in the atlas that were populated.
        If it is not masked, a voxel is visible.

        TODO: Where should queries to be handled by an atlas be processed?
        ~     If in a circuit model wrapper, where should the wrapper be defined?
        ~     Otherwise, we will need to include the code in the adapter.
        """

        def _get_visible_voxel_data(circuit_model, query):
            """..."""
            self.logger.debug(self.logger.get_source_info(),
                              """
                              Compute visible voxel data for query\n\t{}
                              """.format(query))
            visible = circuit_model.get_mask(**query)
            visible_voxel_ids = {tuple(ijk) for ijk in zip(*np.where(visible))}
            visible_cell_voxel_ids = list(
                visible_voxel_ids.intersection(
                    circuit_model.voxel_indexed_cell_gids.index.values))
            visible_cell_gids = circuit_model.voxel_indexed_cell_gids\
                                             .loc[visible_cell_voxel_ids]
            visible_voxel_positions =\
                circuit_model.get_voxel_positions(
                    np.array(list(visible_voxel_ids)))

            return SpatialQueryData(query=query,
                                    ids=visible_voxel_ids,
                                    positions=visible_voxel_positions,
                                    cell_gids=visible_cell_gids)

        return QueryDB(_get_visible_voxel_data)

    def get_voxel_indices(self, circuit_model, positions):
        """..."""
        return pd.DataFrame(circuit_model.atlas.positions_to_indices(positions),
                            columns=list("ijk"))

    def get_random_position(self, circuit_model, **query):
        """
        Generate a random position in the circuit location specified by
        keyword arguments `query`.
        """
        positions = self.visible_voxels(circuit_model, query).positions
        return positions.sample(n=1).iloc[0] if not positions.empty else None

    def get_layer_thickness_values(self, circuit_model,
                                   relative=False,
                                   sample_size=10000,
                                   **spatial_query):
        """
        Get layer thickness sample for the region specified in keyword arguments
        `spatial_query`.

        Thickness will be computed for all voxels visible for the spatial query.
        Measurement is made on a sample of positions in the specified region.

        If `relative=True`, result will be thickness divided by total cortical
        thickness.
        """
        positions = self.visible_voxels(circuit_model, spatial_query)\
                        .positions\
                        .sample(n=sample_size, replace=True)
        return circuit_model.get_thickness(positions, relative=relative)

    def get_layer_type(self, circuit_model):
        try:
            return circuit_model.layer_type
        except AttributeError:
            return "Cortical"
        raise RuntimeError("Execution should not reach here.")

    def get_mtypes(self, circuit_model, config=None, **cell_type):
        """..."""
        try:
            custom_mtypes = config.mtypes
        except AttributeError:
            cells = self.get_cells(circuit_model, **cell_type)
            return cells.mtype.unique().to_list()
        else:
            try:
                mtypes = custom_mtypes(self, circuit_model, **kwargs)
            except TypeError:
                return custom_mtypes
            else:
                return mtypes
        raise RuntimeError("Execution should not reach here.")

    def get_etypes(self, circuit_model, config=None):
        """..."""
        try:
            custom_etypes = config.etypes
        except AttributeError:
            return list(circuit_model.cells.etype.unique())
        else:
            try:
                etypes = custom_etypes(self, circuit_model, **kwargs)
            except TypeError:
                return custom_etypes
            else:
                return etypes
        raise RuntimeError("Execution should not reach here.")

    def get_spatial_volume(self, circuit_model, **spatial_query):
        """
        Get total spatial volume of the circuit space that satisfies a spatial
        query.
        """
        count_voxels = circuit_model.get_voxel_count(**spatial_query)
        return count_voxels * circuit_model.volume_voxel

    def _resolve_gids(self, circuit_model, cell_group):
        """
        Resolve cell gids...
        """
        if cell_group is None:
            return None
        if isinstance(cell_group, np.ndarray):
            gids = cell_group
        elif isinstance(cell_group, list):
            gids = np.ndarray(cell_group)
        elif isinstance(cell_group, pd.Series):
            try:
                gids = np.array([cell_group.gid])
            except AttributeError:
                gids = self.get_cells(circuit_model, **cell_group).gid.values
        elif isinstance(cell_group, pd.DataFrame):
            gids = cell_group.gid.values
        else:
            raise ValueError(
                """
                Could not resolve gids from object {}
                """.format(cell_group))
        return gids

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

    def get_connections(self, circuit_model,
                        cell_group,
                        direction,
                        of_type=ConnectomeType.FUNCTIONAL,
                        with_synapse_ids=False,
                        with_synapse_count=True):
        """..."""
        if with_synapse_ids and with_synapse_count:
            raise TypeError(
                """
                `get_connections(...)` called requesting both synapse ids and
                synapse count. Only one of these may be requested.
                """
            )
        _get =(self.get_afferent_connections
               if direction in ("AFF", "afferent", "aff") else
               self.get_efferent_connections)
        return _get(circuit_model, cell_group,
                    with_synapse_ids,
                    with_synapse_count,
                    of_type)

    def get_cells(self, circuit_model,
                  properties=None,
                  target=None,
                  **query):
        """..."""
        cells = circuit_model.get_cells(properties=properties,
                                        target=target,
                                        **query)
        query_atlas = terminology.circuit.atlas.filter(**query)
        if not query_atlas:
            return cells

        visible_cell_gids = self.visible_voxels(circuit_model, query_atlas)\
                                .cell_gids\
                                .values
        return cells.reindex(visible_cell_gids)\
                    .dropna()

    def get_cell_types(self, circuit_model, query):
        """
        Get cell-types in the circuit.

        Arguments
        -------------
        query :: Either an iterable of unique cell type specifiers.
        ~        Or a mapping cell_type_specifier --> value / list(value)

        Returns
        -------------
        A `pandas.DataFrame` containing all cell-types,
        each row providing values for each cell type specifier.
        """
        try:
            query_items = query.items()
        except AttributeError:
            query_with_values = {}
        else:
            query_with_values = {cell_property: value
                                 for cell_property, value in query_items
                                 if value is not None}
        def _values(variable):
            """..."""
            try:
                raw_values = query_with_values[variable]
            except KeyError:
                try:
                    get_values = getattr(self, "get_{}s".format(variable))
                except AttributeError as error:
                    raise AttributeError(
                        """
                        {} adapter does not implement a getter for cell property
                        {}
                        """).format(self.__class__.__name__, variable)
                raw_values = get_values(circuit_model)

            return get_list(raw_values)

        def _get_tupled_values(params):
            """..."""
            if not params: return [[]]

            head_tuples = [[(params[0], value)] for value in _values(params[0])]
            tail_tuples = _get_tupled_values(params[1:])
            return [h + t for h in head_tuples for t in tail_tuples]

        try:
            keys = query.keys()
        except AttributeError:
            keys = query
        specifiers_cell_type = tuple(keys)

        return pd.DataFrame([dict(row)
                             for row in _get_tupled_values(specifiers_cell_type)])

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

    #TODO: remove these functions
    def _get_cell_density_overall(self,
            circuit_model,
            **query_parameters):
        """
        Get cell density over the entire relevant volume.

        Pass only keyword arguments that are accepted for cell queries by
        the circuit model.
        """
        query_spatial = {
            key: query_parameters[key]
            for key in ["region", "layer", "depth", "height"]
            if key in query_parameters}
        count_cells = circuit_model.cells.get(
            self.circuit_model.query_cells(**query_parameters)
        ).shape[0]
        count_voxels = circuit_model.atlas.count_voxels(**query_spatial)
        return count_cells/(count_voxels*1.e-9*circuit_model.atlas.voxel_volume)

    def get_orientations(self, model, positions):
        """
        Get the orientations for voxel-positions in the circuit model atlas.

        Arguments
        -------------
        positions: np.array<N * 3>
        """
        return pd.DataFrame(model.orientation_field.lookup(positions)[:, :, 1],
                            columns=XYZ)
