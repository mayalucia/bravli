"""
Methods that query the circuit's atlas.
"""
import numpy as np

from dmt.tk.field import WithFields, lazyfield
from .query import SpatialQueryData, QueryDB  # disable pylint=relative-beyond-top-level


class CircuitAtlasAdapter(WithFields):
    """
    Adapt methods about the atlas underlying the circuit.
    """
    @lazyfield
    def visible_voxels(self):
        """
        Voxels in the atlas that were populated.
        If it is not masked, a voxel is visible.

        TODO: Where should queries to be handled by an atlas be processed?
        ~     If in a circuit model wrapper, where should the wrapper be defined?
        ~     Otherwise, we will need to include the code in the adapter.
        """

        def _get_visible_voxel_data(circuit, query):
            """..."""
            self.logger.debug(self.logger.get_source_info(),
                              """
                              Compute visible voxel data for query\n\t{}
                              """.format(query))
            visible = circuit.get_mask(**query)
            visible_voxel_ids = {tuple(ijk) for ijk in zip(*np.where(visible))}
            visible_cell_voxel_ids = list(
                visible_voxel_ids.intersection(
                    circuit.voxel_indexed_cell_gids.index.values))
            visible_cell_gids = circuit.voxel_indexed_cell_gids\
                                             .loc[visible_cell_voxel_ids]
            visible_voxel_positions =\
                circuit.get_voxel_positions(
                    np.array(list(visible_voxel_ids)))

            return SpatialQueryData(query=query,
                                    ids=visible_voxel_ids,
                                    positions=visible_voxel_positions,
                                    cell_gids=visible_cell_gids)

        return QueryDB(_get_visible_voxel_data)

    def get_voxel_indices(self, circuit, positions):
        """..."""
        return pd.DataFrame(circuit.atlas.positions_to_indices(positions),
                            columns=list("ijk"))

    def get_random_position(self, circuit, **query):
        """
        Generate a random position in the circuit location specified by
        keyword arguments `query`.
        """
        positions = self.visible_voxels(circuit, query).positions
        return positions.sample(n=1).iloc[0] if not positions.empty else None

    def get_layer_thickness_values(self, circuit,
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
        positions = (self.visible_voxels(circuit, spatial_query)
                     .positions.sample(n=sample_size, replace=True))
        return circuit.get_thickness(positions, relative=relative)

