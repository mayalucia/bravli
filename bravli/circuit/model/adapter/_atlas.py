"""
Methods that query the circuit's atlas.
"""
import numpy as np
import pandas as pd

from dmt.tk.field import WithFields, lazyfield
from .query import SpatialQueryData, QueryDB  # disable pylint=relative-beyond-top-level

from neuro_dmt import terminology

X = terminology.bluebrain.cell.x
Y = terminology.bluebrain.cell.y
Z = terminology.bluebrain.cell.z
XYZ = [X, Y, Z]


class CircuitAtlasAdapter(WithFields):
    """
    Adapt methods about the physical space modeled as a brain-atlas
    that contains the circuit.
    """

    def get_orientations(self, circuit, positions):
        """
        Get the orientations for voxel-positions in the circuit model atlas.

        Arguments
        -------------
        positions: np.array<N * 3>
        """
        return pd.DataFrame(circuit.orientation_field.lookup(positions)[:, :, 1],
                            columns=XYZ)


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
            visible_voxel_pos = (circuit
                                 .get_voxel_positions(np.array(list(visible_voxel_ids))))

            try:
                voxel_cell_gids = circuit.voxel_indexed_cell_gids
            except AttributeError:
                pass
            else:
                visible_cell_voxel_ids = list(visible_voxel_ids
                                              .intersection(voxel_cell_gids
                                                            .index.values))
                visible_cell_gids = (voxel_cell_gids
                                     .loc[visible_cell_voxel_ids])

                return SpatialQueryData(query=query,
                                        ids=visible_voxel_ids,
                                        positions=visible_voxel_pos,
                                        cell_gids=visible_cell_gids)

            return SpatialQueryData(query=query,
                                    ids=visible_voxel_ids,
                                    positions=visible_voxel_pos)

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
                                   sample_size=None,
                                   **spatial_query):
        """
        Get layer thickness sample for the region specified in keyword arguments
        `spatial_query`.

        Thickness will be computed for all voxels visible for the spatial query.
        Measurement is made on a sample of positions in the specified region.

        If `relative=True`, result will be thickness divided by total cortical
        thickness.
        """
        voxels = self.visible_voxels(circuit, spatial_query)
        if voxels.empty:
            return None


        positions = (voxels.positions.sample(n=sample_size, replace=False)
                     if sample_size and sample_size < voxels.shape[0] else
                     voxels.positions)
        return circuit.get_thickness(positions, relative=relative)
