"""
Adapt BBP's neocortical models.
"""

from collections import OrderedDict
import numpy as np
import pandas as pd
import neurom as nm
from dmt.tk.field import Field, lazyfield, WithFields, Record, NA
from neuro_dmt.utils.geometry import Cuboid
from neuro_dmt import terminology
from .query import SpatialQueryData, QueryDB# disable pylint=relative-beyond-top-level
from ._connectome import CircuitConnectomeAdapter
from ._nodes import CircuitCellsAdapter
from ._atlas import CircuitAtlasAdapter

class BlueBrainModelAdapter(CircuitCellsAdapter,
                            CircuitConnectomeAdapter,
                            CircuitAtlasAdapter,
                            WithFields):
    """
    Adapt a circuit / simulation from the Blue Brain Project.
    """
    model_has_subregions = Field(
        """
        Does this brain region model contain sub-regions?
        """,
        __default_value__=False)

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
        return np.array(circuit_model.layers)

    def get_mtypes(self, circuit_model, config=None, **cell_type):
        """...

        TODO: Remove config -- that belongs in the analysis, not here.
        """
        try:
            custom_mtypes = config.mtypes

        except AttributeError:
            cells = self.get_cells(circuit_model, **cell_type)
            return cells.mtype.unique().to_list()
        else:
            try:
                mtypes = custom_mtypes(self, circuit_model, **cell_type)
            except TypeError:
                return custom_mtypes
            else:
                return mtypes
        raise RuntimeError("Execution should not reach here.")

    def get_regions(self, circuit_model, **cell_type):
        """
        Regions in the circuit_model.
        """
        cells = self.get_cells(circuit_model, **cell_type)
        return cells.region.unique().to_list()

    def get_layer_type(self, circuit_model):
        try:
            return circuit_model.layer_type
        except AttributeError:
            return "Cortical"
        raise RuntimeError("Execution should not reach here.")

    def get_etypes(self, circuit_model, config=None):
        """...
        TODO: Remove config -- that belongs in the analysis, not here.
        """
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

    def get_synapses_per_bouton(self, circuit_model):
        """..."""
        try:
            return circuit_model.synapses_per_bouton
        except AttributeError:
            pass
        return 1.0

    def get_cells(self, circuit_model,
                  properties=None,
                  target=None,
                  **query):
        """..."""
        listed = [properties] if isinstance(properties, str) else properties
        cells = circuit_model.get_cells(properties=listed,
                                        target=target,
                                        **query)
        query_atlas = terminology.circuit.atlas.filter(**query)
        if not query_atlas:
            return cells
        visible_cell_gids = self.visible_voxels(circuit_model, query_atlas)\
                                .cell_gids\
                                .values
        return cells.reindex(visible_cell_gids).dropna()

    def get_input_morphologies(self, circuit_model, for_cells=None):
        """Morphologies used to build the circuit for some cells."""
        return circuit_model.morphdb.get(for_cells)["morphology"]
