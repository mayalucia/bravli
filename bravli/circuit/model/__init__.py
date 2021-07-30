"""
Wrap a BBP circuit to provide extended functionality for analyses.
"""
from collections.abc import Iterable
import os
from copy import deepcopy
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import neurom
from bluepy.v2.circuit import Circuit as BluePyCircuit
from bluepy.exceptions import BluePyError
from bluepy.v2.enums import Cell, Segment, Section
from dmt.tk import collections
from dmt.tk.field import Field, LambdaField, lazyfield, WithFields, Record, NA
from dmt.tk.journal import Logger
from dmt.tk.collections import take
from neuro_dmt import terminology
from neuro_dmt.analysis.reporting import CircuitProvenance
from .build.morphologies.morphdb import MorphDB
from .atlas import BlueBrainCircuitAtlas

XYZ = [Cell.X, Cell.Y, Cell.Z]
LAYER = Cell.LAYER

LOGGER = Logger(client=__file__)

def _get_bounding_box(roi):
    """Allows a tuple to be a region of interest."""
    try:
        return roi.bbox
    except AttributeError:
        return roi
    return None


class BlueBrainCircuitModel(WithFields):
    """
    A circuit model developed at the Blue Brain Project.

    Circuit model's data is available as HDF5 files.
    `BluePy` provides an API to load this data and present as a `bluepy.Circuit`
    object. `BlueBrainCircuitModel` extend's `bluepy.Circuit`'s API.
    """
    provenance = Field(
        """
        `CircuitProvenance` instance describing the circuit model
        """,
        __default_value__=CircuitProvenance(
            label="BlueBrainCircuitModel",
            authors=["BBP Team"],
            date_release="Not Available",
            uri="Not Available",
            animal="Not Available",
            age="Not Available",
            brain_region="Not Available"))
    label = Field(
        """
        A label to represent your circuit model instance.
        """,
        __default_value__="BlueBrainCircuitModel")
    path_circuit_data = Field(
        """
        Path to the location of this circuit's data. This data is loaded as a
        BluePy circuit if a Bluepy circuit is not provided at initialization.
        """,
        __default_value__="not-available")
    circuit_config_structural = Field(
        """
        Sometimes we work with circuit structural connectome.
        """,
        __default_value__="CircuitConfig_struct")
    circuit_config_functional = Field(
        """
        Sometimes we work with circuit structural connectome.
        """,
        __default_value__="CircuitConfig")
    circuit_config = LambdaField(
        """
        Name of the file (under `.path_circuit_data`) that contains the
        circuit's configuration and provides paths to the data containing
        cells and connectome.
        """,
        lambda self: self.circuit_config_functional)
    circuit_config_base = Field(
        """
        After the first phase of circuit build, that creates a cell collection,
        a basic circuit config file is created in the circuit directory.
        """,
        __default_value__="CircuitConfig_base")
    path_mconfig = Field(
        """
        A YAML that contains location of data from measurements made on the
        circuit.
        """,
        __default_value__=NA)

    def __init__(self, circuit=None, *args, **kwargs):
        """
        Initialize with a circuit.

        Arguments
        -------------
        circuit: A BluePy circuit.
        """
        if circuit:
            if isinstance(circuit, BluePyCircuit):
                bluepy_circuit = circuit
            else:
                try:
                    bluepy_circuit = BluePyCircuit(str(circuit))
                except IsADirectoryError:
                    bluepy_circuit = None
                    kwargs["path_circuit_data"] = circuit
                else:
                    bluepy_circuit = bluepy_circuit
            if bluepy_circuit:
                self._bluepy_circuit = bluepy_circuit
        super().__init__(*args, **kwargs)

    def get_path(self, *relative_path):
        """
        Absolute path to the file at `relative_path`

        Arguments
        ----------
        `relative_path`: Sequence of strings describing a path relative to
        the circuit's location.
        """
        return Path(self.path_circuit_data).joinpath(*relative_path)

    @lazyfield
    def biodata(self):
        """Biological data that was used to build the circuit."""
        path_bioname = self.get_path("bioname")

        assert path_bioname.exists()
        assert path_bioname.is_dir()

        try:
            with open(path_bioname / "MANIFEST.yaml", 'r') as fptr:
                manifest = yaml.load(fptr, Loader=yaml.FullLoader)
        except FileNotFoundError:
            return None

        try:
            return manifest["common"]
        except KeyError:
            pass
        return None

    @lazyfield
    def morphdb(self):
        """..."""
        return MorphDB(path_file=self.get_path("bioname")/"extNeuronDB.dat")

    @lazyfield
    def bluepy_circuit(self):
        """
        An instance of the BluePy circuit object.
        """
        try:
            circuit = BluePyCircuit(str(self.get_path(self.circuit_config)))
        except FileNotFoundError:
            circuit = BluePyCircuit(str(self.get_path(self.circuit_config_base)))
        assert isinstance(circuit, BluePyCircuit)
        return circuit

    @lazyfield
    def bluepy_circuit_structural(self):
        """
        An instance of the BluePy circuit object, with connectome
        set to the structural connectome.
        """
        try:
            circuit = BluePyCircuit(str(self.get_path(self.circuit_config_structural)))
        except FileNotFoundError as error:
            raise ValueError("""Missing structural connectome for the circuit
            at {}""".format(self.path_circuit_data))
        assert isinstance(circuit, BluePyCircuit)
        return circuit

    @lazyfield
    def atlas(self):
        """
        Atlas associated with this circuit.
        """
        biodata = self.biodata
        if biodata:
            try:
                atlas = self.biodata["atlas"]
            except KeyError:
                pass
            else:
                return BlueBrainCircuitAtlas(path=Path(atlas))

        try:
            return BlueBrainCircuitAtlas(self.bluepy_circuit.atlas)
        except AttributeError:
            return BlueBrainCircuitAtlas(path=Path(self.bluepy_circuit.atlas.dirpath))

        raise TypeError("This circuit may not have an atlas!")

    @lazyfield
    def cell_collection(self):
        """
        Cells for the circuit.
        """
        try:
            bluepy_circuit = self.bluepy_circuit
        except BluePyError as error:
            LOGGER.warn(
                LOGGER.get_source_info(),
                "Circuit does not have cells.",
                "BluePy complained:\n\t {}".format(error))
        else:
            return bluepy_circuit.cells
        return None

    @lazyfield
    def cells(self):
        """
        Pandas data-frame with cells in rows.
        """
        cells = self.cell_collection.get()
        return cells.assign(gid=cells.index.values)

    @lazyfield
    def connectome(self):
        """
        connectome for the circuit.
        """
        try:
            bp = self.bluepy_circuit
            return bp.connectome
        except BluePyError as error:
            LOGGER.warn(
                LOGGER.get_source_info(),
                "circuit does not have a connectome.",
                "bluepy complained: \n\t {}".format(error))
        return None

    @lazyfield
    def connectome_structural(self):
        """
        connectome for the structural circuit.
        """
        try:
            bp = self.bluepy_circuit_structural
            return bp.connectome
        except BluePyError as error:
            LOGGER.warn(
                LOGGER.get_source_info(),
                "circuit does not have a structural connectome.",
                "bluepy complained: \n\t {}".format(error))
        return None

    @lazyfield
    def brain_regions(self):
        """
        Brain regions (or sub regions) that the circuit models.
        """
        return self.cells.region.unique()

    @lazyfield
    def layers(self):
        """
        All the layers used in this circuit.
        """
        return self.atlas.layers


    @lazyfield
    def mtypes(self):
        """
        All the mtypes used in this circuit.
        """
        return sorted(list(self.cells.mtype.unique()))

    @lazyfield
    def etypes(self):
        """
        All the etypes in this circuit.
        """
        return self.cells.etype.unique()

    @lazyfield
    def voxel_cell_count(self):
        """
        A pandas.Series providing cell counts in each voxel ---
        indexed by a tuple representing the voxel indices.

        Returns
        -----------
        A numpy.ndarray of the same dimensions as `self.atlas` voxel data.
        """
        return self.atlas.get_bin_counts(self.cells)

    @lazyfield
    def voxel_indexed_cell_gids(self):
        """
        A pandas series mapping a cell's gid to it's voxel index.
        """
        positions_cells = self.cells[XYZ].values
        index = pd.Series(list(self.atlas.positions_to_indices(positions_cells)))\
                  .apply(tuple)
        return  pd.Series(self.cells.index.values, index=index)

    def get_voxel_positions(self, voxel_ids):
        """
        Get positions of voxels from the atlas.

        Arguments
        -----------
        voxel_ids : Iterable of voxel indices as tuples (i, j, k)
        """
        LOGGER.debug(
            """
            LOGGER.get_source_info(),
            get_voxel_positions for voxel_ids of shape {}
            """.format(voxel_ids.shape))

        if voxel_ids.shape[0] == 0:
            return pd.DataFrame([],
                                columns=XYZ,
                                index=pd.MultiIndex.from_arrays([[], [], []],
                                                                names=["i", "j", "k"]))

        return pd.DataFrame(self.atlas.indices_to_positions(voxel_ids),
                            columns=XYZ,
                            index=pd.MultiIndex.from_arrays([voxel_ids[:, 0],
                                                             voxel_ids[:, 1],
                                                             voxel_ids[:, 2]],
                                                            names=["i", "j", "k"]))

    def _atlas_value(self, key, value):
        """
        Value of query parameter as understood by the atlas.
        """
        if value is None:
            return None
        if key == terminology.circuit.region:
            return self.atlas.used_value(region=value)
        if key == terminology.circuit.layer:
            return self.atlas.used_value(layer=value)
        raise RuntimeError(
            "Unknown / NotYetImplemented query parameter {}".format(key))

    @terminology.use(*(terminology.circuit.terms + terminology.cell.terms))
    def _resolve_query_region(self, **query):
        """
        Resolve region in query.

        Arguments
        ------------
        query : a dict providing parameters for a circuit query.
        """
        if not (terminology.circuit.roi in query
                or (terminology.circuit.region in query
                    and not isinstance(query[terminology.circuit.region], str))):
            return query

        for axis in XYZ:
            if axis in query:
                raise TypeError(
                    """
                    Cell query contained coordinates: {}.
                    To query in a region, use its bounding box as the value
                    for key `roi`.
                    """.format(axis))

        if terminology.circuit.roi in query:
            roi = query.pop(terminology.circuit.roi)
            if terminology.circuit.region in query:
                region = query.pop(terminology.circuit.region)
                raise TypeError(
                    """
                    Cannot disambiguate query.
                    Query contained both {}: {}
                    and {}: {}
                    """.format(
                        terminology.circuit.roi, roi,
                        terminology.circuit.region, region))
        else:
            roi = query.pop(terminology.circuit.region)

        corner_0, corner_1 = _get_bounding_box(roi)
        query.update({
            Cell.X: (corner_0[0], corner_1[0]),
            Cell.Y: (corner_0[1], corner_1[1]),
            Cell.Z: (corner_0[2], corner_1[2])})
        return query

    def _get_bluepy_cell_query(self, query):
        """
        Convert `query` that will be accepted by a `BluePyCircuit`.
        """
        def _get_query_layer(layers):
            """
            Arguments
            -------------
            layers : list or a singleton
            """
            if isinstance(layers, list):
                return [_get_query_layer(layer) for layer in layers]

            layer = layers
            if isinstance(layer, (int, np.int, np.int32)):
                return layer
            if layer.startswith('L') and layer[1] in "123456":
                return int(layer[1])
            return layer
            #return None

        cell_query = terminology.bluebrain.cell.filter(**query)

        try:
            layer = cell_query[LAYER]
        except KeyError:
            pass
        else:
            cell_query[LAYER] = _get_query_layer(layer)

        return cell_query

    @terminology.use(*(terminology.circuit.terms + terminology.cell.terms))
    def get_cells(self,
            properties=None,
            target=None,
            **query):
        """
        Get cells in a region, with requested properties.

        Arguments
        --------------
        properties : single cell property or  list of cell properties to fetch.
        query : sequence of keyword arguments providing query parameters.
        with_gid_column : if True add a column for cell gids.
        """
        LOGGER.debug(
            "Model get cells for query",
            "{}".format(query))

        query_region_resolved = self._resolve_query_region(**query)
        cell_query =  self._get_bluepy_cell_query(query_region_resolved)

        if isinstance(target, str):
            cell_query["$target"] = target

        if self.cell_collection is None:
            return None

        cells =  (self.cell_collection
                  .get(group=cell_query, properties=properties))
        cells.index.name = "gid"
                                           
        if target is not None:
            if isinstance(target, str):
                cells = cells.assign(group=target)
            elif isinstance(target, pd.DataFrame):
                cells = cells.reindex(target.index.to_numpy(np.int32))
            else:
                 cells = cells.reindex(np.sort(np.unique([x for x in target])))\
                              .dropna()
        return cells

    def get_mask(self, relative=True, **query):
        """
        Get a mask from the atlas.

        Arguments
        --------------
        relative : Are depth / height value in `query` relative?
        """
        return self.atlas.get_mask(relative=relative,
                                   **terminology.circuit.get_spatial_query(query))

    def get_voxel_count(self, **spatial_query):
        """..."""
        return self.atlas.get_voxel_count(**spatial_query)

    @lazyfield
    def volume_voxel(self):
        return self.atlas.volume_voxel

    def are_connected(self, pre_neuron, post_neuron):
        """
        Is pre neuron connected to post neuron.
        """
        return pre_neuron in self.connectome.afferent_gids(post_neuron)

    def get_afferent_ids(self, neuron):
        """..."""
        return self.connectome.get_afferent_ids(neuron)

    @lazyfield
    def morphologies(self):
        """
        To help with morphologies.
        """
        return self.bluepy_circuit.morph

    @lazyfield
    def segment_index(self):
        """..."""
        return self.morphology.spatial_index

    def get_segment_length_by_neurite_type(self,
            region_of_interest):
        """..."""
        if not self.segment_index:
            return None

        corner_0, corner_1 =\
            region_of_interest.bbox
        dataframe_segment =\
            self.segment_index\
                .q_window_oncenter(
                    corner_0,
                    corner_1
                ).assign(
                    length=lambda segments: np.linalg.norm(
                        segments[[Segment.X1, Segment.Y1, Segment.Z1]].values
                        - segments[[Segment.X2, Segment.Y2, Segment.Z2]].values
                    )
                )

        def _total_length(neurite_type):
            return np.sum(
                dataframe_segment.length[
                    dataframe_segment[Section.NEURITE_TYPE] == neurite_type
                ].values
            )
        return pd.Series({
            neurom.AXON: _total_length(neurom.AXON),
            neurom.BASAL_DENDRITE: _total_length(neurom.BASAL_DENDRITE),
            neurom.APICAL_DENDRITE: _total_length(neurom.APICAL_DENDRITE)
        })

    def get_segment_length_densities_by_mtype(self,
            region_of_interest):
        """..."""
        if not self.segment_index:
            return None

        def _get_length(segments):
            """
            Compute total length of segments.
            """
            return\
                np.linalg.norm(
                    segments[[Segment.X1, Segment.Y1, Segment.Z1]]
                    - segments[[Segment.X2, Segment.Y2, Segment.Z2]])

        corner_0, corner_1 =\
            region_of_interest.bbox
        dataframe_segments = self\
            .segment_index\
            .q_window_oncenter(
                corner_0,
                corner_1)\
            .assign(
                length=_get_length)\
            .set_index("gid")\
            .join(
                self.cells[
                    Cell.MTYPE]
            ).groupby(
                u'mtype'
            ).apply(
                lambda segments: {
                    neurom.AXON: np.sum(
                        segments.length[
                            segments[Section.NEURITE_TYPE] == neurom.AXON
                        ]).values / region_of_interest.volume,
                    neurom.BASAL_DENDRITE: np.sum(
                        segments.length[
                            segments[Section.NEURITE_TYPE] == neurom.BASAL_DENDRITE
                        ]).values / region_of_interest.volume,
                    neurom.APICAL_DENDRITE: np.sum(
                        segments.length[
                            segments[Section.NEURITE_TYPE] == neurom.APICAL_DENDRITE
                        ]).values / region_of_interest.volume
                }
            )
        raise NotImplementedError

    def get_thickness(self, positions, relative=False):
        """
        Layer thickness as measured at a position.

        Arguments
        ------------
        position :: np.ndarray<x, y, z>
        """
        voxel_indices =\
            self.atlas.positions_to_indices(positions)
        thicknesses =\
            pd.DataFrame(self.atlas.layer_thicknesses(voxel_indices))
        thicknesses.columns.name = "layer"
        return\
            thicknesses.assign(total=lambda df: df.sum(axis=1))\
            if not relative else\
               thicknesses.apply(lambda s: s / s.sum(), axis=1)

    @lazyfield
    def orientation_field(self):
        """
        Orientation field defined on the atlas is a rotation matrix to be used
        to rotate a morphology with its axis along Y-axis towards the principal
        axis of cells in a voxel.
        """
        return self.atlas.orientation_field

    @lazyfield
    def mconfig(self):
        """..."""
        if not self.path_mconfig:
            raise ValueError("Path to measurements config not available.")

        with open(self.path_mconfig, 'r') as fptr:
            _mconfig = yaml.load(fptr, Loader=yaml.FullLoader)

        return _mconfig
