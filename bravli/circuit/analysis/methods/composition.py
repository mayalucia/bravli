"""
Methods to analyze composition of a circuit.
"""
from collections.abc import Mapping
from collections import namedtuple
import numpy as np
import pandas as pd
from dmt.analysis.document.components import Section
from .. import statistical
from . import resolve

def get_mtype_counts(adapter, circuit_model, target, **query):
    """..."""
    counts =  adapter.get_cells(circuit_model, target=target, **query)\
                     .mtype.value_counts()\
                     .rename("count")
    counts.index.name = "mtype"
    return counts

def get_morphology_distribution(adapter, circuit_model, mtype,
                                target=None, **query):
    """
    Get the distribution of morphologies among cells specified by a query.

    Arguments
    ---------------
    mtype : For which the morphology distribution is to be evaluated
    target : named target of cells
    query  : Mapping cell-property ==> value
    """
    def get_frequency(**group):
        f = adapter.get_cells(circuit_model, mtype=mtype, **group)\
                   .morphology\
                   .value_counts()\
                   .rename("frequency")
        return f[f>0.]

    frequency_base = get_frequency()
    pdf_base = statistical.distribution(frequency_base)

    if not target and not query:
        return pdf_base

    return statistical.distribution(get_frequency(target=target, **query)
                                    .reindex(frequency_base.index)
                                    .fillna(0.))

def get_divergence_morphology_distribution(adapter, circuit_model, mtype,
                                           target, reference,
                                           **query):
    """
    How much does distribution of morphologies among target cells diverge
    from that among reference cells?
    """
    def get_distribution(group):
        return get_morphology_distribution(adapter, circuit_model, mtype,
                                           target=group, **query)

    pdf_reference = get_distribution(reference)
    pdf_target = get_distribution(target)
    return np.sum(pdf_target * np.log(pdf_target / pdf_reference))

def get_positions_and_orientations(adapter, model, target=None,
                                   in_left_hemisphere=None,
                                   **query):
    """
    Get positions and orientations from for a region specified by `query`.
    These positions and orientations will be retrieved from the atlas behind
    the circuit model.
    """
    if in_left_hemisphere is None:
        in_left_hemisphere =  lambda pos: pos.z < 5700. #this applies only to the mouse Isocortex
    if target:
        assert not query
        cells = adapter.get_cells(model, target=target)
        xyz = cells[["x", "y", "z"]]
        ijk = adapter.get_voxel_indices(model, xyz.values)
        index = ijk.assign(layer=cells.layer.values, region=cells.region.values)
        position = pd.DataFrame(xyz.values, columns=list("xyz"),
                                index=pd.MultiIndex.from_frame(index))
        orientation = pd.DataFrame(np.vstack(cells["orientation"]
                                             .apply(lambda r: r[:, 1]).values),
                                   columns=["x", "y", "z"],
                                   index=position.index)
        return pd.concat([position, orientation], axis=1,
                         keys=["position", "orientation"])

    regions = resolve.regions(adapter, model, **query)
    layers = resolve.layers(adapter, model, **query)
    hemisphere = resolve.hemisphere(adapter, model, **query)

    def get(region, layer):
        positions = (adapter
                     .visible_voxels(model, {"region": region, "layer": layer})
                     .positions)
        orientations = (adapter
                        .get_orientations(model, positions.to_numpy(np.float)))
        orientations.index = positions.index

        return pd.concat([positions, orientations],
                         axis=1,
                         keys=["position", "orientation"])
    region_layers = [(r, l) for r in regions for l in layers]

    posoris = pd.concat([get(r, l) for r, l in region_layers],
                        axis=0,
                        keys=region_layers,
                        names=["region", "layer"])
    if not hemisphere:
        #Hack borrowed from brainbuilder
        return posoris

    if hemisphere == "left":
        return posoris[in_left_hemisphere(posoris.position)]
    if hemisphere == "right":
        return posoris[np.logical_not(in_left_hemisphere(posoris.position))]
    raise ValueError("Unknown hemisphere {}".format(hemisphere))


Axis = namedtuple("Axis", ["origin", "orientation"])
