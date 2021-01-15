"""
Analyze bouton density in a circuit.
"""

import numpy as np
import pandas as pd
from dmt.tk.author import Author
from dmt.model.interface import interfacemethod
from dmt.analysis.document.builder import ArticleBuilder
from bravli.circuit.analysis.configuration import CircuitAnalysisConfiguration

ART = ArticleBuilder(title="Bouton density in a circuit region.",
                     author=Author.zero,
                     __file__=__file__)

CONFIG = CircuitAnalysisConfiguration()

@interfacemethod
def get_synapses_per_bouton(adapter, model, **cell_query):
    """
    Number of synapse per bouton for a cell query.
    """
    raise NotImplementedError


@interfacemethod
def get_axon_length(adapter, model, cell, region=None):
    """
    Total length of a cell's axon.
    If `region` is queried, compute length over only those segments of the axon
    that lie inside the queried region.
    """
    raise NotImplementedError


@interfacemethod
def get_efferent_synapse_count(adapter, model, cell):
    """
    Number of all efferent synapses of a cell.
    """
    raise NotImplementedError


@interfacemethod
def get_efferent_synapse_segments(adapter, model, cell, region=None):
    """
    Each section of an axonic or a dendritic tree is modeled as contiguous
    sequence of equal lengthed segments
    A segment on a dendrite, or an axon may be thought of essentially as
    two 3D-positions marking its begin and end points.
    We can use a multi-indexed `pandas.Series` to represent a single segment.
    The `MultiIndex`'s first level should have values `['begin', 'end']`.
    It's second level should be `['x', 'y', 'z`]`.
    Get segment points of the dendrites and axons of a cell.

    Returns
    ---------
    A `pandas.DataFrame` each row of which represents the segment that contains
    a synapse on `cell`'s axon.
    If a `region` is queried, filter segments that are contained in the region.
    If a `neurite_type` is not queried, index the dataframe by segment `neurite_type`.
    """
    raise NotImplementedError


@ART.methods
def get_bouton_density(adapter, model, cell, region=None):
    """
    Get bouton density for a cell.
    If a `region` is specified, only those segments in the cell's axon are
    considered that fall in the queried region.
    """
    if not region:
        synapses = adapter.count_efferent_synapses(model, cell)
        axon_length = adapter.get_axon_length(model, cell)
    else:
        segments = adapter.get_efferent_synapse_segments(adapter, model, cell, region)
        synapses = segments.shape[0]
        axon_length = np.linalg.norm(segments.end - segments.begin).sum()

    synapses_per_bouton = adapter.get_synapses_per_bouton( model)
    boutons = synapses / synapses_per_bouton
    return boutons / axon_length


@ART.methods
def get_axon_apposition_density(adapter, model, cell, region=None):
    """
    Get number of appositions per unit length of the axon.
    """
    if not region:
        appositions = adapter.count_axon_appositions(model, cell)
        axon_length = adapter.get_axon_length(model, cell)
    else:
        segments = adapter.get_axon_apposition_segments(adapter, model, cell,region)
        appositions = segments.shape[0]
        axon_length = np.linalg.norm(segments.end - segments.begin).sum()

    appositions_per_bouton = adapter.get_synapses_per_bouton(model)
    boutons = appositions / appositions_per_bouton
    return boutons / axon_length

@ART.sections
def bio_data(adapter, model):
    """
    Biological data for bouton densities.
    TODO: What and from where?
    """
    pass


@ART.sections
def bouton_density(adapter, model):
    """
    TODO: about bouton-density
    """
    pass


@ART.sections.bouton_density.measurements
def get_bouton_density_by_mtype(adapter, model, config=CONFIG,
                                **spatial_query):
    """
    Arguments
    --------------
    """
    from bravli.circuit.analysis.methods import sample_cells
    try:
        mtypes = config.mtypes
    except AttributeError:
        mtypes = adapter.get_mtypes(model)

    try:
        sample_size = config.cell_sample_size
    except AttributeError:
        try:
            sample_size = config.sample_size
        except AttributeError:
            sample_size = 20

    def get(mtype):
        cells = sample_cells(adapter, model, size_sample=sample_size,
                             mtype=mtype, **spatial_query)
        bouton_densities = [get_bouton_density(adapter, model, cell)
                            for cell in cells.itertuples()]
        return pd.DataFrame({"mtype": mtype, "bouton_density": bouton_densities})
                             

    return pd.concat([get(mtype) for mtype in mtypes])

@ART.sections.bouton_density.illustration
def plot_bouton_density(adapter, model, **kwargs):
    """
    Plot a bars for mtype bouton density ...
    """
    raise NotImplementedError



