"""
Sample randomly from a circuit.
"""
import numpy as np
from neuro_dmt import terminology
from neuro_dmt.utils.geometry import Cuboid

def get_random_roi(adpater, model, size=50.,
                   config=None, **spatial_query):
    """..."""

    position = adapter.get_random_position(model, **spatial_query)
    if position is None:
        return None

    return Cuboid(position - size / 2., position + size /2.)
                   

def sample_cells(adapter, model, size_sample=None,
                 size_roi=None, return_roi=False,
                 config=None, **query):
    """
    Sample cells  specified by a query
    """
    if size_roi:
        spatial_query = terminology.circuit.get_spatial_query(query)
        cell_query = terminology.cell.filter(**query)
        roi = get_random_roi(adapter, model, size=size_roi,
                             config=config, **spatial_query)
        cells = adapter.get_cells(mnodel, region=roi, **cell_query)

        return (roi, cells) if return_roi else cells

    if not size_sample:
        try:
            size_sample = config.cell_sample_size
        except AttributeError:
            try:
                size_sample = config.sample_size
            except AttributeError:
                sample_size = 20

    return adapter.get_cells(model, **query).sample(n=size_sample)



