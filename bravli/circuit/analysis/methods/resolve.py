"""
Methods that resolve parameters
"""
from collections.abc import Iterable
from neuro_dmt import terminology

def _as_list(value, default, element_type=str):
    """..."""
    if value is None:
        try:
            return default()
        except TypeError:
            return default
    elif isinstance(value, element_type):
        return [value]
    elif isinstance(value, Iterable) and not isinstance(value, str):
        return value
    else:
        raise TypeError(type(value))
    raise RuntimeError("Execution should not reach here.")


def regions(adapter, model, **query):
    """
    Resolve region...
    """
    return _as_list(query.get(terminology.bluebrain.cell.region, None),
                    default=lambda: adapter.get_sub_regions(model))


def layers(adapter, model, **query):
    """
    Resolve layer...
    """
    return _as_list(query.get(terminology.bluebrain.cell.layer, None),
                    default=lambda: adapter.get_layers(model))


def hemisphere(adapter, model, **query):
    """
    Resolve hemisphere
    TODO: Update DMT terminology to handle hemispheres
    """
    return query.get("hemisphere", None)
    #return query.get(terminology.bluebrain.cell.hemisphere, None)
