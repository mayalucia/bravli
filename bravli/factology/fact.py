"""Fact: the atomic unit of scientific measurement.

Heritage: ported from circuit-factology (Blue Brain Project). The core
pattern — Fact namedtuple + @fact decorator + @anatomical/@physiological
registration — is preserved. Renamed to @structural/@connectomic to
match the fly brain domain.
"""

from collections import namedtuple
from functools import wraps


# ---------------------------------------------------------------------------
# The Fact namedtuple
# ---------------------------------------------------------------------------

class Fact(namedtuple("Fact", ["label", "name", "description", "unit", "value"])):
    """A single scientific measurement with full metadata.

    Fields
    ------
    label : str
        Machine-readable identifier (typically the method name).
    name : str
        Human-readable name (e.g., "Total neuron count").
    description : str
        What was measured and how (from the method's docstring).
    unit : str or None
        Unit of measurement (e.g., "neurons", "proportion", None).
    value : any
        The measured value.
    """

    def __str__(self):
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit_str}"

    def to_dict(self):
        """Convert to a plain dict for serialization."""
        return {
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "value": self.value,
        }


# ---------------------------------------------------------------------------
# @fact(name, unit) — wrap a method's return value in a Fact
# ---------------------------------------------------------------------------

def fact(name, unit=None):
    """Decorator factory: wrap a method's return value as a Fact.

    Usage::

        @fact("Neuron count", "neurons")
        def neuron_count(self):
            '''Total number of annotated neurons.'''
            return len(self.annotations)

    When called, returns a Fact with:
        label = "neuron_count" (method name)
        name = "Neuron count" (from decorator)
        description = "Total number of annotated neurons." (docstring)
        unit = "neurons" (from decorator)
        value = <the return value>
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self):
            value = method(self)
            return Fact(
                label=method.__name__,
                name=name,
                description=method.__doc__ or "",
                unit=unit,
                value=value,
            )
        wrapper.__defines_a_fact__ = True
        wrapper._fact_name = name
        wrapper._fact_unit = unit
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# @structural / @connectomic — register and cache facts by category
# ---------------------------------------------------------------------------

def structural(method):
    """Decorator: register a fact as structural and make it a cached property.

    Structural facts describe the anatomy: neuron counts, cell type
    distributions, region volumes, morphological statistics.
    """
    @wraps(method)
    def wrapper(self):
        value = method(self)
        if not hasattr(self, '_structural_facts'):
            self._structural_facts = []
        self._structural_facts.append(value)
        return value
    wrapper.__defines_a_fact__ = True
    wrapper.__fact_type__ = "structural"
    # Make it a cached property
    return _cached_property(wrapper)


def connectomic(method):
    """Decorator: register a fact as connectomic and make it a cached property.

    Connectomic facts describe connectivity: synapse counts, input/output
    regions, degree distributions, connection probabilities.
    """
    @wraps(method)
    def wrapper(self):
        value = method(self)
        if not hasattr(self, '_connectomic_facts'):
            self._connectomic_facts = []
        self._connectomic_facts.append(value)
        return value
    wrapper.__defines_a_fact__ = True
    wrapper.__fact_type__ = "connectomic"
    return _cached_property(wrapper)


def _cached_property(method):
    """Turn a method into a cached property (computed once, then stored)."""
    attr_name = f"_cached_{method.__name__}"

    @wraps(method)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, method(self))
        return getattr(self, attr_name)

    # Preserve the fact metadata
    for attr in ('__defines_a_fact__', '__fact_type__', '_fact_name', '_fact_unit'):
        if hasattr(method, attr):
            setattr(wrapper, attr, getattr(method, attr))
    return wrapper


# ---------------------------------------------------------------------------
# @interface — two-level delegation
# ---------------------------------------------------------------------------

def interface(method):
    """Decorator: declare a method as part of the class interface.

    If the method raises NotImplementedError, the decorator tries to find
    an implementation on self._helper. This lets a Factology work with
    different data backends by swapping the helper object.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NotImplementedError:
            if not hasattr(self, '_helper') or self._helper is None:
                raise
            helper_method = getattr(self._helper, method.__name__, None)
            if helper_method is None:
                raise NotImplementedError(
                    f"{method.__name__} not implemented on {self.__class__.__name__} "
                    f"or its helper {self._helper.__class__.__name__}"
                )
            return helper_method(*args, **kwargs)
    return wrapper
