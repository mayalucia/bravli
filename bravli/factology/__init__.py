"""factology — Structured scientific measurements.

Every measurement is a Fact: a named, described, typed value. A Factology
is a collection of Facts about a subject (a neuropil, a cell type, etc.).
"""

from .fact import Fact, fact, structural, connectomic, interface
from .factology import Factology, NeuropilFacts
