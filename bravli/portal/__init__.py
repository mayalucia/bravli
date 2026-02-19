"""portal — Interactive digital twin exploration.

A Panel-based application that wires bravli's atlas, connectivity,
physiology, and simulation modules into a navigable, reactive portal.

Launch with:
    panel serve bravli/portal/app.py
or in a notebook:
    from bravli.portal.app import build_portal
    portal = build_portal()
    portal.servable()

Requires: panel >= 1.0
"""

from .views import (
    atlas_view,
    composition_view,
    connectivity_view,
    physiology_view,
    simulate_view,
)
from .app import build_portal
