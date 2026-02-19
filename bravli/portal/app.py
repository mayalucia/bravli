"""Assemble the bravli portal from individual views.

Usage:
    # In a notebook
    from bravli.portal.app import build_portal
    portal = build_portal()
    portal.servable()

    # As a standalone app
    panel serve bravli/portal/app.py --show
"""

import panel as pn

from bravli.utils import get_logger

LOG = get_logger("portal.app")


def build_portal(annotations=None, edges=None, circuit=None,
                 render_fn=None, detail_fn=None):
    """Build the complete portal application.

    Parameters
    ----------
    annotations : pd.DataFrame, optional
        FlyWire neuron annotations. If None, attempts to load from
        the default data path.
    edges : pd.DataFrame, optional
        Processed edge table (connectivity pipeline output).
    circuit : Circuit, optional
        Pre-built simulation circuit. If None, a demo circuit is used.
    render_fn : callable, optional
        Atlas rendering function (requires navis + fafbseg).
    detail_fn : callable, optional
        Neuropil detail rendering function.

    Returns
    -------
    pn.Tabs
        The complete portal, ready for .servable() or .show().
    """
    pn.extension("plotly", sizing_mode="stretch_width")

    from bravli.portal.views import (
        atlas_view,
        composition_view,
        connectivity_view,
        physiology_view,
        simulate_view,
    )

    # --- Load annotations if not provided ---
    if annotations is None:
        annotations = _try_load_annotations()

    # --- Build tabs ---
    tabs = []

    if annotations is not None:
        tabs.append(("Atlas", atlas_view(
            annotations, render_fn=render_fn, detail_fn=detail_fn,
        )))
        tabs.append(("Composition", composition_view(annotations)))
    else:
        tabs.append(("Atlas", pn.pane.Markdown(
            "# Atlas\n\n*No annotations loaded. Pass annotations=df to build_portal().*",
            styles={"color": "#c9d1d9"},
        )))

    tabs.append(("Connectivity", connectivity_view(edges=edges)))
    tabs.append(("Physiology", physiology_view(edges=edges)))
    tabs.append(("Simulate", simulate_view(circuit=circuit)))

    portal = pn.Tabs(*tabs, sizing_mode="stretch_both")

    LOG.info("Portal built: %d tabs", len(tabs))
    return portal


def _try_load_annotations():
    """Attempt to load FlyWire annotations from standard paths."""
    from pathlib import Path

    search_paths = [
        Path("data/flywire_annotations"),
        Path("../data/flywire_annotations"),
        Path.home() / "Darshan/research/develop/agentic/mayalucia/bravli/code/bravli/data/flywire_annotations",
    ]

    for base in search_paths:
        if base.is_dir():
            tsvs = list(base.glob("*.tsv"))
            if tsvs:
                import pandas as pd
                path = tsvs[0]
                LOG.info("Loading annotations from %s", path)
                return pd.read_csv(path, sep="\t")

    LOG.warning("Could not find FlyWire annotations. Pass annotations=df explicitly.")
    return None


# --- Standalone entry point ---
if __name__ == "__main__" or __name__.startswith("bokeh"):
    portal = build_portal()
    portal.servable()
