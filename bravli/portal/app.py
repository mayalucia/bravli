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
                 render_fn=None, detail_fn=None, auto_pipeline=True):
    """Build the complete portal application.

    Parameters
    ----------
    annotations : pd.DataFrame, optional
        FlyWire neuron annotations. If None, attempts to load from
        the default data path.
    edges : pd.DataFrame, optional
        Processed edge table (connectivity pipeline output).
        If None and auto_pipeline=True, attempts to run the full
        edge pipeline from the Zenodo feather file.
    circuit : Circuit, optional
        Pre-built simulation circuit. If None and edges are available,
        builds a circuit from the pipeline. Otherwise uses a demo.
    render_fn : callable, optional
        Atlas rendering function (requires navis + fafbseg).
    detail_fn : callable, optional
        Neuropil detail rendering function.
    auto_pipeline : bool
        If True, automatically run the connectivity -> physiology ->
        circuit pipeline when data files are found. ~23s on first load.

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
        mb_view,
        simulate_view,
    )

    # --- Load annotations if not provided ---
    if annotations is None:
        annotations = _try_load_annotations()

    # --- Run the full pipeline if data available ---
    if auto_pipeline and edges is None:
        edges, circuit = _try_run_pipeline(annotations, circuit)
        # If pipeline produced a circuit with no synapses, fall back to None
        # (the simulate view will create a demo circuit)
        if circuit is not None and circuit.n_synapses == 0:
            LOG.warning("Pipeline circuit has 0 synapses — falling back to demo")
            circuit = None

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
    tabs.append(("Mushroom Body", mb_view(annotations=annotations, edges=edges)))
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


def _try_find_feather():
    """Search for the Zenodo edge list feather file."""
    from pathlib import Path

    search_paths = [
        Path("data/zenodo/proofread_connections_783.feather"),
        Path("../data/zenodo/proofread_connections_783.feather"),
        Path.home() / "Darshan/research/develop/agentic/mayalucia/bravli/code/bravli/data/zenodo/proofread_connections_783.feather",
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


def _try_run_pipeline(annotations, circuit):
    """Run the full connectivity -> physiology -> circuit pipeline.

    Returns (edges, circuit) — either populated or (None, circuit).
    """
    feather_path = _try_find_feather()
    if feather_path is None:
        LOG.info("No feather file found; Connectivity/Physiology tabs will show instructions.")
        return None, circuit

    import time
    t0 = time.time()
    LOG.info("Running data pipeline from %s ...", feather_path)

    from bravli.connectivity import (
        load_edges, threshold_edges, assign_dominant_nt, aggregate_by_pair,
    )
    from bravli.physiology import assign_synapse_models, compute_synaptic_weights

    # Edge pipeline
    edges = load_edges(feather_path)
    edges = threshold_edges(edges, min_syn=5)
    edges = assign_dominant_nt(edges)
    edges = aggregate_by_pair(edges)
    edges = assign_dominant_nt(edges)

    # Physiology
    edges = assign_synapse_models(edges, mode="shiu")
    edges = compute_synaptic_weights(edges, mode="shiu")

    t1 = time.time()
    LOG.info("Edge pipeline complete: %d edges, %.1fs", len(edges), t1 - t0)

    # Build circuit if annotations available and no circuit provided
    if circuit is None and annotations is not None:
        from bravli.models import assign_cell_models
        from bravli.simulation import build_circuit

        neurons = assign_cell_models(annotations, mode="class_aware")
        circuit = build_circuit(neurons, edges, dt=0.1)

        t2 = time.time()
        LOG.info("Circuit built: %s, %.1fs", circuit.summary().split('\n')[0], t2 - t1)

    elif circuit is None:
        # No annotations — build uniform circuit from edges
        from bravli.simulation import build_circuit_from_edges
        circuit = build_circuit_from_edges(edges, dt=0.1)
        t2 = time.time()
        LOG.info("Uniform circuit built: %s, %.1fs",
                 circuit.summary().split('\n')[0], t2 - t1)

    total = time.time() - t0
    LOG.info("Full pipeline: %.1fs", total)
    return edges, circuit


# --- Standalone entry point ---
if __name__ == "__main__" or __name__.startswith("bokeh"):
    portal = build_portal()
    portal.servable()
