"""Render the fly brain atlas: neuropil meshes + neuron morphologies.

Produces interactive plotly HTML figures combining neuropil boundary
meshes from fafbseg with neuron skeletons from the Zenodo parquet.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bravli.atlas.neuropils import (
    load_neuropils,
    NEUROPIL_GROUPS,
)
from bravli.utils import get_logger

LOG = get_logger("atlas.render")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_plotly():
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

REGION_COLORS = {
    "mushroom_body": "#F44336",
    "antennal_lobe": "#4CAF50",
    "lateral_horn": "#FF9800",
    "central_complex": "#9C27B0",
    "optic_medulla": "#90CAF9",
    "optic_lobula": "#64B5F6",
    "optic_lobula_plate": "#42A5F5",
    "optic_lamina": "#BBDEFB",
    "optic_accessory_medulla": "#E3F2FD",
    "superior_protocerebrum": "#FFAB91",
    "ventrolateral_protocerebrum": "#FFCC80",
    "lateral_accessory_lobe": "#CE93D8",
    "bulb": "#F48FB1",
    "anterior_optic_tubercle": "#80CBC4",
    "gnathal_ganglion": "#A1887F",
    "other_midline": "#78909C",
}
"""Colors for neuropil groups."""

DARK_THEME = dict(
    bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font_color="white",
)


def _neuropil_color(name):
    """Get color for a neuropil based on its group membership."""
    for group, members in NEUROPIL_GROUPS.items():
        if name in members:
            return REGION_COLORS.get(group, "#546E7A")
    return "#546E7A"


# ---------------------------------------------------------------------------
# Skeleton → plotly line trace
# ---------------------------------------------------------------------------

def _skeleton_lines(neuron):
    """Convert a TreeNeuron to plotly-compatible line arrays.

    Returns (xs, ys, zs) where segments are separated by None values,
    suitable for Scatter3d with mode='lines'.
    """
    nodes = neuron.nodes
    coords = dict(zip(
        nodes["node_id"],
        zip(nodes["x"], nodes["y"], nodes["z"]),
    ))

    xs, ys, zs = [], [], []
    for nid, pid in zip(nodes["node_id"], nodes["parent_id"]):
        if pid >= 0 and pid in coords:
            x0, y0, z0 = coords[nid]
            x1, y1, z1 = coords[pid]
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            zs.extend([z0, z1, None])
    return xs, ys, zs


# ---------------------------------------------------------------------------
# render_atlas — whole brain
# ---------------------------------------------------------------------------

def render_atlas(
    neurons=None,
    neuron_color: str = "#00E676",
    highlight_groups: Optional[List[str]] = None,
    title: str = "Drosophila whole-brain atlas",
    width: int = 1300,
    height: int = 900,
) -> "go.Figure":
    """Render the whole-brain atlas with all 78 neuropil meshes.

    Parameters
    ----------
    neurons : navis.NeuronList, optional
        Neuron skeletons to overlay.
    neuron_color : str
        Default color for neuron skeletons.
    highlight_groups : list of str, optional
        Neuropil groups to render at higher opacity (e.g.,
        ['mushroom_body', 'central_complex']).
    title : str
        Figure title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    highlight = set()
    if highlight_groups:
        for g in highlight_groups:
            highlight.update(NEUROPIL_GROUPS.get(g, []))

    fig = go.Figure()
    meshes = load_neuropils()

    for name, vol in sorted(meshes.items()):
        v = np.asarray(vol.vertices)
        f = np.asarray(vol.faces)
        is_highlighted = name in highlight
        color = _neuropil_color(name)
        opacity = 0.2 if is_highlighted else 0.06

        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=opacity,
            name=name, hoverinfo="name",
            showlegend=is_highlighted,
        ))

    if neurons is not None:
        _add_neurons(fig, neurons, color=neuron_color)

    _apply_layout(fig, title=title, width=width, height=height)
    LOG.info("Rendered whole-brain atlas with %d neuropils", len(meshes))
    return fig


# ---------------------------------------------------------------------------
# render_neuropil_detail — focused view
# ---------------------------------------------------------------------------

def render_neuropil_detail(
    group: str,
    neurons=None,
    neuron_meta: Optional[Dict[int, Dict]] = None,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 800,
) -> "go.Figure":
    """Render a focused view of a neuropil group with neuron overlays.

    Parameters
    ----------
    group : str
        Neuropil group name (e.g., 'mushroom_body', 'antennal_lobe').
    neurons : navis.NeuronList, optional
        Neuron skeletons to overlay.
    neuron_meta : dict, optional
        {root_id: {'class': str, 'type': str}} for coloring neurons.
    title : str, optional
        Figure title. Defaults to the group name.
    width, height : int
        Figure dimensions.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    meshes = load_neuropils(group=group)
    fig = go.Figure()

    # Distinct color per compartment
    compartment_colors = [
        "#F44336", "#EF9A9A", "#FF9800", "#FFCC80",
        "#4CAF50", "#A5D6A7", "#2196F3", "#90CAF9",
        "#9C27B0", "#CE93D8", "#FF5722", "#FFAB91",
    ]
    for i, (name, vol) in enumerate(sorted(meshes.items())):
        v = np.asarray(vol.vertices)
        f = np.asarray(vol.faces)
        color = compartment_colors[i % len(compartment_colors)]
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=0.12,
            name=name, hoverinfo="name",
        ))

    if neurons is not None:
        _add_neurons(fig, neurons, meta=neuron_meta)

    _apply_layout(
        fig,
        title=title or f"{group.replace('_', ' ').title()} — neuropil detail",
        width=width, height=height,
    )
    LOG.info("Rendered detail view for '%s' with %d compartments",
             group, len(meshes))
    return fig


# ---------------------------------------------------------------------------
# render_morphologies — skeletons only
# ---------------------------------------------------------------------------

def render_morphologies(
    neurons,
    meta: Optional[Dict[int, Dict]] = None,
    title: str = "Neuron morphologies",
    width: int = 1200,
    height: int = 800,
) -> "go.Figure":
    """Render neuron skeletons colored by metadata.

    Parameters
    ----------
    neurons : navis.NeuronList
        Skeletons to render.
    meta : dict, optional
        {root_id: {'class': str, 'type': str}} for legend grouping
        and coloring.
    title : str
        Figure title.
    width, height : int
        Figure dimensions.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    fig = go.Figure()
    _add_neurons(fig, neurons, meta=meta)
    _apply_layout(fig, title=title, width=width, height=height)
    LOG.info("Rendered %d neuron morphologies", len(neurons))
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Kenyon_Cell": "#00E676",
    "MBON": "#E91E63",
    "DAN": "#9C27B0",
    "MBIN": "#FF9800",
    "ALPN": "#4CAF50",
    "ALLN": "#8BC34A",
    "CX": "#2196F3",
}


def _add_neurons(fig, neurons, color=None, meta=None):
    """Add neuron skeleton traces to a figure."""
    added_legends = set()

    for neuron in neurons:
        nid = neuron.id
        info = (meta or {}).get(nid, {})
        cls = info.get("class", "neuron")
        ct = info.get("type", str(nid))

        neuron_color = color or CLASS_COLORS.get(cls, "#00E676")
        legend = ct if meta else cls

        show = legend not in added_legends
        added_legends.add(legend)

        xs, ys, zs = _skeleton_lines(neuron)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=neuron_color, width=1.5),
            name=legend,
            showlegend=show,
            legendgroup=legend,
            hoverinfo="name",
        ))


def _apply_layout(fig, title, width, height):
    """Apply standard dark-theme layout to a figure."""
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            zaxis_title="Z (nm)",
            aspectmode="data",
            bgcolor=DARK_THEME["bgcolor"],
        ),
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        font_color=DARK_THEME["font_color"],
        width=width,
        height=height,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(13,17,23,0.8)"),
    )
