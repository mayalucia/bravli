"""Visualization wrappers around navis for 3D fly brain rendering.

Heritage: this module replaces the BBP visualization stack (VoxelData +
Blender/ParaView + custom OpenGL) with navis, which handles meshes,
skeletons, and interactive 3D natively.  The wrapper functions speak
bravli's domain language — neuropils, cell types, neurotransmitters —
so that downstream code (Lesson 05+) never touches navis directly.

Design principle: every function *returns* the figure/viewer handle.
Nothing is displayed implicitly — the caller decides when and how to show.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from bravli.utils import get_logger

LOG = get_logger("viz")

# Try importing navis — fail gracefully so the rest of bravli works
# even without visualization dependencies installed.
try:
    import navis
    import navis.plotting  # ensure plotting submodule loaded
    HAS_NAVIS = True
except ImportError:
    HAS_NAVIS = False
    LOG.warning("navis not installed — visualization functions will raise ImportError")

# Try importing plotly for figure composition and static export
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_navis():
    """Guard: raise ImportError with a helpful message if navis is missing."""
    if not HAS_NAVIS:
        raise ImportError(
            "navis is required for visualization. "
            "Install it with: pip install 'bravli[viz]'"
        )


# ---------------------------------------------------------------------------
# Color palettes — domain-meaningful defaults
# ---------------------------------------------------------------------------

NT_COLORS: Dict[str, str] = {
    "acetylcholine": "#2196F3",   # blue
    "gaba":          "#F44336",   # red
    "glutamate":     "#4CAF50",   # green
    "dopamine":      "#FF9800",   # orange
    "serotonin":     "#9C27B0",   # purple
    "octopamine":    "#00BCD4",   # cyan
    "tyramine":      "#795548",   # brown
    "unknown":       "#9E9E9E",   # grey
}
"""Colors for neurotransmitter types, chosen for perceptual distinctness."""

SUPERCLASS_COLORS: Dict[str, str] = {
    "central":             "#3F51B5",   # indigo
    "optic":               "#8BC34A",   # light green
    "sensory":             "#FF5722",   # deep orange
    "ascending":           "#009688",   # teal
    "descending":          "#E91E63",   # pink
    "motor":               "#FFC107",   # amber
    "endocrine":           "#607D8B",   # blue grey
    "visual_projection":   "#CDDC39",   # lime
    "visual_centrifugal":  "#00E676",   # green accent
}
"""Colors for FlyWire super_class categories."""

DEFAULT_NEUROPIL_COLOR = (0.75, 0.75, 0.85, 0.25)
"""Default RGBA for neuropil volumes: translucent blue-grey."""


# ---------------------------------------------------------------------------
# show_neuropil — render neuropil meshes
# ---------------------------------------------------------------------------

def show_neuropil(
    meshes: Union[Dict[str, Any], List[Any]],
    names: Optional[List[str]] = None,
    colors: Optional[Union[Dict[str, str], str, list]] = None,
    alpha: float = 0.25,
    title: Optional[str] = None,
    backend: str = "plotly",
    **kwargs,
):
    """Render one or more neuropil meshes as translucent 3D volumes.

    Parameters
    ----------
    meshes : dict or list
        If dict: {neuropil_name: mesh_data} where mesh_data is either
        a navis.Volume, a trimesh.Trimesh, or a (vertices, faces) tuple.
        If list: list of mesh objects (paired with `names`).
    names : list of str, optional
        Names for list-mode meshes. Ignored if meshes is a dict.
    colors : dict or str or list, optional
        Colors per neuropil name (dict), a single color (str), or a
        list of colors matching the meshes.  Defaults to blue-grey.
    alpha : float
        Transparency (0 = invisible, 1 = opaque).
    title : str, optional
        Title for the plot.
    backend : str
        Navis plotting backend: "plotly" (default), "vispy", "octarine".
    **kwargs
        Passed through to navis.plot3d.

    Returns
    -------
    fig : plotly.graph_objects.Figure or navis viewer
        The rendered figure.
    """
    _require_navis()

    # Normalize to list of (name, mesh) pairs
    if isinstance(meshes, dict):
        pairs = list(meshes.items())
    else:
        if names is None:
            names = [f"neuropil_{i}" for i in range(len(meshes))]
        pairs = list(zip(names, meshes))

    # Convert each mesh to navis.Volume
    volumes = []
    for name, mesh_data in pairs:
        vol = _to_volume(mesh_data, name=name, alpha=alpha)

        # Assign color
        if isinstance(colors, dict):
            col = colors.get(name, DEFAULT_NEUROPIL_COLOR)
        elif isinstance(colors, list):
            idx = [n for n, _ in pairs].index(name)
            col = colors[idx] if idx < len(colors) else DEFAULT_NEUROPIL_COLOR
        elif colors is not None:
            col = colors
        else:
            col = DEFAULT_NEUROPIL_COLOR

        vol.color = col
        volumes.append(vol)

    LOG.info("Rendering %d neuropil volumes", len(volumes))
    fig = navis.plot3d(volumes, backend=backend, **kwargs)

    if title and HAS_PLOTLY and isinstance(fig, go.Figure):
        fig.update_layout(title=title)

    return fig


def _to_volume(mesh_data, name="volume", alpha=0.25):
    """Convert various mesh representations to navis.Volume.

    Accepts:
    - navis.Volume (returned as-is)
    - trimesh.Trimesh (wrapped)
    - (vertices, faces) tuple
    - dict with 'vertices' and 'faces' keys
    """
    if isinstance(mesh_data, navis.Volume):
        return mesh_data

    if hasattr(mesh_data, 'vertices') and hasattr(mesh_data, 'faces'):
        return navis.Volume(
            vertices=np.asarray(mesh_data.vertices),
            faces=np.asarray(mesh_data.faces),
            name=name,
            color=(*DEFAULT_NEUROPIL_COLOR[:3], alpha),
        )

    if isinstance(mesh_data, tuple) and len(mesh_data) == 2:
        vertices, faces = mesh_data
        return navis.Volume(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            name=name,
            color=(*DEFAULT_NEUROPIL_COLOR[:3], alpha),
        )

    if isinstance(mesh_data, dict) and 'vertices' in mesh_data:
        return navis.Volume(
            vertices=np.asarray(mesh_data['vertices']),
            faces=np.asarray(mesh_data['faces']),
            name=name,
            color=(*DEFAULT_NEUROPIL_COLOR[:3], alpha),
        )

    raise TypeError(
        f"Cannot convert {type(mesh_data)} to navis.Volume. "
        "Expected Volume, trimesh, (vertices, faces) tuple, or dict."
    )


# ---------------------------------------------------------------------------
# show_neurons — render neuron meshes colored by property
# ---------------------------------------------------------------------------

def show_neurons(
    neurons: Union[Any, List[Any]],
    color_by: Optional[str] = None,
    annotations: Optional[pd.DataFrame] = None,
    palette: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    backend: str = "plotly",
    **kwargs,
):
    """Render neuron meshes, optionally colored by a property.

    Parameters
    ----------
    neurons : navis.NeuronList or list of navis neurons
        The neurons to render.
    color_by : str, optional
        Column in `annotations` to use for coloring. Common choices:
        "top_nt" (neurotransmitter), "super_class", "cell_type".
        If None, navis default coloring is used.
    annotations : pd.DataFrame, optional
        Annotation table indexed or containing 'root_id'. Required if
        `color_by` is specified.
    palette : dict, optional
        {value: color} mapping. Defaults to NT_COLORS or SUPERCLASS_COLORS
        depending on color_by.
    title : str, optional
        Title for the plot.
    backend : str
        Navis plotting backend.
    **kwargs
        Passed through to navis.plot3d.

    Returns
    -------
    fig : plotly.graph_objects.Figure or navis viewer
    """
    _require_navis()

    # Ensure we have a NeuronList
    if isinstance(neurons, (list, tuple)):
        neuron_list = navis.NeuronList(neurons)
    elif isinstance(neurons, navis.NeuronList):
        neuron_list = neurons
    else:
        neuron_list = navis.NeuronList([neurons])

    # Build color mapping if requested
    color_kwarg = {}
    if color_by is not None and annotations is not None:
        if palette is None:
            palette = _default_palette(color_by)
        color_map = _build_color_map(neuron_list, annotations, color_by, palette)
        if color_map:
            color_kwarg["color"] = color_map

    LOG.info("Rendering %d neurons", len(neuron_list))
    fig = navis.plot3d(neuron_list, backend=backend, **color_kwarg, **kwargs)

    if title and HAS_PLOTLY and isinstance(fig, go.Figure):
        fig.update_layout(title=title)

    return fig


def _default_palette(color_by):
    """Return the default palette for a given color_by column."""
    if color_by == "top_nt":
        return NT_COLORS
    if color_by == "super_class":
        return SUPERCLASS_COLORS
    return {}


def _build_color_map(neuron_list, annotations, color_by, palette):
    """Build a {neuron_id: color} dict from annotations and palette."""
    color_map = {}
    for neuron in neuron_list:
        nid = neuron.id
        # Look up the annotation row for this neuron
        if "root_id" in annotations.columns:
            row = annotations[annotations["root_id"] == nid]
        elif annotations.index.name == "root_id":
            row = annotations.loc[[nid]] if nid in annotations.index else pd.DataFrame()
        else:
            continue

        if len(row) == 0:
            continue

        value = row.iloc[0].get(color_by, None)
        if value is not None and value in palette:
            color_map[nid] = palette[value]

    return color_map


# ---------------------------------------------------------------------------
# show_neuropil_connectivity — overlay connectivity edges
# ---------------------------------------------------------------------------

def show_neuropil_connectivity(
    centroids: Dict[str, np.ndarray],
    weights: pd.DataFrame,
    meshes: Optional[Dict[str, Any]] = None,
    top_n: int = 20,
    min_weight: float = 0.0,
    edge_color: str = "#FF5722",
    title: Optional[str] = None,
    backend: str = "plotly",
    **kwargs,
):
    """Visualize neuropil-to-neuropil connectivity as 3D edges.

    Draws lines between neuropil centroids, with line width proportional
    to connection weight.  Optionally renders neuropil meshes underneath.

    Parameters
    ----------
    centroids : dict
        {neuropil_name: np.array([x, y, z])} centroid coordinates.
    weights : pd.DataFrame
        Connectivity matrix (neuropils × neuropils). Values are synapse
        counts or connection strengths.
    meshes : dict, optional
        {neuropil_name: mesh_data} for background neuropil rendering.
    top_n : int
        Only draw the top N strongest connections.
    min_weight : float
        Minimum weight threshold for drawing an edge.
    edge_color : str
        Color for connectivity edges.
    title : str, optional
        Title for the plot.
    backend : str
        Plotting backend (only "plotly" fully supports edge overlays).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for connectivity visualization. "
            "Install it with: pip install plotly"
        )
    _require_navis()

    fig = go.Figure()

    # Render background meshes if provided
    if meshes is not None:
        for name, mesh_data in meshes.items():
            vol = _to_volume(mesh_data, name=name, alpha=0.1)
            verts = np.asarray(vol.vertices)
            faces_arr = np.asarray(vol.faces)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces_arr[:, 0], j=faces_arr[:, 1], k=faces_arr[:, 2],
                opacity=0.1,
                name=name,
                hoverinfo="name",
            ))

    # Extract edges from the weight matrix
    edges = []
    for src in weights.index:
        for tgt in weights.columns:
            w = weights.loc[src, tgt]
            if w > min_weight and src in centroids and tgt in centroids:
                edges.append((src, tgt, w))

    # Sort by weight and keep top N
    edges.sort(key=lambda e: e[2], reverse=True)
    edges = edges[:top_n]

    if not edges:
        LOG.warning("No edges to draw (all below threshold or missing centroids)")
    else:
        # Normalize weights for line width scaling
        max_w = max(e[2] for e in edges)
        min_w = min(e[2] for e in edges)
        w_range = max_w - min_w if max_w != min_w else 1.0

        for src, tgt, w in edges:
            p0 = centroids[src]
            p1 = centroids[tgt]
            width = 1 + 9 * (w - min_w) / w_range  # scale 1–10
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color=edge_color, width=width),
                name=f"{src}→{tgt} ({w:.0f})",
                hoverinfo="name",
            ))

    # Add centroid markers
    labeled = set()
    for src, tgt, _ in edges:
        labeled.add(src)
        labeled.add(tgt)

    if labeled:
        pts = np.array([centroids[n] for n in labeled])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            marker=dict(size=4, color="#212121"),
            text=list(labeled),
            textposition="top center",
            textfont=dict(size=8),
            name="neuropils",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=title or "Neuropil connectivity",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        showlegend=True,
    )

    LOG.info("Rendered %d connectivity edges", len(edges))
    return fig
