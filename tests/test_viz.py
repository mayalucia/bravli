"""Smoke tests for bravli.viz.

These tests verify that visualization functions execute without error
and return expected types.  They do NOT verify visual correctness.
All tests use synthetic data (tiny tetrahedra) — no FlyWire data needed.
"""

import pytest
import numpy as np
import pandas as pd

# Skip entire module if navis is not available
navis = pytest.importorskip("navis")
plotly = pytest.importorskip("plotly")

from bravli.viz import (
    show_neuropil,
    show_neurons,
    show_neuropil_connectivity,
    NT_COLORS,
    SUPERCLASS_COLORS,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic meshes
# ---------------------------------------------------------------------------

def _tetrahedron(offset=(0, 0, 0)):
    """Return (vertices, faces) for a unit tetrahedron at offset."""
    ox, oy, oz = offset
    vertices = np.array([
        [ox + 0.0, oy + 0.0, oz + 0.0],
        [ox + 1.0, oy + 0.0, oz + 0.0],
        [ox + 0.5, oy + 1.0, oz + 0.0],
        [ox + 0.5, oy + 0.5, oz + 1.0],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=int)
    return vertices, faces


@pytest.fixture
def single_mesh():
    """A single tetrahedron mesh as (vertices, faces)."""
    return _tetrahedron()


@pytest.fixture
def mesh_dict():
    """Two named meshes as a dict."""
    return {
        "neuropil_A": _tetrahedron((0, 0, 0)),
        "neuropil_B": _tetrahedron((3, 0, 0)),
    }


@pytest.fixture
def sample_neuron():
    """A navis MeshNeuron from a tetrahedron."""
    verts, faces = _tetrahedron()
    return navis.MeshNeuron((verts, faces), id=42, name="test_neuron")


@pytest.fixture
def sample_annotations():
    """Minimal annotation DataFrame matching sample neurons."""
    return pd.DataFrame({
        "root_id": [42, 43],
        "top_nt": ["acetylcholine", "gaba"],
        "super_class": ["central", "optic"],
        "cell_type": ["KC", "Mi1"],
    })


# ---------------------------------------------------------------------------
# Tests: show_neuropil
# ---------------------------------------------------------------------------

class TestShowNeuropil:

    def test_single_mesh_tuple(self, single_mesh):
        """Render a single (vertices, faces) tuple."""
        fig = show_neuropil(
            meshes=[single_mesh],
            names=["test"],
            backend="plotly",
        )
        assert fig is not None

    def test_mesh_dict(self, mesh_dict):
        """Render meshes from a name→mesh dict."""
        fig = show_neuropil(
            meshes=mesh_dict,
            backend="plotly",
        )
        assert fig is not None

    def test_with_colors(self, mesh_dict):
        """Render with custom colors per neuropil."""
        fig = show_neuropil(
            meshes=mesh_dict,
            colors={"neuropil_A": "red", "neuropil_B": "blue"},
            backend="plotly",
        )
        assert fig is not None

    def test_with_title(self, single_mesh):
        """Render with a title annotation."""
        fig = show_neuropil(
            meshes=[single_mesh],
            names=["test"],
            title="Test Neuropil",
            backend="plotly",
        )
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Neuropil"

    def test_navis_volume_input(self):
        """Accept a pre-built navis.Volume."""
        verts, faces = _tetrahedron()
        vol = navis.Volume(verts, faces, name="pre_built")
        fig = show_neuropil(
            meshes={"pre_built": vol},
            backend="plotly",
        )
        assert fig is not None

    def test_empty_mesh_list(self):
        """Empty mesh list renders without error."""
        fig = show_neuropil(meshes={}, backend="plotly")
        assert fig is not None


# ---------------------------------------------------------------------------
# Tests: show_neurons
# ---------------------------------------------------------------------------

class TestShowNeurons:

    def test_single_neuron(self, sample_neuron):
        """Render a single MeshNeuron."""
        fig = show_neurons(
            neurons=sample_neuron,
            backend="plotly",
        )
        assert fig is not None

    def test_neuron_list(self, sample_neuron):
        """Render a NeuronList."""
        nl = navis.NeuronList([sample_neuron])
        fig = show_neurons(neurons=nl, backend="plotly")
        assert fig is not None

    def test_color_by_nt(self, sample_neuron, sample_annotations):
        """Color neurons by neurotransmitter type."""
        fig = show_neurons(
            neurons=sample_neuron,
            color_by="top_nt",
            annotations=sample_annotations,
            backend="plotly",
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# Tests: show_neuropil_connectivity
# ---------------------------------------------------------------------------

class TestShowNeuropilConnectivity:

    def test_basic_connectivity(self):
        """Render simple 2-neuropil connectivity."""
        centroids = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([5.0, 0.0, 0.0]),
        }
        weights = pd.DataFrame(
            [[0, 100], [50, 0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        fig = show_neuropil_connectivity(
            centroids=centroids,
            weights=weights,
        )
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        # Should have at least edge traces + centroid markers
        assert len(fig.data) >= 2

    def test_with_background_meshes(self, mesh_dict):
        """Render connectivity with background neuropil meshes."""
        centroids = {
            "neuropil_A": np.array([0.5, 0.5, 0.25]),
            "neuropil_B": np.array([3.5, 0.5, 0.25]),
        }
        weights = pd.DataFrame(
            [[0, 200], [150, 0]],
            index=["neuropil_A", "neuropil_B"],
            columns=["neuropil_A", "neuropil_B"],
        )
        fig = show_neuropil_connectivity(
            centroids=centroids,
            weights=weights,
            meshes=mesh_dict,
        )
        assert fig is not None

    def test_empty_connectivity(self):
        """No edges when all weights are zero."""
        centroids = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([5.0, 0.0, 0.0]),
        }
        weights = pd.DataFrame(
            [[0, 0], [0, 0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        fig = show_neuropil_connectivity(
            centroids=centroids,
            weights=weights,
        )
        assert fig is not None

    def test_top_n_filtering(self):
        """Only top_n edges are drawn."""
        centroids = {n: np.array([i * 2.0, 0, 0]) for i, n in enumerate("ABCD")}
        data = np.array([
            [0, 10, 20, 30],
            [10, 0, 5, 15],
            [20, 5, 0, 25],
            [30, 15, 25, 0],
        ])
        weights = pd.DataFrame(data, index=list("ABCD"), columns=list("ABCD"))
        fig = show_neuropil_connectivity(
            centroids=centroids,
            weights=weights,
            top_n=3,
        )
        # Count line traces (each edge = one Scatter3d with mode="lines")
        line_traces = [t for t in fig.data if hasattr(t, 'mode') and t.mode == "lines"]
        assert len(line_traces) == 3


# ---------------------------------------------------------------------------
# Tests: color palettes
# ---------------------------------------------------------------------------

class TestColorPalettes:

    def test_nt_colors_has_common_types(self):
        """NT palette covers the main neurotransmitters."""
        for nt in ["acetylcholine", "gaba", "glutamate", "dopamine"]:
            assert nt in NT_COLORS

    def test_superclass_colors_has_common_types(self):
        """Super-class palette covers common FlyWire classes."""
        for sc in ["central", "optic", "sensory", "motor"]:
            assert sc in SUPERCLASS_COLORS
