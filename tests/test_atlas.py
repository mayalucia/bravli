"""Smoke tests for bravli.atlas.

Tests verify that atlas functions execute without error using the
bundled fafbseg neuropil meshes.  Skeleton tests require the Zenodo
parquet and are skipped if not present.
"""

import pytest
import numpy as np
from pathlib import Path

fafbseg = pytest.importorskip("fafbseg")
plotly = pytest.importorskip("plotly")

from bravli.atlas.neuropils import (
    list_neuropils,
    load_neuropil,
    load_neuropils,
    NEUROPIL_GROUPS,
)
from bravli.atlas.render import (
    render_atlas,
    render_neuropil_detail,
)

PARQUET_PATH = Path(__file__).parent.parent / "data" / "zenodo" / "sk_lod1_783_healed_ds2.parquet"


# ---------------------------------------------------------------------------
# Neuropil tests
# ---------------------------------------------------------------------------

class TestNeuropils:

    def test_list_returns_78(self):
        """All 78 neuropils are available."""
        names = list_neuropils()
        assert len(names) == 78

    def test_load_single(self):
        """Loading a single neuropil returns a mesh."""
        vol = load_neuropil("MB_CA_R")
        assert hasattr(vol, "vertices")
        assert hasattr(vol, "faces")
        assert len(vol.vertices) > 100

    def test_load_group(self):
        """Loading a group returns dict of meshes."""
        meshes = load_neuropils(group="mushroom_body")
        assert len(meshes) == 8
        assert "MB_CA_R" in meshes

    def test_load_all(self):
        """Loading all neuropils returns 78 meshes."""
        meshes = load_neuropils()
        assert len(meshes) == 78

    def test_groups_cover_known_regions(self):
        """Neuropil groups include major brain regions."""
        assert "mushroom_body" in NEUROPIL_GROUPS
        assert "central_complex" in NEUROPIL_GROUPS
        assert "antennal_lobe" in NEUROPIL_GROUPS

    def test_mb_has_four_compartments_per_side(self):
        """MB group has calyx, pedunculus, vertical and medial lobes."""
        mb = NEUROPIL_GROUPS["mushroom_body"]
        for part in ["CA", "PED", "VL", "ML"]:
            assert f"MB_{part}_R" in mb
            assert f"MB_{part}_L" in mb


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------

class TestRender:

    def test_render_atlas_no_neurons(self):
        """Whole-brain atlas renders without neurons."""
        fig = render_atlas()
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 78  # at least one trace per neuropil

    def test_render_atlas_with_highlight(self):
        """Atlas with highlighted groups renders."""
        fig = render_atlas(highlight_groups=["mushroom_body", "central_complex"])
        assert fig is not None

    def test_render_neuropil_detail(self):
        """Neuropil detail view renders."""
        fig = render_neuropil_detail("mushroom_body")
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        # Should have 8 MB compartment traces
        mesh_traces = [t for t in fig.data if isinstance(t, go.Mesh3d)]
        assert len(mesh_traces) == 8


# ---------------------------------------------------------------------------
# Skeleton tests (require Zenodo parquet)
# ---------------------------------------------------------------------------

class TestSkeletons:

    pytestmark = pytest.mark.skipif(
        not PARQUET_PATH.exists(),
        reason="Zenodo skeleton parquet not found",
    )

    def test_load_single_skeleton(self):
        """Load a single skeleton by root ID."""
        import pandas as pd
        from bravli.atlas.skeletons import load_skeletons, sample_neuron_ids

        ann = pd.read_csv(
            Path(__file__).parent.parent / "data" / "flywire_annotations" / "Supplemental_file1_neuron_annotations.tsv",
            sep="\t", low_memory=False,
            usecols=["root_id", "cell_class", "cell_type", "side"],
        )
        ids = sample_neuron_ids(ann, cell_class="Kenyon_Cell", n=1)
        assert len(ids) == 1

        neurons = load_skeletons(ids, parquet_path=PARQUET_PATH)
        assert len(neurons) == 1
        assert neurons[0].n_nodes > 100

    def test_render_with_skeletons(self):
        """Atlas with skeleton overlay renders."""
        import pandas as pd
        from bravli.atlas.skeletons import load_skeletons, sample_neuron_ids

        ann = pd.read_csv(
            Path(__file__).parent.parent / "data" / "flywire_annotations" / "Supplemental_file1_neuron_annotations.tsv",
            sep="\t", low_memory=False,
            usecols=["root_id", "cell_class", "cell_type", "side"],
        )
        ids = sample_neuron_ids(ann, cell_class="MBON", n=2)
        neurons = load_skeletons(ids, parquet_path=PARQUET_PATH)

        fig = render_neuropil_detail("mushroom_body", neurons=neurons)
        assert fig is not None
        # Should have mesh traces + skeleton traces
        assert len(fig.data) > 8
