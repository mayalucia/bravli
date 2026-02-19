"""Smoke tests for the portal module.

Tests that views can be constructed without errors.
Does NOT test Panel rendering (requires a browser).
"""

import numpy as np
import pandas as pd
import pytest

from bravli.simulation.circuit import Circuit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_annotations():
    """Minimal annotation DataFrame."""
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "root_id": range(100),
        "super_class": rng.choice(
            ["central", "optic", "sensory", "motor"], 100
        ),
        "cell_class": rng.choice(
            ["KC", "PN", "MBON", "Mi", "T4"], 100
        ),
        "cell_type": [f"type_{i % 20}" for i in range(100)],
        "top_nt": rng.choice(
            ["acetylcholine", "GABA", "glutamate"], 100
        ),
    })


@pytest.fixture
def sample_edges():
    """Minimal edge DataFrame."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "pre_pt_root_id": rng.randint(0, 50, n),
        "post_pt_root_id": rng.randint(0, 50, n),
        "neuropil": rng.choice(["MB", "AL", "LH"], n),
        "syn_count": rng.randint(1, 100, n),
        "dominant_nt": rng.choice(
            ["acetylcholine", "GABA", "glutamate"], n
        ),
        "weight": rng.uniform(-2.0, 2.0, n),
    })


@pytest.fixture
def demo_circuit():
    """Small demo circuit."""
    rng = np.random.RandomState(42)
    n = 20
    pre = rng.randint(0, n, 40).astype(np.int32)
    post = rng.randint(0, n, 40).astype(np.int32)
    mask = pre != post
    return Circuit(
        n_neurons=n,
        v_rest=np.full(n, -52.0),
        v_thresh=np.full(n, -45.0),
        v_reset=np.full(n, -52.0),
        tau_m=np.full(n, 20.0),
        t_ref=np.full(n, 2.2),
        pre_idx=pre[mask],
        post_idx=post[mask],
        weights=rng.uniform(-1, 1, mask.sum()),
        tau_syn=5.0,
        delay_steps=18,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestViews:
    def test_atlas_view(self, sample_annotations):
        from bravli.portal.views import atlas_view
        view = atlas_view(sample_annotations)
        assert view is not None

    def test_composition_view(self, sample_annotations):
        from bravli.portal.views import composition_view
        view = composition_view(sample_annotations)
        assert view is not None

    def test_connectivity_view_with_data(self, sample_edges):
        from bravli.portal.views import connectivity_view
        view = connectivity_view(edges=sample_edges)
        assert view is not None

    def test_connectivity_view_no_data(self):
        from bravli.portal.views import connectivity_view
        view = connectivity_view(edges=None)
        assert view is not None

    def test_physiology_view_with_data(self, sample_edges):
        from bravli.portal.views import physiology_view
        view = physiology_view(edges=sample_edges)
        assert view is not None

    def test_physiology_view_no_data(self):
        from bravli.portal.views import physiology_view
        view = physiology_view(edges=None)
        assert view is not None

    def test_simulate_view_with_circuit(self, demo_circuit):
        from bravli.portal.views import simulate_view
        view = simulate_view(circuit=demo_circuit)
        assert view is not None

    def test_simulate_view_demo(self):
        from bravli.portal.views import simulate_view
        view = simulate_view(circuit=None)
        assert view is not None


class TestApp:
    def test_build_portal_minimal(self, sample_annotations):
        from bravli.portal.app import build_portal
        portal = build_portal(annotations=sample_annotations)
        assert portal is not None

    def test_build_portal_full(self, sample_annotations, sample_edges, demo_circuit):
        from bravli.portal.app import build_portal
        portal = build_portal(
            annotations=sample_annotations,
            edges=sample_edges,
            circuit=demo_circuit,
        )
        assert portal is not None

    def test_build_portal_no_data(self):
        from bravli.portal.app import build_portal
        portal = build_portal()
        assert portal is not None
