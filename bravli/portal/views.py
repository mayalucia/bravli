"""View-building functions for each portal tab.

Each view function takes data sources as arguments and returns
a Panel layout. Views are composable and testable independently.
"""

import numpy as np
import pandas as pd

try:
    import panel as pn
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from bravli.utils import get_logger

LOG = get_logger("portal.views")


def _require_panel():
    if not HAS_PANEL:
        raise ImportError(
            "Panel is required for the portal. Install with: pip install panel"
        )


def _empty_figure(title="No data"):
    """A placeholder plotly figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Atlas
# ---------------------------------------------------------------------------

def atlas_view(annotations, render_fn=None, detail_fn=None, neuropil_groups=None):
    """Build the atlas exploration tab.

    Parameters
    ----------
    annotations : pd.DataFrame
        FlyWire neuron annotations.
    render_fn : callable, optional
        Function(highlight_groups) -> go.Figure. If None, uses a placeholder.
    detail_fn : callable, optional
        Function(group) -> go.Figure for neuropil detail views.
    neuropil_groups : list of str, optional
        Available neuropil groups for the selector.

    Returns
    -------
    pn.Column
    """
    _require_panel()

    groups = neuropil_groups or [
        "(none)", "mushroom_body", "antennal_lobe", "central_complex",
        "lateral_horn", "superior_protocerebrum",
        "ventrolateral_protocerebrum",
        "medulla", "lobula", "lobula_plate", "lamina",
        "gnathal_ganglion",
    ]

    group_select = pn.widgets.Select(
        name="Highlight region", options=groups, value="(none)",
    )

    n_neurons = len(annotations)
    n_types = annotations["cell_type"].nunique() if "cell_type" in annotations.columns else "?"
    super_counts = ""
    if "super_class" in annotations.columns:
        top = annotations["super_class"].value_counts().head(5)
        super_counts = ", ".join(f"{k}: {v:,}" for k, v in top.items())

    info_md = pn.pane.Markdown(
        f"### FlyWire Connectome\n"
        f"- **{n_neurons:,}** neurons, **{n_types}** cell types\n"
        f"- Top super-classes: {super_counts}\n"
        f"- 78 neuropil compartments\n\n"
        f"*Select a region to highlight. Each neuropil is a question: "
        f"what computation does this geometry serve?*",
        styles={"color": "#c9d1d9"},
    )

    @pn.depends(group_select)
    def _atlas_plot(group):
        if render_fn is not None:
            try:
                highlight = [group] if group != "(none)" else None
                fig = render_fn(highlight_groups=highlight)
                return pn.pane.Plotly(fig, sizing_mode="stretch_both")
            except Exception as e:
                LOG.warning("Atlas render failed: %s", e)
        return pn.pane.Plotly(
            _empty_figure("Atlas (requires navis + fafbseg)"),
            sizing_mode="stretch_both",
        )

    @pn.depends(group_select)
    def _detail_plot(group):
        if detail_fn is not None and group != "(none)":
            try:
                fig = detail_fn(group)
                return pn.pane.Plotly(fig, sizing_mode="stretch_both")
            except Exception as e:
                LOG.warning("Detail render failed: %s", e)
        return pn.pane.Markdown(
            "*Select a region above to see its compartments.*",
            styles={"color": "#8b949e"},
        )

    return pn.Column(
        pn.pane.Markdown("# Atlas", styles={"color": "#c9d1d9"}),
        info_md,
        group_select,
        pn.Row(_atlas_plot, _detail_plot, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Tab 2: Composition
# ---------------------------------------------------------------------------

def composition_view(annotations):
    """Build the composition exploration tab.

    Shows cell type distributions and neurotransmitter profiles
    per neuropil or super-class.
    """
    _require_panel()

    # Group-by selector
    groupby = pn.widgets.Select(
        name="Group by", options=["super_class", "cell_class", "cell_type"],
        value="super_class",
    )

    # NT filter
    nt_col = "top_nt" if "top_nt" in annotations.columns else None

    @pn.depends(groupby)
    def _composition_table(group_col):
        if group_col not in annotations.columns:
            return pn.pane.Markdown(f"*Column '{group_col}' not in annotations.*")
        counts = annotations[group_col].value_counts().head(30)
        df = counts.reset_index()
        df.columns = [group_col, "count"]
        return pn.pane.DataFrame(df, sizing_mode="stretch_width")

    @pn.depends(groupby)
    def _composition_bar(group_col):
        if group_col not in annotations.columns:
            return pn.pane.Plotly(_empty_figure("No data"))
        counts = annotations[group_col].value_counts().head(20)
        fig = go.Figure(go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker_color="#58a6ff",
        ))
        fig.update_layout(
            title=f"Top 20 by {group_col}",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    def _nt_pie():
        if nt_col and nt_col in annotations.columns:
            nt_counts = annotations[nt_col].value_counts()
            from bravli.viz.viz import NT_COLORS
            colors = [NT_COLORS.get(nt, "#888") for nt in nt_counts.index]
            fig = go.Figure(go.Pie(
                labels=nt_counts.index,
                values=nt_counts.values,
                marker=dict(colors=colors),
                hole=0.4,
            ))
            fig.update_layout(
                title="Neurotransmitter distribution",
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                height=400,
            )
            return pn.pane.Plotly(fig)
        return pn.pane.Markdown(
            "*NT profile requires 'top_nt' column in annotations.*",
            styles={"color": "#8b949e"},
        )

    provocation = pn.pane.Markdown(
        "> *8,453 cell types — but how many are functionally distinct? "
        "The classification is morphological. Two neurons with identical "
        "branching patterns may express different ion channels, respond to "
        "different neuromodulators, sit in different activity regimes. "
        "The type is a hypothesis, not a fact.*",
        styles={"color": "#8b949e", "font-style": "italic"},
    )

    return pn.Column(
        pn.pane.Markdown("# Composition", styles={"color": "#c9d1d9"}),
        provocation,
        groupby,
        pn.Row(_composition_bar, _composition_table, sizing_mode="stretch_both"),
        _nt_pie(),
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Tab 3: Connectivity
# ---------------------------------------------------------------------------

def connectivity_view(edges=None):
    """Build the connectivity exploration tab.

    Parameters
    ----------
    edges : pd.DataFrame, optional
        Processed edge table (output of connectivity pipeline).
        If None, shows instructions for loading data.
    """
    _require_panel()

    if edges is None or len(edges) == 0:
        return pn.Column(
            pn.pane.Markdown("# Connectivity", styles={"color": "#c9d1d9"}),
            pn.pane.Markdown(
                "**No edge data loaded.** To populate this tab:\n\n"
                "```python\n"
                "from bravli.connectivity import load_edges, threshold_edges, "
                "assign_dominant_nt\n"
                "edges = load_edges('data/zenodo/proofread_connections_783.feather')\n"
                "edges = threshold_edges(edges)\n"
                "edges = assign_dominant_nt(edges)\n"
                "```\n\n"
                "Then pass `edges=edges` to `build_portal()`.",
                styles={"color": "#c9d1d9"},
            ),
        )

    # Neuropil selector
    neuropils = sorted(edges["neuropil"].unique()) if "neuropil" in edges.columns else []
    neuropil_select = pn.widgets.Select(
        name="Neuropil", options=["(all)"] + neuropils, value="(all)",
    )

    # Top-N slider
    top_n = pn.widgets.IntSlider(name="Top N pathways", start=5, end=50, value=20)

    @pn.depends(neuropil_select, top_n)
    def _pathway_table(neuropil, n):
        subset = edges if neuropil == "(all)" else edges[edges["neuropil"] == neuropil]
        top = (subset.groupby(["pre_pt_root_id", "post_pt_root_id"])
               .agg({"syn_count": "sum"})
               .reset_index()
               .nlargest(n, "syn_count"))
        return pn.pane.DataFrame(top, sizing_mode="stretch_width")

    @pn.depends(neuropil_select)
    def _synapse_histogram(neuropil):
        subset = edges if neuropil == "(all)" else edges[edges["neuropil"] == neuropil]
        syn = subset["syn_count"].values
        fig = go.Figure(go.Histogram(
            x=syn, nbinsx=50,
            marker_color="#f0883e",
        ))
        fig.update_layout(
            title=f"Synapse count distribution ({neuropil})",
            xaxis_title="Synapses per edge",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            yaxis_type="log",
            height=400,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    @pn.depends(neuropil_select)
    def _nt_breakdown(neuropil):
        subset = edges if neuropil == "(all)" else edges[edges["neuropil"] == neuropil]
        if "dominant_nt" not in subset.columns:
            return pn.pane.Markdown("*Run assign_dominant_nt() first.*")
        counts = subset.groupby("dominant_nt")["syn_count"].sum()
        from bravli.viz.viz import NT_COLORS
        colors = [NT_COLORS.get(nt, "#888") for nt in counts.index]
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=colors,
        ))
        fig.update_layout(
            title=f"Synapses by NT type ({neuropil})",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=350,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    provocation = pn.pane.Markdown(
        "> *54.5 million synapses, but we threshold at 5 and discard 80% "
        "of edges. The weak connections we throw away — are they noise, or are "
        "they the substrate of neuromodulatory volume transmission, of slow "
        "learning, of context-dependent gating? The Dorkenwald threshold is a "
        "pragmatic choice, not a biological one.*",
        styles={"color": "#8b949e", "font-style": "italic"},
    )

    return pn.Column(
        pn.pane.Markdown("# Connectivity", styles={"color": "#c9d1d9"}),
        provocation,
        pn.Row(neuropil_select, top_n),
        pn.Row(_synapse_histogram, _nt_breakdown, sizing_mode="stretch_both"),
        _pathway_table,
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Tab 4: Physiology
# ---------------------------------------------------------------------------

def physiology_view(edges=None):
    """Build the synapse physiology exploration tab.

    Parameters
    ----------
    edges : pd.DataFrame, optional
        Edge table with physiology columns (from assign_synapse_models).
    """
    _require_panel()

    if edges is None:
        return pn.Column(
            pn.pane.Markdown("# Physiology", styles={"color": "#c9d1d9"}),
            pn.pane.Markdown(
                "**No physiology data.** Run the connectivity + physiology pipeline first.",
                styles={"color": "#c9d1d9"},
            ),
        )

    # Synapse model database summary
    from bravli.physiology.synapse_models import SYNAPSE_DB
    model_rows = []
    for name, model in SYNAPSE_DB.items():
        model_rows.append(model.to_dict())
    model_df = pd.DataFrame(model_rows)

    # Weight distribution
    has_weight = "weight" in edges.columns

    def _model_table():
        return pn.pane.DataFrame(model_df, sizing_mode="stretch_width")

    def _weight_dist():
        if not has_weight:
            return pn.pane.Markdown("*No 'weight' column — run compute_synaptic_weights().*")
        w = edges["weight"].dropna().values
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=w[w > 0], name="Excitatory", marker_color="#58a6ff",
            nbinsx=50, opacity=0.7,
        ))
        fig.add_trace(go.Histogram(
            x=w[w < 0], name="Inhibitory", marker_color="#f85149",
            nbinsx=50, opacity=0.7,
        ))
        fig.update_layout(
            title="Synaptic weight distribution",
            xaxis_title="Weight (mV)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            barmode="overlay",
            height=400,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    provocation = pn.pane.Markdown(
        "> *Every synapse gets the same W_syn = 0.275 mV per contact, "
        "modulated only by sign and count. In reality, a Kenyon cell synapse "
        "onto an MBON has been sculpted by hours of olfactory experience — "
        "depression here, potentiation there, a history written in "
        "phosphorylated proteins and reshuffled receptors. Our model is "
        "amnesic. It has the connectome's anatomy but none of its biography.*",
        styles={"color": "#8b949e", "font-style": "italic"},
    )

    return pn.Column(
        pn.pane.Markdown("# Physiology", styles={"color": "#c9d1d9"}),
        provocation,
        pn.pane.Markdown("### Synapse Model Database", styles={"color": "#c9d1d9"}),
        _model_table(),
        pn.pane.Markdown("### Weight Distribution", styles={"color": "#c9d1d9"}),
        _weight_dist(),
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Tab 5: Simulate
# ---------------------------------------------------------------------------

def simulate_view(circuit=None):
    """Build the simulation tab.

    Parameters
    ----------
    circuit : Circuit, optional
        A pre-built circuit. If None, builds a small demo circuit.
    """
    _require_panel()

    from bravli.simulation.engine import simulate
    from bravli.simulation.stimulus import step_stimulus, poisson_stimulus
    from bravli.simulation.analysis import (
        firing_rates, spike_raster, population_rate, active_fraction,
    )

    # --- Demo circuit if none provided ---
    if circuit is None:
        from bravli.simulation.circuit import Circuit
        rng = np.random.RandomState(42)
        n = 100
        n_exc = 80
        n_inh = 20
        # Random sparse connectivity: ~5% density
        n_syn = 500
        pre = rng.randint(0, n, n_syn)
        post = rng.randint(0, n, n_syn)
        mask = pre != post
        pre, post = pre[mask], post[mask]
        weights = np.where(pre < n_exc, 0.5, -1.0) * rng.uniform(0.5, 1.5, len(pre))

        circuit = Circuit(
            n_neurons=n,
            v_rest=np.full(n, -52.0),
            v_thresh=np.full(n, -45.0),
            v_reset=np.full(n, -52.0),
            tau_m=np.full(n, 20.0),
            t_ref=np.full(n, 2.2),
            pre_idx=pre.astype(np.int32),
            post_idx=post.astype(np.int32),
            weights=weights,
            tau_syn=5.0,
            delay_steps=18,
        )

    # --- Scale-aware defaults ---
    is_large = circuit.n_neurons > 10000
    default_dur = 100 if is_large else 500
    max_dur = 500 if is_large else 2000
    max_stim = min(200, circuit.n_neurons) if is_large else min(50, circuit.n_neurons)
    default_stim = 50 if is_large else 10
    n_record = min(10, circuit.n_neurons)

    # --- Widgets ---
    duration_slider = pn.widgets.IntSlider(
        name="Duration (ms)", start=50, end=max_dur, step=50, value=default_dur,
    )
    n_stim = pn.widgets.IntSlider(
        name="Stimulated neurons", start=1, end=max_stim,
        step=1, value=default_stim,
    )
    stim_type = pn.widgets.Select(
        name="Stimulus", options=["poisson", "step"], value="poisson",
    )
    stim_strength = pn.widgets.FloatSlider(
        name="Stimulus strength", start=1.0, end=100.0, step=1.0, value=50.0,
    )
    run_button = pn.widgets.Button(name="Run simulation", button_type="primary")

    # --- Result storage ---
    result_holder = [None]

    size_note = ""
    if is_large:
        size_note = (
            " **Large circuit** — simulation may take minutes. "
            "Start with short durations (50-100 ms)."
        )

    status_md = pn.pane.Markdown(
        f"*Circuit: {circuit.n_neurons:,} neurons, {circuit.n_synapses:,} synapses."
        f"{size_note} Configure and press Run.*",
        styles={"color": "#c9d1d9"},
    )

    raster_pane = pn.pane.Plotly(_empty_figure("Spike raster (run simulation first)"))
    rate_pane = pn.pane.Plotly(_empty_figure("Population rate"))
    trace_pane = pn.pane.Plotly(_empty_figure("Voltage traces"))

    def _run(event):
        dur = duration_slider.value
        n_steps = int(dur / 0.1)
        targets = list(range(n_stim.value))

        if stim_type.value == "poisson":
            stim, _ = poisson_stimulus(
                circuit.n_neurons, n_steps, targets,
                rate_hz=stim_strength.value, weight=68.75, seed=42,
            )
        else:
            stim, _ = step_stimulus(
                circuit.n_neurons, n_steps, targets,
                amplitude=stim_strength.value,
                start_ms=50.0, end_ms=dur - 50.0,
            )

        status_md.object = "*Running...*"
        result = simulate(
            circuit, duration=dur, dt=0.1, stimulus=stim,
            record_v=True, record_idx=list(range(n_record)),
        )
        result_holder[0] = result

        # Raster
        times, neurons = spike_raster(result)
        fig_raster = go.Figure()
        if len(times) > 0:
            fig_raster.add_trace(go.Scattergl(
                x=times, y=neurons, mode="markers",
                marker=dict(size=2, color="#58a6ff"),
            ))
        fig_raster.update_layout(
            title=f"Spike raster — {result.n_spikes} spikes, "
                  f"mean {result.mean_rate():.1f} Hz",
            xaxis_title="Time (ms)", yaxis_title="Neuron index",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=350,
        )
        raster_pane.object = fig_raster

        # Population rate
        t_bins, pop_rate = population_rate(result, bin_ms=10.0)
        fig_rate = go.Figure(go.Scatter(
            x=t_bins, y=pop_rate, mode="lines",
            line=dict(color="#f0883e", width=2),
        ))
        fig_rate.update_layout(
            title="Population firing rate",
            xaxis_title="Time (ms)", yaxis_title="Rate (Hz)",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=300,
        )
        rate_pane.object = fig_rate

        # Voltage traces
        if result.v_trace is not None:
            fig_v = go.Figure()
            for i in range(min(5, result.v_trace.shape[0])):
                t_axis = np.arange(result.v_trace.shape[1]) * result.dt
                fig_v.add_trace(go.Scatter(
                    x=t_axis, y=result.v_trace[i],
                    mode="lines", name=f"Neuron {result.recorded_idx[i]}",
                    line=dict(width=1),
                ))
            fig_v.update_layout(
                title="Membrane potential (first 5 recorded neurons)",
                xaxis_title="Time (ms)", yaxis_title="V (mV)",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=300,
            )
            trace_pane.object = fig_v

        frac = active_fraction(result, threshold_hz=1.0)
        status_md.object = (
            f"**Done.** {result.n_spikes} spikes, mean rate {result.mean_rate():.1f} Hz, "
            f"{frac*100:.1f}% neurons active (>1 Hz)."
        )

    run_button.on_click(_run)

    provocation = pn.pane.Markdown(
        f"> *You are watching {circuit.n_neurons:,} differential equations evolve in time. "
        "Each spike is a threshold crossing — a discontinuity in a continuous "
        "dynamical system. The raster plot is a projection: it shows you /when/ "
        "neurons fire but hides /why/. The voltage traces show the subthreshold "
        "struggle — the tug-of-war between excitation and inhibition, between "
        "the leak current pulling toward rest and the synaptic current pushing "
        "toward threshold. Change the stimulus strength. Watch the transition "
        "from silence to sparse firing to synchronous bursting. Where is the "
        "phase transition? Is it sharp or gradual? Does it depend on the E/I "
        "ratio?*",
        styles={"color": "#8b949e", "font-style": "italic"},
    )

    return pn.Column(
        pn.pane.Markdown("# Simulate", styles={"color": "#c9d1d9"}),
        provocation,
        pn.Row(duration_slider, n_stim, stim_type, stim_strength),
        run_button,
        status_md,
        raster_pane,
        pn.Row(rate_pane, trace_pane, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )
