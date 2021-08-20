"""
Code to help analyze PSP traces.

Plots, summaries, fits, etc.
"""
from enum import Enum
from lazy import lazy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from ...utils import timestamp# pylint: disable=relative-beyond-top-level
from .traces import TraceCollection# pylint: disable=relative-beyond-top-level


class ReportScheme(Enum):
    """
    Schemes to structure folders where the report will be saved.
    """
    PATHWAY = 0
    SECTION = 1


# pylint: disable=too-many-arguments
def get_path_document(scheme, section, doctype, pathway,
                      variable=None):
    """..."""
    if scheme == ReportScheme.PATHWAY:
        parent = Path(pathway) / section
        return (parent, doctype) if not variable else (parent / doctype, variable)

    if scheme == ReportScheme.SECTION:
        parent = Path(section) / doctype
        return (parent / variable if variable else parent, pathway)

    raise ValueError(scheme)

# pylint: disable=invalid-name
def _plot_baseline(self, graphic, variable, baseline, dy, **kwargs):
    """..."""
    raise NotImplementedError

# pylint: disable=invalid-name
def plot_prestim(self,
                 variable="voltage",
                 baseline=None,
                 dy=0.2,
                 show_legend=False,
                 **kwargs):
    """
    Plot the pre-stimulation part of traces,
    and save the figure.

    Arguments
    ---------------
    `self` : A `TraceCollection` instance.
    `dy` : The extra amount of y axis to include in addition to its values in the
    ~      plotted data.
    `baseline`: Float #Clamp voltage.
    ~           If not provided, its value will be computed during the end
    steady state voltage just before the arrival of the first spike.
    """
    prestim_means = self.mean.mask_time_window(0.0, self.t_stim)
    plt.figure(figsize=(15, 12))
    graphic = sn.lineplot(x="time", y=variable,
                          hue="connection",
                          data=prestim_means.raw.reset_index())
    graphic.legend().set_visible(show_legend)

    return _plot_baseline(prestim_means, graphic, variable, baseline, dy, **kwargs)


def _get_path_figure(scheme, pathway, variable, value, dirtime,
                     output_path=Path.cwd()/"figures"):
    """..."""
    reldirpath, basename = get_path_document(scheme, value, "figure",
                                             pathway, variable)
    dirpath = output_path / reldirpath / dirtime
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath / "{}.png".format(basename)


def report_prestim(
    filepath_config,
    filepath_synapse_classes,
    dirpath_traces,
    pathways=None,
    output_path=Path.cwd().joinpath("reports"),
    schema=(ReportScheme.PATHWAY, ReportScheme.SECTION),
    **kwargs
):
    """..."""
    dirtime = timestamp(datetime.now())

    def save(pathway, variable):
        print("Analyze pathway {} {}.".format(pathway, variable))
        traces = TraceCollection.load(
            filepath_config / "{}.yaml".format(pathway),
            filepath_synapse_classes,
            dirpath_traces,
            **kwargs
        )
        graphic = plot_prestim(traces, variable, **kwargs)

        def save(scheme):
            path = _get_path_figure(scheme, pathway, variable, "prestim",
                                    dirtime, output_path)
            graphic.figure.savefig(path)
            return path

        try:
            ischema = iter(schema)
        except TypeError:
            return save(schema)
        return tuple(save(scheme) for scheme in ischema)

    pathways = pathways or traces.pathways
    
    plots = {variable: {save(pathway, variable) for pathway in pathways}
             for variable in ["voltage", "Y"]}

    return {"plots": plots}


def plot_stochasticity(self,
                       variable="voltage",
                       baseline=None,
                       dy=0.2,
                       output_path=Path.cwd() / "figures",
                       show_legend=False,
                       figsize=(15, 12),
                       **kwargs):
    """
    Plot the stochastic / deterministic (mean-)traces for a pathway.
    """
    deterministic = self.with_kinetics.deterministic.mean
    figure_deterministic = plt.figure(figsize=figsize)
    graphic_deterministic = sn.lineplot(
        x="time", y=variable, hue="connection", data=deterministic.raw.reset_index()
    )
    _plot_baseline(deterministic, graphic_deterministic, variable, baseline, dy)
    graphic_deterministic.get_legend().set_visible(False)

    stochastic = self.with_kinetics.stochastic.mean
    figure_stochastic = plt.figure(figsize=figsize)
    graphic_stochastic = sn.lineplot(
        x="time", y=variable, hue="connection", data=stochastic.reset_index()
    )
    _plot_baseline(stochastic, graphic_stochastic, variable, baseline, dy)

    return {"deterministic": graphic_deterministic, "stochastic": graphic_stochastic}


class AnalysisReport:
    """
    Analyze PSP traces of pathways.
    """

    def __init__(
        self,
        filepath_config,
        filepath_synapse_classes,
        dirpath_traces,
        pathways=None,
        figsize=(15, 12),
        output_path=Path.cwd().joinpath("reports"),
        output_schema=(ReportScheme.PATHWAY, ReportScheme.SECTION),
    ):
        """..."""
        self._dirtime = timestamp(datetime.now())
        self._filepath_config = filepath_config
        self._filepath_synapse_classes = filepath_synapse_classes
        self._dirpath_traces = dirpath_traces
        self._pathways = pathways
        self._figsize = figsize
        self._output_path = output_path
        self._output_schema = output_schema

    @staticmethod
    def _plot_baseline(traces, axes, variable, baseline, dy):
        """..."""
        if not baseline:
            baseline = traces.get_baseline()

        if variable == "voltage":
            variable_baseline = baseline
        elif variable == "Y":
            variable_baseline = 0.0
        else:
            raise ValueError("Unknown variable {}".format(variable))

        value_min = np.min(traces.raw[variable])
        value_max = np.max(traces.raw[variable])

        axes.set_ylim(
            np.maximum(value_min, variable_baseline - dy),
            np.minimum(value_max, variable_baseline + dy),
        )
        return axes

    @lazy
    def path_figures(self):
        return self._output_path / "figures"

    def _save_plot(self, graphic, scheme, pathway, variable, theme):
        path = _get_path_figure(scheme, pathway, variable, theme, self._dirtime)
        graphic.figure.savefig(path)
        return path

    def plot_stochasticity(
        self,
        traces,
        variable="voltage",
        baseline=None,
        dy=0.2,
        show_legend=False,
        figsize=None,
        **kwargs
    ):
        """
        Plot the stochastic / deterministic mean-traces for a bunch of traces.
        """

        def _plot(data):
            _, ax = plt.subplots(figsize=(figsize or self._figsize))
            graphic = sn.lineplot(
                x="time",
                y=variable,
                hue="connection",
                data=data.mean.raw.reset_index(),
                ax=ax,
            )
            return self._plot_baseline(data.mean, graphic, variable, baseline, dy)

        return {
            "deterministic": _plot(traces.with_kinetics.deterministc),
            "stochastic": _plot(traces.with_kinetics.stochastic),
        }

    def plot_prestim(
        self,
        traces,
        variable="voltage",
        baseline=None,
        dy=0.2,
        show_legend=False,
        figsize=None,
        **kwargs
    ):
        """
        Plot the pre-stimulation part of traces.
        """
        prestim_means = traces.mean.mask_time_window(0.0, traces._t_stim)
        _, ax = plt.subplots(figsize=(figsize or self._figsize))
        graphic = sn.lineplot(
            x="time",
            y=variable,
            hue="connection",
            data=presetim_means.raw.reset_index(),
        )
        graphic.legend().set_visible(show_legend)

        return self._plot_baseline(presetim_means, graphic, variable, baseline, dy)

    def plot(self, traces, theme, **kwargs):
        """..."""
        if phenomenon == "stochasticity":
            return self.plot_stochasticity(traces, **kwargs)
        if phenomenon == "prestim":
            return self.plot_prestim(traces, **kwargs)
        return None

    @property
    def themes(self):
        """
        Themes of analysis.
        """

        def parse(attr):
            split = attr.split("_")
            if len(split) == 1 or split[0] != "plot":
                return None
            return "_".join(plot[1:])

        return [theme for theme in (parse(attr) for attr in dir(self)) if theme]

    def report(self, pathway, themes=None, **kwargs):
        """..."""
        traces = TraceCollection.load(
            self._filepath_config,
            self._filepath_synapse_classes,
            self._dirpath_traces,
            **kwargs
        )
        plots = {
            theme: self.plot(traces, theme, **kwargs)
            for theme in (themes or self.themes)
        }

        return plots


def report(
    plot_value,
    filepath_config,
    filepath_synapse_classes,
    dirpath_traces,
    pathways=None,
    output_path=Path.cwd().joinpath("reports"),
    schema=(ReportScheme.PATHWAY, ReportScheme.SECTION),
    **kwargs
):
    """..."""
    dirtime = timestamp(datetime.now())

    def save(pathway, variable):
        print("Analyze pathway {} {}.".format(pathway, variable))
        traces = TraceCollection.load(
            filepath_config / "{}.yaml".format(pathway),
            filepath_synapse_classes,
            dirpath_traces,
            **kwargs
        )
        graphics = plot_value(traces, variable, **kwargs)

        def save(scheme):
            def get_path(label=None):
                label = label or "_".join(plot_value.split("_")[1:])
                return _get_path_figure(scheme, pathway, variable, label, dirtime)

            try:
                igraphic = graphics.items()
            except TypeError:
                return graphics.figure.savefig(get_path())
            return {
                label: graphic.figure.savefig(get_path(label))
                for label, graphic in igraphic
            }

        try:
            ischema = iter(schema)
        except TypeError:
            return save(schema)
        return tuple(save(scheme) for scehem in ischema)

    pathways = pathways or traces.pathways
    plots = {
        variable: {save(pathway, variable) for pathway in pathways}
        for variable in ["voltage", "Y"]
    }
    return {plot_value.__name__: plots}


def report_stochasticity(
    filepath_config,
    filepath_synapse_classes,
    dirpath_traces,
    pathways=None,
    output_path=Path.cwd().joinpath("reports"),
    schema=(ReportScheme.PATHWAY, ReportScheme.SECTION),
    **kwargs
):
    """..."""
    return report(
        plot_stochasticity,
        filepath_config,
        filepath_synapse_classes,
        dirpath_traces,
        pathways=None,
        output_path=Path.cwd().joinpath("reports"),
        schema=(ReportScheme.PATHWAY, ReportScheme.SECTION),
        **kwargs
    )
