"""
PSP features.
"""

from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ExpFitParams = namedtuple("ExpFitParams", ["tau", "phi", "err"])


class ExpFitType(Enum):
    """Type of an exponential fit."""
    DSC = 0
    ASC = 1


def fit_exponential(exptype, trace, variable):
    """..."""

    def get_exponential(time, tau, phi):
        """..."""
        begin_time = time[0]
        time_since = time - begin_time
        exposed = np.exp(-time_since / tau)
        if exptype == ExpFitType.ASC:
            return phi * (1. - exposed)
        if exptype == ExpFitType.DSC:
            return phi * exposed
        raise ValueError("Unknown exponential fit type {}".format(exptype))

    if trace.empty:
        return ExpFitParams(np.nan, np.nan, np.nan)
    popt, pcov = curve_fit(get_exponential,
                           trace.time.to_numpy(np.float),
                           trace[variable].to_numpy(np.float))
    return ExpFitParams(popt[0], popt[1], np.sqrt(np.diag(pcov)))


def _scale_y(trace, peak, Y0=None):
    """
    Append column for normalized Y
    """
    if Y0 is None:
        Y0 = trace.Y.iloc[0]
    return (trace.Y - Y0) / (peak.Y - Y0)


def get_rise_curve(trace, from_time=0.0, upto_time=np.infty):
    """
    Rising part of a trace.

    Assumption
    ----------
    That there is only one maximum in the trace...
    """
    in_time = trace[np.logical_and(from_time <= trace.time, trace.time < upto_time)]

    if in_time.empty:
        raise ValueError("""
        Empty trace component in time window: ({}, {})
        """.format(from_time, upto_time))

    peak = in_time.iloc[np.argmax(in_time.Y.to_numpy(np.float))]
    return (in_time[in_time.time <= peak.time]
            .assign(y=lambda t: _scale_y(t, peak)))


def get_clamp_relaxation(trace, t_stim):
    """..."""
    prestim = trace[np.logical_and(trace.time < t_stim)]
    bottom = prestim.iloc[np.argmin(prestim.Y)]
    raise NotImplementedError


def get_decay_curve(trace, from_time=0.0, upto_time=np.infty):
    """
    Decaying part of a trace

    Assumption
    ------------
    That there is only one maximum in the trace...
    """
    in_time = trace[np.logical_and(from_time <= trace.time, trace.time < upto_time)]

    if in_time.empty:
        raise ValueError("""
        Empty trace component in time window: ({}, {})
        """.format(from_time, upto_time))

    peak = in_time.iloc[np.argmax(in_time.Y.to_numpy(np.float))]
    return in_time[in_time.time >= peak.time].assign(y=lambda t: _scale_y(t, 0.))


def get_velocity_curve(trace, from_time=0.0, upto_time=np.infty):
    """
    Rate at which Y increases.
    """
    in_time = trace[np.logical_and(from_time <= trace.time, trace.time < upto_time)]
    Y = in_time.Y.to_numpy(np.float)
    T = in_time.time.to_numpy(np.float)

    return pd.DataFrame({"time": T[:-1],
                         "Y": Y[:-1],
                         "velocity": (Y[1:] - Y[:-1]) / (T[1:] - T[:-1])})


def measure(traces,
            quantity,
            using_mean_traces,
            by_connection,
            statistics=None):
    """..."""

    def mean_quantity_over_trials(connection_traces):
        return connection_traces.groupby("trial").apply(quantity).mean()

    mean_quantity_over_trials.__name__ = quantity.__name__

    def summarize_quantity_over_trials(connection_traces):
        stats = ["mean", "std"]
        return connection_traces.groupby("trial").apply(quantity).agg(stats)

    if using_mean_traces:
        values = traces.map(quantity, over_mean_traces=True)
    elif by_connection:
        values = traces.map(summarize_quantity_over_trials, by_connection=True)
    else:
        values = traces.map(quantity)
    return values if not statistics else values.agg(statistics)


def get_rise_time(traces,
                  from_fraction=0.1,
                  to_fraction=0.9,
                  using_mean_traces=False,
                  by_connection=False,
                  statistics=None):
    """..."""

    def risetime(trace):
        rise_curve = get_rise_curve(trace)
        time_from = rise_curve.time[np.searchsorted(rise_curve.y, from_fraction)]
        time_upto = rise_curve.time[np.searchsorted(rise_curve.y, to_fraction)]
        return time_upto - time_from

    return measure(traces.responsive.post_stimulation,
                   risetime,
                   using_mean_traces=using_mean_traces,
                   by_connection=by_connection,
                   statistics=statistics)


def get_latency(traces,
                to_fraction=0.05,
                using_mean_traces=False,
                by_connection=False,
                statistics=None):

    """..."""

    def latency(trace):
        rise_curve = get_rise_curve(trace)
        return rise_curve.time[np.searchsorted(rise_curve.y, to_fraction)]

    return measure(traces.responsive.post_stimulation,
                   latency,
                   using_mean_traces=using_mean_traces,
                   by_connection=by_connection,
                   statistics=statistics)


def get_decay_time(traces,
                   from_fraction=0.8,
                   to_fraction=0.2,
                   using_mean_traces=False,
                   by_connection=False,
                   statistics=None,
                   with_error=False):
    """..."""

    def decay(trace):
        curve = get_decay_curve(trace)
        y = curve.y.to_numpy(np.float)
        window = np.logical_and(from_fraction >= y, y > to_fraction)
        fit = fit_exponential(ExpFit.DSC, curve[window], "y")
        return fit.tau if not with_error else fit

    return measure(traces.responsive.post_stimulation,
                   decay,
                   using_mean_traces=using_mean_traces,
                   by_connection=by_connection,
                   statistics=statistics)


def get_fluctuation_size(traces, connection):
    """
    How big are voltage fluctuations in a connection's traces?
    Fluctuations can be hard to determine in the presence of a signal.
    Thus we use the voltage-clamped part of a trace just before stimulation
    onset.
    """
    raise NotImplementedError("Does this method even belong here?")


def get_psp_amplitude(traces,
                      using_mean_traces=False,
                      by_connection=False,
                      statistics=None):
    """..."""

    def psp_amplitude(trace):
        return trace.Y.max()

    return measure(traces.responsive.post_stimulation,
                   psp_amplitude,
                   using_mean_traces=using_mean_traces,
                   by_connection=by_connection,
                   statistics=statistics)


def get_psp_cv(traces, statistics=None):
    """..."""

    def psp_cv(trials):
        amplitudes = trials.groupby("trial").apply(lambda trace: trace.Y.max())
        return amplitudes.std() / amplitudes.mean()

    values = traces.filtered.post_stimulation.map(psp_cv, by_connection=True)
    return values if not statistics else values.agg(statistics)


def get_failure_rate(traces, lower_bound_success=0.01, statistics=None):
    """..."""

    def failure(trials):
        return (trials.groupby("trial")
                .apply(lambda trace: trace.Y.max() < lower_bound_success)
                .mean())

    values = traces.filtered.post_stimulation.map(failure, by_connection=True)
    return values if not statistics else values.agg(statistics)
