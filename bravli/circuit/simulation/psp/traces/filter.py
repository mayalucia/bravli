"""
Filter traces.
"""
from abc import (
    ABC,
    abstractproperty,
    abstractmethod,
    abstractclassmethod,
    abstractclassmethod,
)
import numpy as np
import pandas as pd


def mask_time_window(trace, time_from, time_upto):
    """..."""
    time_window = np.logical_and(time_from <= trace.time, trace.time <= time_upto)
    return trace[time_window]


class TraceFilter:
    """
    Filter traces.
    """

    @abstractmethod
    def check(self, trace, time_window, **metadata):
        """
        Does `trace` pass the filter?
        """
        raise NotImplementedError

    def get_mask(self, traces):
        """..."""
        return traces.map(self.check)

    def get_failing(self, traces):
        """..."""
        return traces.accumulate(t for t in traces if not self.check(t))

    @classmethod
    def _resolve_time_window(cls, from_time, upto_time):
        """..."""
        if not from_time:
            from_time = -np.infty
        if not upto_time:
            upto_time = np.infty
        return (from_time, upto_time)

    @classmethod
    def _mask(cls, trace, from_time, upto_time):
        """..."""
        if not upto_time and not from_time:
            return trace
        from_time, upto_time = cls._resolve_time_window(from_time, upto_time)
        in_time = np.logical_and(from_time <= trace.time, trace.time < upto_time)
        return traces[in_time]

    def __call__(self, traces, **metadata):
        """Filter traces...
        Arguments
        -----------
        metadata : Data defining the  filter...
        """
        return traces.accumulate(
            trace for trace in traces if self.check(trace, **metadata)
        )


class TimeWindowed:
    """
    Process using voltage / Y of a trace in a specified time window.
    """

    def __init__(self, from_time=0.0, upto_time=np.infty, *args, **kwargs):
        """
        Arguments
        -------------
        from_time: Start time to check for stochasticity.
        upto_time: Stop time to check for stochasticity.
        """
        self._from_time = from_time
        self._upto_time = upto_time
        super().__init__(*args, **kwargs)


class StochasticConnectionFilter(TraceFilter, TimeWindowed):
    """
    Filter traces that are stochastic.
    It is assumed that all traces for a connection will be stochastic.
    """

    def check(self, trace, stochastic_connections):
        """..."""
        raise NotImplementedError("Does not apply to this filter.")

    def get_connections(self, traces):
        time_statistics = traces.get_time_summary(
            "voltage", self._from_time, self._upto_time, statistics="mean"
        )
        variance_over_connections = time_statistics.groupby("connection").agg("var")
        is_deterministic = np.isclose(variance_over_connections, 0.0)
        is_stochastic = np.logical_not(is_deterministic)

        return {
            "stochastic": variance_over_connections[is_stochastic].index,
            "deterministic": variance_over_connections[is_deterministic].index,
        }

    def get_stochastic(self, traces):
        """..."""
        connections = self.get_connections(traces)
        return traces.with_data(traces.raw.loc[connections["stochastic"]])

    def get_deterministic(self, traces):
        connections = self.get_connections(traces)
        return traces.with_data(traces.raw.loc[connections["deterministic"]])

    def get_mask(self, traces):
        """..."""
        raise NotImplementedError("Does not apply to this filter.")

    def get_failing(self, traces):
        """..."""
        return self.get_deterministic(traces)

    def __call__(self, traces):
        """..."""
        return self.get_stochastic(traces)


class ExtremeVoltageFilter(TraceFilter):
    """
    Remove traces with high-voltage.
    """

    def __init__(self, upper_bound, *args, **kwargs):
        """
        Remove traces with an extreme value in them.
        """
        self._upper_bound = upper_bound
        super().__init__(*args, **kwargs)

    def check(self, trace, from_time=None, upto_time=None):
        masked = self._mask(trace, from_time, upto_time)
        return masked.voltage.max() < self._upper_bound


class LargeAmplitudeFilter(TraceFilter):
    """
    Remove traces with large amplitudes.
    """

    def __init__(self, upper_bound=5.0, *args, **kwargs):
        """
        Arguments
        ------------
        Upper bound on Y.
        """
        self._upper_bound = upper_bound
        super().__init__(*args, **kwargs)

    def check(self, trace, from_time=None, upto_time=None):
        masked = self._mask(trace, from_time, upto_time)
        return masked.Y.max() < self._upper_bound


class LowAmplitudeFilter(TraceFilter):
    """..."""

    def __init__(self, lower_bound=0.01, *args, **kwargs):
        """..."""
        self._lower_bound = lower_bound
        super().__init__(*args, **kwargs)

    def check(self, trace, from_time=None, upto_time=None):
        """..."""
        masked = self._mask(trace, from_time, upto_time)
        return masked.Y.max() > self._lower_bound

    def __call__(self, traces):
        """
        Remove traces that have a low amplitude post stimulation.
        """
        steady = traces.pre_stimulation
        response = traces.post_stimulation


class ResponseFilter(LowAmplitudeFilter):
    """..."""

    def check(self, trace, expected=None, fluctuation=0.0):
        """
        Arguments
        ------------
        fluctuations : How much could this trace fluctuate if it failed to
        respond?
        """
        print(
            "response filter expected {} fluctuation {}.".format(expected, fluctuation)
        )
        if trace.empty:
            return False
        if expected is None:
            expected = trace.Y[0]
        return trace.Y.max() - expected.Y > max(fluctuation.Y, self._lower_bound)

    def __call__(
        self,
        traces,
        t_stim=500.0,
        dt_steady=100.0,
        using_pre_stim_expected=False,
        using_pre_stim_max=False,
    ):
        """
        Arguments
        --------------
        t_stim : Time of stimulation
        dt_steady : Time duration before `t_stim` when trace features will be
        ~           considered. This assumes that the trace is in some sort
        ~           of steady state during this period.
        """
        steady = traces.mask_time_window(t_stim - dt_steady, t_stim)
        groups = steady.raw.groupby("connection")
        fluctuations = groups.std()

        if using_pre_stim_expected:
            expected = groups.mean()
        elif using_pre_stim_max:
            expected = groups.max()
        else:

            def end(trace):
                return trace.iloc[-1]

            expected = steady.map(end)

        def get_expected(connection, trial):
            if using_pre_stim_max or using_pre_stim_expected:
                return expected.loc[connection]
            return expected.loc[(connection, trial)]

        def get_fluctuation(connection, trial):
            return fluctuations.loc[connection]

        itraces = traces.iter(with_indices=True)
        return traces.accumulate(
            trace
            for (connection, trial), trace in itraces
            if self.check(
                trace,
                get_expected(connection, trial),
                get_fluctuation(connection, trial),
            )
        )
