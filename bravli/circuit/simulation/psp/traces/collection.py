"""
Handle traces produced by simulation.s
"""
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import h5py
import yaml
from lazy import lazy
import numpy as np
import pandas as pd
import seaborn as sn

#pylint : disable=relative-beyond-top-level
from .. import Pathway, Protocol, Config
# pylint : disable=relative-beyond-top-level
from .filter import (StochasticConnectionFilter,
                     ExtremeVoltageFilter,
                     LargeAmplitudeFilter,
                     LowAmplitudeFilter,
                     ResponseFilter)


class VoltageClamp:
    """..."""

    def __init__(
        self,
        hold_V,
        fluctuation=0.01,
        baseline=None,
        t_start=0.0,
        dt_relax=450.0,
        t_stim=500.0,
    ):
        """
        Arguments
        ------------
        hold_V : Voltage at which the voltage has been clamped.
        t_relax : Time at which the soma voltage should relax...
        """
        self._hold_V = hold_V
        self._fluctuation = fluctuation
        self._baseline = baseline
        self._t_start = t_start
        self._dt_relax = dt_relax
        self._t_stim = t_stim

    @lazy
    def hold_V(self):
        """
        TODO: What?
        """
        return self._hold_V

    @lazy
    def dt_relax(self):
        """
        TODO: What?
        """
        return self._dt_relax

    def get_baseline(self, traces, by=None, statistics="mean"):
        """..."""
        if self._baseline:
            return self._baseline

        def baseline(trace):
            prestim = mask_time_window(
                trace, self._t_start + self._dt_relax, self._t_stim
            )
            return prestim.voltage.agg("mean")

        if isinstance(traces, pd.DataFrame):
            return baseline(traces)

        baselines = traces.map(baseline)

        if not by:
            return baselines

        return baselines.groupby(by).agg(statistics)

    def check(self, trace):
        tmin = self._t_start + self._dt_relax
        tmax = self._t_stim
        in_time_window = np.logical_and(tmin <= trace.time, trace.time < tmax)
        return np.abs(trace.Y[in_time_window].max()) < self._fluctuation

    def __call__(self, trace):
        """
        Clamp a trace, returning part of the trace that is expected to be
        clamped.
        We do not quality check the trace to determine if it was clamped.
        """
        time_window = np.logical_and(
            self._t_start + self._dt_relax <= trace.time, trace.time < self._t_stim
        )
        return trace[time_window]


class GroupedDataFrame:
    """
    Accumulate traces stored in `pandas.DataFrame` arranged using a
    `pandas.MultiIndex<connection, trial>`...
    """

    def __init__(
        self, *args, index=["connection", "trial"], return_index=False, **kwargs
    ):
        """..."""
        self._index = index
        self._return_index = return_index
        super().__init__(*args, **kwargs)

    def iter(self, traces):
        """
        Extract individual traces from a `pandas.DataFrame`
        """
        return [
            ((index, trace) if self._return_index else trace)
            for index, trace in traces.groupby(self._index)
        ]

    def accumulate(self, generator_traces):
        """..."""
        return pd.concat([t for t in generator_traces])


class Kinetics(Enum):
    DETERMINISTIC = 0
    STOCHASTIC = 1


class TraceKinetics:
    """
    Handle stochasticity of a trace-collection.

    TODO: Discuss what is stochastic.
    """

    def __init__(self, traces, from_time=0.0, upto_time=np.infty):
        """..."""
        self._traces = traces

        time_means = traces.get_time_summary(
            "voltage", from_time, upto_time, statistics="mean"
        )
        self._variance = time_means.groupby("connection").agg("var")

    @lazy
    def kinetics(self):
        K = pd.Series(
            pd.Categorical(
                self._variance.shape[0] * ["stochastic"],
                categories=["stochastic", "deterministic"],
            ),
            index=self._variance.index,
            name="kinetics",
        )
        is_deterministic = np.isclose(self._variance, 0.0)
        K[is_deterministic] = "deterministic"
        return K
        # return pd.DataFrame({"variance": self._variance, "kinetics": K})

    @lazy
    def stochastic(self):
        """..."""
        stochastic = self.kinetics[self.kinetics == "stochastic"]
        return self._traces.with_data(self._traces.raw.loc[stochastic.index])

    @lazy
    def stochastic_fraction(self):
        """..."""
        is_stochastic = self.kinetics == "stochastic"
        return is_stochastic.mean()

    @lazy
    def deterministic(self):
        """..."""
        deterministic = self.kinetics[self.kinetics == "deterministic"]
        return self._traces.with_data(self._traces.raw.loc[deterministic.index])

    @lazy
    def deterministic_fraction(self):
        """..."""
        is_deterministic = self.kinetics == "deterministic"
        return is_deterministic.mean()


class TraceCollection:
    """
    Load and analyze PSP traces.
    """

    mtype_prefixes = ["L1", "L2", "L23", "L3", "L4", "L5", "L6"]
    variables = ["time", "voltage", "Y"]

    @classmethod
    def get_index(values):
        """..."""
        return pd.MultiIndex.from_tuples(values, names=self._index)

    @staticmethod
    def read_synapse_class(path, mtype):
        """
        Read the file `dirpath / mtypes.tsv` that provides mtype synapse class.
        """
        synapse_class = pd.read_csv(path, sep=u"\t").set_index("mtype").synapse_class
        return synapse_class[mtype]

    def __init__(
        self,
        data,
        pathway,
        voltage_clamp,
        dt_record,
        t_stim,
        t_stop,
        dt_response=None,
        filter_spiking=ExtremeVoltageFilter(-30.0),
        upper_bound_spike_prob=0.5,
        filter_large_amplitudes=LargeAmplitudeFilter(5.0),
        filter_low_amplitudes=LowAmplitudeFilter(0.01),
        filter_responsive=ResponseFilter(0.01),
    ):
        """..."""
        self._pathway = pathway
        self._voltage_clamp = voltage_clamp
        self._dt_record = dt_record
        self._t_stim = t_stim
        self._dt_response = (t_stop - t_stim) if dt_response is None else dt_response
        self._t_stop = t_stop
        self._filter_spiking = filter_spiking
        self._upper_bound_spike_prob = (upper_bound_spike_prob,)
        self._filter_large_amplitudes = filter_large_amplitudes
        self._filter_low_amplitudes = filter_low_amplitudes
        self._filter_responsive = filter_responsive
        self._amplitude_fluctuation = filter_low_amplitudes._lower_bound

        self._index = data.index.names
        self._data = data
        if "Y" not in data:

            def get_y(trace):
                return trace.voltage.to_numpy(
                    np.float
                ) - self._voltage_clamp.get_baseline(trace)

            self._data["Y"] = np.concatenate([get_y(t) for t in self])

    @lazy
    def t_stim(self):
        return self._t_stim

    @lazy
    def t_steady(self):
        return self._voltage_clamp.dt_relax

    @lazy
    def t_response(self):
        """..."""
        return self._t_stim + self._dt_response

    @lazy
    def contains_mean_traces(self):
        return self._index == ["connection"]

    @lazy
    def raw(self):
        """..."""
        return self._data

    def _filter_stochastic(self):
        """..."""
        return StochasticConnectionFilter(
            from_time=self._voltage_clamp._dt_relax, upto_time=self._t_stim
        )

    @lazy
    def with_kinetics(self):
        """..."""
        return TraceKinetics(self, self._voltage_clamp._dt_relax, self._t_stim)

    @lazy
    def pathway(self):
        return self._pathway

    @lazy
    def connections(self):
        return np.unique(self.raw.index.get_level_values("connection"))

    @lazy
    def indices(self):
        return np.unique(self.raw.index.values)

    def with_data(self, data):
        """
        Replace data...
        """
        return self.__class__(
            data,  # .assign(Y=self.get_y(data)),
            self._pathway,
            self._voltage_clamp,
            self._dt_record,
            self._t_stim,
            self._t_stop,
            filter_spiking=self._filter_spiking,
            upper_bound_spike_prob=self._upper_bound_spike_prob,
            filter_large_amplitudes=self._filter_large_amplitudes,
            filter_low_amplitudes=self._filter_low_amplitudes,
        )

    def get_tsteps(self, trace):
        """..."""
        return np.array(np.floor(trace.time / self._dt_record), dtype=np.int)

    def iter(self, over_mean_traces=False, by_connection=False, with_indices=False):
        """
        Arguments
        -----------
        over_mean_traces : each yielded element should be the mean trace of a
        ~                  connection.
        by_connection : each yielded element should be the traces of a
        ~               connection.
        """
        if self.contains_mean_traces:
            groups = self.raw.groupby("connection")
        else:
            if over_mean_traces:

                def get_mean(connection, traces):
                    return (
                        traces.assign(tstep=self.get_tsteps)
                        .groupby("tstep")
                        .agg("mean")
                        .assign(connection=connection)
                        .set_index("connection")
                    )

                groups = (
                    (connection, get_mean(connection, traces))
                    for connection, traces in self.raw.groupby("connection")
                )
            elif by_connection:
                groups = self._data.groupby("connection")
            else:
                groups = self._data.groupby(self._index)

        for indices, trace in groups:
            yield (indices, trace) if with_indices else trace

    def sample(self, connections=None, n_connections=1, n_trials=1, with_index=False):
        """..."""
        if not connections:
            connections = self.connections
        chosen_connections = np.random.choice(connections, n_connections, replace=False)
        traces_connections = self.raw.loc[chosen]
        if self.contains_mean_traces or n_trials is None:
            return traces_connections
        trials = np.unique(traces_chosen.index.values)
        chosen_trials = np.random.choice(trials, n_trials)
        return traces_chosen.loc[chosen_trials]

    def get_iterator(self, traces, **kwargs):
        """..."""
        if isinstance(traces, self.__class__):
            return self.iter(**kwargs)

        assert isinstance(traces, pd.DataFrame)

        return self.with_data(traces.iter(**kwargs))

    def __iter__(self):
        """..."""
        return self.iter()

    def accumulate(self, traces):
        """..."""
        return self.with_data(pd.concat(list(traces)))
        # return pd.concat(list(traces))

    def map(self, function, over_mean_traces=False, by_connection=False, variable=None):
        """..."""
        if not variable:
            variable = function.__name__

        only_connection = self.contains_mean_traces or over_mean_traces or by_connection

        def get(indices, traces):
            value = function(traces)
            index_series = (
                pd.Series({"connection": indices})
                if only_connection
                else pd.Series({"connection": indices[0], "trial": indices[1]})
            )

            value_series = (
                pd.Series(value)
                if isinstance(value, (Mapping, pd.Series))
                else pd.Series({variable: value})
            )
            return index_series.append(value_series)

        values = [
            get(indices, trace)
            for indices, trace in self.iter(
                over_mean_traces=over_mean_traces,
                by_connection=by_connection,
                with_indices=True,
            )
        ]
        dataframe = pd.DataFrame(values).set_index(
            "connection" if only_connection else self._index
        )
        return (
            dataframe if len(dataframe.columns) > 1 else dataframe[dataframe.columns[0]]
        )

    def get_baseline(
        self,
        over_mean_traces=False,
        by_connection=False,
        use_clamp=False,
        statistics="mean",
    ):
        """..."""
        if use_clamp:
            return self._voltage_clamp.hold_V

        def get(traces, by):
            return self._voltage_clamp.get_baseline(traces, by, statistics)

        if over_mean_traces:
            return get(traces=self.mean, by=None)

        if by_connection:
            return get(traces=self, by="connection")

        return get(traces=self, by=None)

    def get_count(self, by_connection=False):
        """
        Count traces, all or by connection.
        """

        def count(trace):
            return 1

        counts = self.map(count)
        if not by_connection:
            return counts.sum()
        return counts.groupby("connection").sum()

    @lazy
    def count(self):
        """..."""
        return self.get_count()

    @lazy
    def count_by_connection(self):
        """..."""
        return self.get_count(by_connection=True)

    @lazy
    def not_spiking(self):
        """..."""
        return self._filter_spiking(self)

    @lazy
    def spiking(self):
        """..."""
        return self._filter_spiking.get_failing(self)

    @lazy
    def spiking_probability(self):
        """..."""
        return (
            np.logical_not(self._filter_spiking.get_mask(self))
            .groupby("connection")
            .agg("mean")
        )

    def get_spiking_probability(self, connection=None):
        """..."""
        return self.spiking_probability.loc[connection]

    def get_fluctuation_size(self, connection):
        """
        How big are voltage fluctuations in a connection's traces?
        Fluctuations can be hard to determine in the presence of a signal.
        Thus we use the voltage-clamped part of a trace just before stimulation
        onset.

        Arguments
        ------------
        trace : A single trace for a `(connection, trial)`
        """
        raise NotImplementedError("Is it even required?")

    def get_time_summary(
        self,
        variable=None,
        from_time=0.0,
        upto_time=np.infty,
        statistics=["mean", "var"],
    ):
        """..."""
        in_time = self.mask_time_window(from_time, upto_time)
        return (
            in_time.raw[variable or ["voltage", "Y"]]
            .groupby(self._index)
            .agg(statistics)
        )

    @lazy
    def clamped(self):
        """Traces that could be clamped."""
        return self.accumulate(
            trace for trace in self if self._voltage_clamp.check(trace)
        )

    @lazy
    def unclamped(self):
        """Traces that could not be clamped."""
        return self.accumulate(
            trace for trace in self if not self._voltage_clamp.check(trace)
        )

    @lazy
    def without_spikes(self):
        """..."""
        p = self.spiking_probability
        connections = p.index[p < 0.5]
        filtered = self._filter_spiking(self)
        return self.with_data(filtered.raw.loc[connections].dropna())

    def mask_time_window(self, time_from, time_upto):
        """..."""
        time_window = np.logical_and(
            time_from <= self.raw.time, self.raw.time < time_upto
        )
        return self.with_data(self.raw[time_window])

    @lazy
    def pre_stimulation(self):
        """..."""
        time_to_steady = self.tsteady  # self._voltage_clamp._t_relax
        time_of_stimulation = self._t_stim
        masked = self.mask_time_window(
            from_time=time_to_steady, upto_time=time_of_stimulation
        )
        time_since_steady = masked.raw.time - time_to_steady
        return self.with_data(masked.raw.assign(time=time_since_steady))

    @lazy
    def post_stimulation(self):
        """..."""
        time_of_stimulation = self._t_stim
        time_response_end = time_of_stimulation + self._dt_response
        masked = self.mask_time_window(
            from_time=time_of_stimulation, upto_time=time_response_end
        )
        time_since_stimulation = masked.raw.time - time_of_stimulation
        return self.with_data(masked.raw.assign(time=time_since_stimulation))

    @lazy
    def filtered(self):
        """..."""
        if not self._filter_large_amplitudes:
            return self.without_spikes
        return self._filter_large_amplitudes(self.without_spikes)

    @lazy
    def responsive(self):
        """..."""
        return self._filter_responsive(
            self.filtered, t_stim=self._t_stim, dt_steady=self._voltage_clamp.dt_relax
        )

    @lazy
    def sign(self):
        return 1 if self._pathway.synapse_class == "EXC" else -1

    def get_mask_local_peaks(self, trace):
        """All local peaks."""
        n = trace.shape[0]
        a = trace.amplitude.to_numpy(np.float)

        a_left = np.concatenate([[a[0]], a[:-1]])
        a_right = np.concatenate([a[1:], [a[-1]]])
        peaked = np.logical_and(a_left < a, a > a_right)
        did_not_fluctate = a > self._amplitude_fluctuation
        return np.logical_and(peaked, did_not_fluctate)

    def get_peaks(self, from_time=None, upto_time=None, count=False):
        """..."""
        if from_time is None:
            from_time = 0.0
        if upto_time is None:
            upto_time = self.raw.time.max()

        def get(trace):
            time_window = np.logical_and(
                from_time <= trace.time, trace.time < upto_time
            )
            trace_in_time = trace[time_window]
            mask_peaks = self.get_mask_local_peaks(trace_in_time)
            n_peaks = np.sum(mask_peaks)
            if count:
                return pd.Series({"number": n_peaks})
            return trace_in_time[mask_peaks]

        groups = self.raw.groupby(self._index)
        if count:
            return pd.DataFrame(
                [
                    get(t).append(pd.Series({"connection": c, "trial": i}))
                    for (c, i), t in groups
                ]
            ).set_index(self._index)

        def _get(c, i, t):
            d = get(t)
            if not d.empty:
                return d
            index = pd.MultiIndex.from_tuples([(c, i)], names=self._index)
            return pd.DataFrame(
                [[np.nan, np.nan, np.nan]], columns=d.columns, index=index
            )

        return pd.concat([_get(c, i, t) for (c, i), t in groups])

    def filter_peaks(self, n=1):
        """
        TODO: refactor
        """
        traces = self.raw
        peaks = self.get_peaks(from_time=self._time_stimulation, count=True)

        if isinstance(n, (np.int, np.float)):
            filtered_index = peaks.index[peaks.number == n].values
        elif callable(n):
            filtered_index = peaks.index[n(peaks.number.to_numpy(np.int))].values
        else:
            raise NotImplementedError("Unhandled argument type n: {}".format(type(n)))

        _traces = traces.reset_index().set_index(["connection", "trial"])
        return self.with_data(_traces.loc[filtered_index])

    def get_mean_traces(self, measurement=None, variable=None, raw=True):
        """..."""
        mean_traces = (
            self.raw.assign(tstep=self.get_tsteps)
            .groupby(["connection", "tstep"])
            .agg("mean")
            .droplevel("tstep")
        )
        if not measurement:
            return mean_traces if raw else self.with_data(mean_traces)

        values = mean_traces.groupby("connection").apply(measurement)
        return values if not variable else values.rename(variable)

    @lazy
    def mean(self):
        """..."""
        return self.get_mean_traces(raw=False)

    def get_prestim(self, variable="Y", statistics="mean", use="raw"):
        """..."""
        traces = getattr(self, use)
        prestim = traces[traces.time < self._t_stim]
        raise NotImplementedError

    def plot_traces(
        self,
        variable="voltage",
        from_time=None,
        upto_time=None,
        connections=None,
        by_connection=False,
        by_trial=False,
        figsize=(15, 12),
        output_path=Path.cwd(),
        label=None,
        baseline=None,
        tstim=None,
    ):
        """..."""
        from_time = from_time if from_time else 0.0
        upto_time = upto_time if upto_time else self.raw.time.max()
        traces = self.raw if connections is None else self.raw.loc[connections]
        traces_in_time = traces[
            np.logical_and(from_time <= traces.time, traces.time < upto_time)
        ]
        if not label:
            label = random.randint(0, 1000000)

        def save(graphic, filename):
            pathway = output_path / self.pathway.label
            pathway.mkdir(exist_ok=True)
            path = pathway / "{}-{}-{}.png".format(variable, filename, label)
            graphic.figure.savefig(path)
            return path

        def plot_baseline():
            if baseline:
                plt.plot(
                    [from_time, upto_time], [baseline, baseline], "k-", label="baseline"
                )

        def plot_tstim(from_y, to_y):
            if tstim:
                plt.plot([tstim, tstim], [from_y, to_y], label="t_stim")

        if not by_trial:
            figure = plt.figure(figsize=figsize)

            def tstep(t):
                return np.array(np.round(t.time / self._dt_record), dtype=np.int)

            mean_traces = (
                traces_in_time.assign(tstep=tstep)
                .groupby(["connection", "tstep"])
                .agg("mean")
            )
            print(mean_traces.head())
            graphic = sn.lineplot(
                x="time",
                y=variable,
                hue="connection" if by_connection else None,
                data=mean_traces.reset_index(),
            )
            plot_baseline()
            value = mean_traces[variable]
            plot_tstim(value.min(), value.max())
            plt.title(self.pathway.label)
            filename = "{}-n{}".format("mean_traces", len(connections))
            return save(graphic, filename)

        graphics = {}
        if connections is None:
            connections = np.unique(self.raw.index.get_level_values("connection"))
        for connection in connections:
            figure = plt.figure(figsize=figsize)
            data = traces_in_time.loc[connection]
            graphic = sn.lineplot(
                x="time", y=variable, hue="trial", data=data.reset_index()
            )
            plot_baseline()
            value = data[variable]
            plot_tstim(value.min(), value.max())
            plt.title("{}[{}]".format(self.pathway.label, connection))
            graphics[connection] = save(graphic, connection)
            plt.legend()
        return graphics

    def plot_densities(
        self, quantity, connections=None, by_connection=False, figsize=(15, 12)
    ):
        """..."""
        figure = plt.figure(figsize=figsize)
        raise NotImplementedError

    @classmethod
    def load_config(cls, path_config, path_synapse_classes):
        """..."""
        with open(path_config, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        print(config_dict["pathway"])
        print(config_dict["protocol"])

        pathway = Pathway(**config_dict["pathway"])
        pathway.synapse_class = cls.read_synapse_class(
            path_synapse_classes, pathway.pre
        )
        protocol = Protocol(**config_dict["protocol"])

        return Config(pathway=pathway, protocol=protocol)

    @classmethod
    def load_traces(
        cls,
        config,
        dirpath,
        suffix="traces.h5",
        key_traces="traces",
        key_trials="trials",
        columns=["voltage", "time"],
    ):

        """..."""
        path = dirpath / "{}.{}".format(config.pathway.label, suffix)

        with h5py.File(path, "r") as hdf5:
            raw = hdf5[key_traces]

            def get(connection):
                traces = raw[connection][key_trials]
                trials = range(traces.shape[0])

                def get(trial_traces):
                    return pd.DataFrame(trial_traces.transpose(), columns=columns)

                return pd.concat([get(t) for t in traces], keys=trials, names=["trial"])

            connections = list(raw.keys())
            return pd.concat(
                [get(c) for c in connections], keys=connections, names=["connection"]
            ).droplevel(None)

    @classmethod
    def get_voltage_clamp(cls, protocol, t_relax=450.0):
        """..."""
        return VoltageClamp(protocol.hold_V, dt_relax=t_relax, t_stim=protocol.t_stim)
        # baseline=protocol.hold_V)

    @classmethod
    def load(
        cls,
        filepath_config,
        filepath_synapse_classes,
        dirpath_traces,
        suffix="traces.h5",
        key_traces="traces",
        key_trials="trials",
        columns=["voltage", "time"],
        **kwargs
    ):
        """..."""
        config = cls.load_config(filepath_config, filepath_synapse_classes)
        pathway = config.pathway
        protocol = config.protocol
        voltage_clamp = cls.get_voltage_clamp(protocol)
        traces = cls.load_traces(
            config,
            dirpath_traces,
            suffix=suffix,
            key_traces=key_traces,
            key_trials=key_trials,
            columns=columns,
        )
        return cls(
            data=traces,
            pathway=pathway,
            voltage_clamp=voltage_clamp,
            dt_record=protocol.record_dt,
            t_stim=protocol.t_stim,
            t_stop=protocol.t_stop,
        )
