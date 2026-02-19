"""Tests for the Dataset hierarchy and @evaluate_datasets."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bravli.bench.dataset import (
    Dataset,
    LocalDataset,
    CuratedDataset,
    GeneratedDataset,
    evaluate_datasets,
)


class TestDatasetDefinition:
    """Dataset objects can describe themselves."""

    def test_base_dataset_define(self):
        ds = Dataset(name="test", ftype="csv", description="a test")
        defn = ds.define()
        assert defn["name"] == "test"
        assert defn["ftype"] == "csv"
        assert defn["class"] == "Dataset"

    def test_local_dataset_define(self):
        ds = LocalDataset(name="local", ftype="csv", origin="/tmp/data.csv")
        defn = ds.define()
        assert defn["origin"] == "/tmp/data.csv"
        assert defn["class"] == "LocalDataset"

    def test_curated_dataset_define(self):
        ds = CuratedDataset(
            name="lit", ftype="csv",
            author="Dorkenwald", source="Nature 2024",
        )
        defn = ds.define()
        assert defn["author"] == "Dorkenwald"
        assert defn["source"] == "Nature 2024"


class TestDatasetLoadSave:
    """Datasets can round-trip through the filesystem."""

    def test_csv_round_trip(self, tmp_path):
        path = tmp_path / "neurons.csv"
        df = pd.DataFrame({"neuron_id": [1, 2, 3], "type": ["KC", "KC", "PN"]})

        ds = Dataset(name="neurons", ftype="csv")
        ds.save(df, path)
        assert path.exists()

        loaded = ds.load(path)
        assert list(loaded.columns) == ["neuron_id", "type"]
        assert len(loaded) == 3

    def test_tsv_round_trip(self, tmp_path):
        path = tmp_path / "annotations.tsv"
        df = pd.DataFrame({"root_id": [100, 200], "neuropil": ["MB", "AL"]})

        ds = Dataset(name="annotations", ftype="tsv")
        ds.save(df, path)
        loaded = ds.load(path)
        assert len(loaded) == 2

    def test_local_dataset_loads_from_origin(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)

        ds = LocalDataset(name="auto", ftype="csv", origin=path)
        val = ds.value
        assert len(val) == 2

    def test_with_data(self):
        ds = Dataset(name="inmem", ftype="csv")
        df = pd.DataFrame({"a": [1]})
        ds.with_data(df)
        assert len(ds.value) == 1


class TestGeneratedDataset:
    """Derived datasets compute from inputs."""

    def test_generate_from_inputs(self):
        src = Dataset(name="source", ftype="csv").with_data(
            pd.DataFrame({"n": [10, 20, 30]})
        )

        def double_sum(df):
            return df["n"].sum() * 2

        gen = GeneratedDataset(
            name="doubled", ftype="csv",
            inputs=[src],
            computation=double_sum,
        )
        assert gen.value == 120


class TestEvaluateDatasets:
    """The @evaluate_datasets decorator unwraps .value transparently."""

    def test_unwraps_dataset_arg(self):
        @evaluate_datasets
        def add_column(df, col_name):
            df = df.copy()
            df[col_name] = 0
            return df

        ds = Dataset(name="test", ftype="csv").with_data(
            pd.DataFrame({"x": [1, 2]})
        )
        result = add_column(ds, "new_col")
        assert "new_col" in result.columns

    def test_passes_raw_data_through(self):
        @evaluate_datasets
        def noop(x):
            return x

        assert noop(42) == 42
        assert noop("hello") == "hello"

    def test_unwraps_kwargs(self):
        @evaluate_datasets
        def merge(left, right):
            return pd.merge(left, right, on="id")

        ds_left = Dataset(name="l", ftype="csv").with_data(
            pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        )
        ds_right = Dataset(name="r", ftype="csv").with_data(
            pd.DataFrame({"id": [1, 2], "b": [30, 40]})
        )
        result = merge(ds_left, right=ds_right)
        assert "a" in result.columns and "b" in result.columns
