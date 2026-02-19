"""Composition analysis: cell type counts, neurotransmitter profiles.

Every function here is a pure transformation: DataFrame in, summary out.
The @evaluate_datasets decorator allows passing Dataset objects directly.

Heritage: these functions replace the complex atlas-densities pipeline
from BBP. There, composition was /inferred/ from sparse ISH data via
constrained optimization. Here, it is /counted/ from complete annotations.
The shift from inference to counting changes everything — except the
questions we want to answer.
"""

import pandas as pd
import numpy as np

from bravli.bench.dataset import evaluate_datasets


@evaluate_datasets
def count_by(annotations, column):
    """Count neurons grouped by a column.

    Parameters
    ----------
    annotations : pd.DataFrame
        Neuron annotation table.
    column : str
        Column to group by (e.g., 'super_class', 'cell_class', 'cell_type').

    Returns
    -------
    pd.Series
        Neuron counts indexed by the grouping column, sorted descending.
    """
    return (annotations
            .groupby(column)
            .size()
            .sort_values(ascending=False)
            .rename("neuron_count"))


@evaluate_datasets
def cell_type_distribution(annotations, grouping="cell_type", normalize=False):
    """Distribution of cell types within an annotation set.

    Parameters
    ----------
    annotations : pd.DataFrame
        Neuron annotations (typically a subset for one region/class).
    grouping : str
        Column to use for cell type identity.
    normalize : bool
        If True, return proportions instead of counts.

    Returns
    -------
    pd.DataFrame
        Columns: neuron_count (and proportion if normalize=True).
    """
    counts = count_by.__wrapped__(annotations, grouping)
    result = counts.to_frame()
    if normalize:
        result["proportion"] = result["neuron_count"] / result["neuron_count"].sum()
    return result


@evaluate_datasets
def neurotransmitter_profile(annotations, nt_column="top_nt",
                             conf_column="top_nt_conf", min_confidence=0.0):
    """Neurotransmitter composition of a set of neurons.

    Parameters
    ----------
    annotations : pd.DataFrame
        Neuron annotations.
    nt_column : str
        Column with neurotransmitter prediction.
    conf_column : str
        Column with prediction confidence.
    min_confidence : float
        Minimum confidence to include a prediction (0.0 = include all).

    Returns
    -------
    pd.DataFrame
        Columns: neuron_count, proportion, mean_confidence.
    """
    df = annotations
    if conf_column in df.columns and min_confidence > 0:
        df = df[df[conf_column] >= min_confidence]

    counts = df.groupby(nt_column).size().rename("neuron_count")
    total = counts.sum()
    result = counts.to_frame()
    result["proportion"] = result["neuron_count"] / total

    if conf_column in df.columns:
        mean_conf = df.groupby(nt_column)[conf_column].mean()
        result["mean_confidence"] = mean_conf

    return result.sort_values("neuron_count", ascending=False)


@evaluate_datasets
def compare_divisions(annotations, column, division_column="super_class"):
    """Compare a property across anatomical divisions.

    Parameters
    ----------
    annotations : pd.DataFrame
        Full neuron annotation table.
    column : str
        Property to compare (e.g., 'top_nt', 'cell_class').
    division_column : str
        Column defining the divisions (default: 'super_class').

    Returns
    -------
    pd.DataFrame
        Cross-tabulation: divisions as columns, property values as rows.
        Values are proportions within each division.
    """
    ct = pd.crosstab(
        annotations[column],
        annotations[division_column],
        normalize="columns",
    )
    return ct


@evaluate_datasets
def top_types(annotations, n=10, grouping="cell_type"):
    """The N most abundant cell types.

    Parameters
    ----------
    annotations : pd.DataFrame
        Neuron annotations.
    n : int
        How many types to return.
    grouping : str
        Column defining cell type identity.

    Returns
    -------
    pd.DataFrame
        Top N types with neuron_count and proportion.
    """
    dist = cell_type_distribution.__wrapped__(annotations, grouping=grouping,
                                              normalize=True)
    return dist.head(n)
