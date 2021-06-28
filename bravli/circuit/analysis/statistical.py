"""
Statistical things
"""
import numpy as np
import pandas as pd

def normalized(values):
    s = np.sum(values)
    try:
        values = values / s
    except TypeError:
        values = np.array([v / s for v in values])
    return pd.Series(values)

def get_categorical(index, name=""):
    """
    A categorical index from a sequence.

    TODO: Worry about ordered or not.
    """
    if isinstance(index, pd.CategoricalIndex):
        return index

    if isinstance(index, pd.MultiIndex):
        raise NotImplementedError(
                """
                There does not appear to be a unique way to convert
                a multi-index into a categorical-index.
                How would you do it?
                Let us know.
                """)

    if isinstance(index, pd.Index):
        return get_categorical(index.values, name=index.name)

    return pd.CategoricalIndex(index, name=name)

def is_distribution(xs):
    """
    Define a distribution.
    """
    return(isinstance(xs, pd.Series) and
           np.isclose(xs.sum(), 1.) and
           isinstance(xs.index, pd.CategoricalIndex))

def distribution(values, categories=None):
    """
    Define a finite discrete space distribution.

    Arguments
    --------------
    values : At least a sequence of numbers.
    """
    if is_distribution(values):
        return values
    if not isinstance(values, pd.Series):
        return distribution(values=normalized(values),
                            categories=categories)
    if categories is None:
        return distribution(values=values,
                            categories=values.index.values)

    assert len(categories) == len(values), "index & values lengths differ"

    return pd.Series(normalized(values), name="pdf",
                     index=get_categorical(categories))

def uniform(xs):
    """
    Make a uniform distribution with the same state space as a given one.

    Arguments
    ----------
    xs : A distribution-like sequence
    """
    return distribution(xs).apply(lambda _: 1.)

def entropy(xs, log=np.log):
    """
    Entropy of a distribution.

    Arguments
    ----------
    xs : That can be converted to a distribution (see method `distribution` above).
    log : ...
    """
    pdf = distribution(xs)
    return -np.sum(pdf * log(pdf))

def max_entropy(xs, log=np.log):
    """
    Maximum possible entropy for the state-space of a distribution.

    Arguments
    ----------
    xs : That can be converted to a distribution (see method `distribution` above).
    log : ...
    """
    return entropy(uniform(distribution(xs)), log)

def information(xs, log=np.log):
    """
    Information tells us the structure in a distribution,
    compared to a uniform distribution over the same categories.

    Arguments
    -----------
    distribution : `pandas.Series`,
    ~               indexed by a categorical variable (`pandas.Categorical`)
    base : Log base to compute with.
    """
    pdf = distribution(xs)
    return max_entropy(pdf, log) - entropy(pdf, log)

def divergence(p, q, kind="Kullback-Leibler", log=np.log):
    assert kind == "Kullback-Leibler", "No other has yet been implemented."
    return np.sum(p * log(p/q))

def gini(pdf):
    """
    pdf: A distribution
    """
    n = pdf.shape[0]
    return np.sum([np.sum(np.abs(x - pdf)) for x in pdf])/(2.*n*n*pdf.mean())
