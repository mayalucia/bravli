"""
Simple utilities used by analysis code.
"""
import functools

def partial(method, **kwargs):
    """
    Adapt a method... and keep it's name.
    """
    get = functools.partial(method, **kwargs)
    get.__name__ = method.__name__
    return get
