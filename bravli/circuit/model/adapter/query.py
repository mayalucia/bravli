"""
BBP circuits are atlas basesd.
For an atlas based circuit, and we use the atlas.
Here we develop tools to process queries for an atlas based circuit.
"""
from dmt.tk.collections.data import make_hashable
from dmt.tk.field import Field, lazyfield, WithFields

class QueryDB(WithFields):
    """
    Cache data associated with a circuit query.
    """
    method_to_memoize = Field("A callable to get values to cache.")

    def __init__(self, method_to_memoize):
        """..."""
        super().__init__(method_to_memoize=method_to_memoize)

    @lazyfield
    def store(self): return {}

    @classmethod
    def _hashable(cls, query_dict):
        hashable_items = ((key, make_hashable(value))
                          for key, value in query_dict.items()
                          if value is not None)
        return tuple(sorted(hashable_items,
                            key=lambda key_value: key_value[0]))

    @classmethod
    def _hashed(cls, query_dict):
        """..."""
        return hash(cls._hashable(query_dict))

    def _cached(self, circuit_model, query_dict):
        """
        Get a cache store of data for `(circuit_model, query_dict)`.
        """
        if circuit_model not in self.store:
            self.store[circuit_model] = {}

        store = self.store[circuit_model]
        hash_query = self._hashable(query_dict)

        if hash_query not in store:
            store[hash_query] = self.method_to_memoize(circuit_model,
                                                       query_dict)
        return store[hash_query]

    def __call__(self, circuit_model, query_dict):
        """..."""
        return self._cached(circuit_model, query_dict)


class SpatialQueryData(WithFields):
    """
    Encapsulate data to answer spatial queries for an atlas based circuit.
    """
    query = Field("Mapping of query parameters to their values")
    ids = Field("Ids of voxels that passed the spatial query filter")
    positions = Field("Positions of the voxels with ids in self.ids")
    cell_gids = Field(
        """
        `pandas.Series` that provides the gids of all the cells in voxels
        that passed the spatial query filter, indexed by their corresponding
        voxel_ids.
        """,
        __required__=False)
    @lazyfield
    def empty(self):
        """..."""
        return self.positions.empty
