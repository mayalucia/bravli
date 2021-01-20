"""
Adapter of cell methods
"""


import numpy as np
import pandas as pd
import neurom as nm

from dmt.tk.field import WithFields
from dmt.tk.collections import get_list
from neuro_dmt import terminology

X = terminology.bluebrain.cell.x
Y = terminology.bluebrain.cell.y
Z = terminology.bluebrain.cell.z
XYZ = [X, Y, Z]



class CircuitCellsAdapter(WithFields):
    """
    Adapt methods about a circuit's cells.
    """

    @staticmethod
    def _resolve_one_gid(cell):
        try:
            gid = cell.gid
        except AttributeError:
            try:
                gid = int(cell)
            except ValueError:
                raise ValueError("Unhandled cell type %s", cell)
        return gid

    def get_morphology(self, circuit, cell):
        """
        Mixin a method to resolve gid from a cell
        """
        gid = self._resolve_one_gid(cell)
        return circuit.morphologies.get(gid, transform=False)

    def get_axon_length(self, circuit, cell):
        """..."""
        return nm.get("neurite_lengths",
                      self.get_morphology(circuit, cell),
                      neurite_type=nm.AXON)[0]

    def get_cell_types(self, circuit, query):
        """
        Get cell-types in the circuit.

        Arguments
        -------------
        query :: Either an iterable of unique cell type specifiers.
        ~        Or a mapping cell_type_specifier --> value / list(value)

        Returns
        -------------
        A `pandas.DataFrame` containing all cell-types,
        each row providing values for each cell type specifier.
        """
        try:
            query_items = query.items()
        except AttributeError:
            query_with_values = {}
        else:
            query_with_values = {cell_property: value
                                 for cell_property, value in query_items
                                 if value is not None}
        def _values(variable):
            """..."""
            try:
                raw_values = query_with_values[variable]
            except KeyError:
                try:
                    get_values = getattr(self, "get_{}s".format(variable))
                except AttributeError as error:
                    raise AttributeError(
                        """
                        {} adapter does not implement a getter for cell property
                        {}
                        """).format(self.__class__.__name__, variable)
                raw_values = get_values(circuit)

            return get_list(raw_values)

        def _get_tupled_values(params):
            """..."""
            if not params: return [[]]

            head_tuples = [[(params[0], value)] for value in _values(params[0])]
            tail_tuples = _get_tupled_values(params[1:])
            return [h + t for h in head_tuples for t in tail_tuples]

        try:
            keys = query.keys()
        except AttributeError:
            keys = query
        specifiers_cell_type = tuple(keys)

        return pd.DataFrame([dict(row)
                             for row in _get_tupled_values(specifiers_cell_type)])

    def get_orientations(self, circuit, positions):
        """
        Get the orientations for voxel-positions in the circuit model atlas.

        Arguments
        -------------
        positions: np.array<N * 3>
        """
        return pd.DataFrame(circuit.orientation_field.lookup(positions)[:, :, 1],
                            columns=XYZ)

