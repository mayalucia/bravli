"""We analyze different aspects of cells's connection probabilities
."""
from collections import namedtuple
import numpy as np
import pandas as pd

from bluepy.enums import Cell, Synapse, Direction

TERM = {}


class Triplets(namendtuple("Triplets",
                          ["angle", "triangle"])):
    pass


NodeType = (int, np.int, np.int32, np.int64)


class NodeConnectivity:
    """Connectivity of a single node.
    """
    @classmethod
    def subset_reference(population, in_connections):
        """..."""
        pre_in_reference = np.in1d(in_connections[Synapse.PRE_GID], population)
        post_in_reference = np.in1d(in_connections[Synapse.POST_GID], population)
        return in_connections(np.logical_and(pre_in_reference, post_in_reference))

    def __init__(self, direction, connections, reference_population):
        """..."""
        self._direction = direction
        self._cnxns = connections
        self._reference_population = reference_population
        self._preview = (self._reference_connections
                         .set_index(Synapse.PRE_GID)[Synapse.POST_GID])
        self._reference_neighbors = self._preview.loc[gid]
        self._postview = (connections.set_index(Synapse.POST_GID)[Synapse.PRE_GID]
                          .loc[gid])

    @lazy
    def reference_connections(self):
        return self.subset_reference(self._reference_population,
                                     self._cnxns)

    @lazy
    def preview(self):
        """..."""
        return (self.reference_connections
                .set_index(Synapse.PRE_GID)[Synapse.POST_GID])
    @lazy
    def postview(self):
        """..."""
        return (self.reference_connections
                .set_index(Synapse.POST_GID)[Synapse.PRE_GID])

    def resolve_view(self, direction):
        """..."""
        if direction == Direction.AFF:
            return self.postview
        if direction == Direction.EFF:
            return preview
        raise ValueError(f"Illegal direction {direction}")

    def get_neighbors(self, at, of_node):
        """..."""
        view = self.resolve_view(at)
        try:
            return view.loc[x]
        except KeyError:
            return None
        raise RuntimeError("Python should not have executed this.")

    @staticmethod
    def ensure_series(xs, name=None):
        """..."""
        return xs if isinstance(xs, pd.Series) else pd.Series(xs, name=name)

    def count_common_neighbors(self, pre, post, directions=None):
        """Number of nodes in reference population adjacent to both pre and post
        If pre and post are node (ids), a number.
        If pre and post are groups of ids, then a collected values for each (x,y)
        where x in pre, and each y in post.
        """
        if isinstance(pre, NodeType) and isinstance(post, NodeType):
            directions = directions or (Direction.AFF, Direction.AFF)
            neighbors_pre = self.get_neighbors(at=directions[0], of_node=pre)
            neighbors_post = self.get_neighbors(at=directions[1], of_node=post)
            return np.in1d(neighors_pre, neighbors_post).sum()

        if isinstance(pre, NodeType):
            x = pre
            ys = self.ensure_series(ys, Synapse.POST_GID)
            return ys.apply(lambda y: self.count_common_neighbors(x, y, directions))

        if isinstance(post, NodeType):
            xs = self.ensure_series(xs, Synapse.PRE_GID)
            y = post
            return xs.apply(lambda x: self.count_common_neighbors(x, y, directions))

        return xs.apply(lambda x: self.count_common_neighbors(x, ys, directions))


    def count_common_neighbors(self, x, ys, direction=None):
        """..."""
        if isinstance(ys, (pd.Series, np.ndarray)):
            return ys.apply(lambda y: self.count_common_neighbors(x, y,
                                                                  direction))
        else:
            assert isinstance(ys, (int, np.int, np.int64))
            y = ys

        direction = direction or (Direction.AFF, Direction.AFF)

        return np.in1d(self.resolve_view(direction[0], x),
                       self.resovle_view(direction[1], y))
        view_0 = self.resolve_view(direction[0])
        view_1 = self.resolve_view(direction[1])
        return np.in1d(view_0.loc[x], view_1.loc[y]).sum()

        postview_x = self.postview.loc[x]
        postview_y = self.postview.loc[y]
        return np.in1d(self.postview.loc[x], self.postview.loc[y]).sum()

    def filter_connected(self, x, ys):
        """..."""
        view = (self._cnxns.set_index(Synapse.PRE_GID)[Synapse.POST_GID]
                if self._direction == Direction.EFF else
                self._cnxns.set_index(Synapse.POST_GID)[Synapse.PRE_GID])
        return ys[np.in1d(ys, view.loc[x])]

    @staticmethod
    def count_triplets(among_population_count, with_connected_nodes):
        """.."""
        number_angles = among_population_count.sum()
        number_triangles = among_population_count.loc[with_connected_nodes].sum()
        return Triplets(angles=number_angles, triangles=number_triangles)

    @staticmethod
    def correlate_common_neighbors(triplets):
        """..."""
        return triplets.triangles / triplets.angles

    def measure_cn_bias(self, among_population_counts, with_cnxns):
        """
        This method's inputs are expected to be outputs of other methods.
        It could be a staticmethod, but it's behavior and expectations are tied
        to this instance.
        among_population_counts : common neighbor counts in self._reference_population
        ~                         returned by a call to self.count_common_neighbors
        with_cnxns : 
        """
        stats = ["mean", "std"]
        expect_among_all = among_population_counts.agg("mean")
        expect_among_cnxns = among_population_counts.loc[with_cnxns].agg("mean")
        return expect_among_cnxns / expect_among_all

    def get_cn_bias(self, x, ys, direction=None):
        """..."""
        direction = direction or Direction.AFF

        cn_counts = self.count_common_neighbors(x, ys, direction)
        x_connected_ys = self.filter_connected(x, ys)

        return self.measure_cn_bias(cn_counts, x_connected_ys)

    def __call__(self, x, ys, direction=None):
        """..."""
        if not direction:
            return {d: self(x, ys, d) for d in [Direction.EFFERENT,
                                                Direction.AFFERENT]}

        cn_counts = self.count_common_neighbors(x, ys, direction)
        x_connected_ys = self.filter_connected(x, ys)

        triplets = self.count_triplets(cncounts, x_connected_ys)
        cn_bias = self.measure_cn_bias(cn_counts, x_connected_ys)
        return {"triplets": triplets, "cn_bias": cn_bias}


class ConnectionProbability:
    def __init__(self, connections,
                 get_annotations=None,
                 reference_population=None):
        """Initialize with a pandas DataFrame of connections.
        Annotate these connections with a call-back:
        ~  For example, this method can return mtypes for each of the
        ~  source (pre-synaptic) and target (post-synaptic) nodes.
        ~  So this call-back method can be used to annotate pathways to
        ~  connections.
        Provide a reference population to compute connectivity biases.
        """
        self._cnxns = connections
        self._anntns = get_annotations(connections)
        self._refpop = reference_population

        raise NotImplementedError("Work in Progress.")
