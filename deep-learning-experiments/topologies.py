import math
from typing import Any, Dict, Iterable

import networkx
import torch


class Topology:
    num_workers: int

    def __init__(self, num_workers):
        self.num_workers = num_workers

    def neighbors(self, worker: int) -> Iterable[int]:
        raise NotImplementedError()

    def degree(self, worker: int) -> int:
        return len(self.neighbors(worker))

    @property
    def workers(self) -> Iterable[int]:
        return range(self.num_workers)

    @property
    def max_degree(self) -> int:
        return max([self.degree(w) for w in self.workers])

    def gossip_matrix(self, weight=None) -> torch.Tensor:
        m = torch.zeros([self.num_workers, self.num_workers])
        for worker in self.workers:
            for neighbor in self.neighbors(worker):
                max_degree = max(self.degree(worker), self.degree(neighbor))
                m[worker, neighbor] = 1 / (max_degree + 1) if weight is None else weight
            # self weight
            m[worker, worker] = 1 - m[worker, :].sum()

        return m

    def to_networkx(self) -> networkx.Graph:
        g = networkx.Graph()
        g.add_nodes_from(range(self.num_workers))
        for worker in range(self.num_workers):
            g.add_edges_from(
                [(worker, neighbor) for neighbor in self.neighbors(worker)]
            )
        return g

    @property
    def max_delay(self):
        g = self.to_networkx()
        distances = dict(networkx.all_pairs_shortest_path_length(g))
        return max(distances[i][j] for i in g.nodes for j in g.nodes)


def configure_topology(num_workers, topology, **kwargs) -> Topology:
    if topology == "ring":
        return RingTopology(num_workers=num_workers)
    elif topology == "chain":
        return ChainTopology(num_workers=num_workers)
    elif topology == "3-tree":
        return TreeTopology(num_workers=num_workers, max_degree=3)
    elif topology == "binary-tree":
        return BinaryTreeTopology(num_workers=num_workers)
    elif topology == "star":
        return StarTopology(num_workers=num_workers)
    elif topology == "torus":
        side1 = kwargs["torus_side"]
        side2 = num_workers // side1
        assert side1 * side2 == num_workers
        return TorusTopology(side1, side2)
    elif topology == "artificial-adversarial":
        return ArtificialAdversarialTopology(
            num_workers=num_workers, rho=kwargs["spectral_gap"]
        )
    elif topology == "artificial-nice":
        return ArtificialNiceTopology(
            num_workers=num_workers, rho=kwargs["spectral_gap"]
        )
    elif topology == "double-binary-trees":
        return [
            BinaryTreeTopology(num_workers=num_workers),
            BinaryTreeTopology(num_workers=num_workers, reverse=True),
        ]
    elif topology == "fully-connected":
        return FullyConnectedTopology(num_workers=num_workers)
    elif topology == "social-network":
        topology = SocialNetworkTopology()
        assert len(topology) == num_workers
        return topology
    elif topology == "social-network-tree":
        topology = SocialNetworkTreeTopology(0)
        assert len(topology) == num_workers
        return topology
    else:
        raise ValueError("Unknown topology {}".format(topology))


class FullyConnectedTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [j for j in range(n) if j != i]


class TwoCliquesTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        if i < n // 2:
            neighbors = [j for j in range(0, n // 2) if j != i]
        else:
            neighbors = [j for j in range(n // 2, n) if j != i]

        if i == 0:
            neighbors.append(n // 2)
        elif i == n // 2:
            neighbors.append(0)

        return neighbors


class DisconnectedTopology(Topology):
    def neighbors(self, worker):
        return []


class MixTopology(FullyConnectedTopology):
    """
    Symmetric doubly-stochastic gossip matrix with all \lambda_{2...} equal.
    """

    def __init__(self, num_workers, spectral_gap):
        super().__init__(num_workers)
        self.spectral_gap = spectral_gap

        ii = torch.eye(num_workers)
        ee = torch.ones_like(ii) / num_workers
        self.W = spectral_gap * ee + (1 - spectral_gap) * ii

    def gossip_matrix(self):
        return self.W


class StarTopology(Topology):
    def neighbors(self, worker):
        i = worker
        if i == 0:
            n = self.num_workers
            return [j for j in range(n) if j != i]
        else:
            return [0]


class ChainTopology(Topology):
    def neighbors(self, worker):
        if worker < 1:
            return [1]
        elif worker >= self.num_workers - 1:
            return [worker - 1]
        else:
            return [worker - 1, worker + 1]


class RingTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        if n == 1:
            return []
        elif n == 2:
            return [(i + 1) % n]
        else:
            return [(i - 1) % n, (i + 1) % n]


class UnidirectionalRingTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [(i + 1) % n]


class HyperCubeTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers

        d = int(math.log2(n))
        assert 2**d == n

        return [i ^ (2**j) for j in range(0, d)]


class TorusTopology(Topology):
    def __init__(self, n, m):
        self.num_workers = n * m
        self.n = n
        self.m = m

    def neighbors(self, worker):
        # i = col + row * m
        i = worker
        col = i % self.m
        row = i // self.m

        idx = lambda row, col: (col + row * self.m) % self.num_workers

        return [
            idx(row - 1, col),
            idx(row + 1, col),
            idx(row, col - 1),
            idx(row, col + 1),
        ]


class TreeTopology(Topology):
    """A tree that divides nodes such that nodes have the same degree if they are not (close to) leaves"""

    num_workers: int
    max_degree: int

    def __init__(self, num_workers, max_degree):
        super().__init__(num_workers=num_workers)
        self._max_degree = max_degree

    def max_workers_up_to_depth(self, layer_number: int) -> int:
        d = self._max_degree
        n = layer_number
        return int(1 + d * ((d - 1) ** n - 1) / (d - 2))

    def depth_of_worker(self, worker_number: int) -> int:
        # TODO: optimize / give direct formula
        depth = 0
        while True:
            if self.max_workers_up_to_depth(depth) > worker_number:
                return depth
            depth += 1

    def parent(self, worker_number: int) -> int:
        depth = self.depth_of_worker(worker_number)
        if depth == 0:
            return None
        index_within_layer = worker_number - self.max_workers_up_to_depth(depth - 1)
        if depth == 1:
            parent_within_layer = index_within_layer // (self._max_degree)
        else:
            parent_within_layer = index_within_layer // (self._max_degree - 1)
        return parent_within_layer + self.max_workers_up_to_depth(depth - 2)

    def children(self, worker_number: int) -> Iterable[int]:
        if worker_number == 0:
            children = [1 + x for x in range(self._max_degree)]
        else:
            depth = self.depth_of_worker(worker_number)
            start_idx_my_depth = self.max_workers_up_to_depth(depth - 1)
            start_idx_next_depth = self.max_workers_up_to_depth(depth)
            i = worker_number - start_idx_my_depth
            d = self._max_degree
            children = [start_idx_next_depth + (d - 1) * i + x for x in range(d - 1)]
        return [c for c in children if c < self.num_workers]

    def neighbors(self, worker: int) -> Iterable[int]:
        if worker == 0:
            return self.children(worker)
        else:
            return [self.parent(worker)] + self.children(worker)


class NetworkxTopology(Topology):
    def __init__(self, nx_graph):
        super().__init__(num_workers=len(nx_graph.nodes))
        self.graph = networkx.relabel.convert_node_labels_to_integers(nx_graph)

    def neighbors(self, worker: int) -> Iterable[int]:
        return list(self.graph.neighbors(worker))


class SocialNetworkTopology(NetworkxTopology):
    def __init__(self):
        nx_graph = networkx.davis_southern_women_graph()
        super().__init__(nx_graph)


class SocialNetworkTreeTopology(NetworkxTopology):
    def __init__(self, root_node):
        g = networkx.davis_southern_women_graph()
        nx_graph = self.best_spanning_tree_with_root(g, root_node)
        super().__init__(nx_graph)

    @staticmethod
    def best_spanning_tree_with_root(nx_graph, root_node):
        g = networkx.relabel.convert_node_labels_to_integers(nx_graph)
        edges = set()
        for n in g.nodes:
            path = sorted(networkx.all_shortest_paths(g, root_node, n))[0]
            for i, j in zip(path[:-1], path[1:]):
                edges.add((i, j))

        gg = networkx.Graph()
        for n in g.nodes:
            gg.add_node(n)
        gg.add_edges_from(edges)
        assert networkx.is_tree(gg)

        return gg


class ArtificialAdversarialTopology(Topology):
    def __init__(self, num_workers, rho):
        self.rho = rho
        super().__init__(num_workers)

    def gossip_matrix(self):
        n = self.num_workers
        ee = torch.ones([n, n]) / n
        m = self.rho * ee
        m += torch.diag(1 - torch.sum(m, 1))
        return m


class ArtificialNiceTopology(Topology):
    def __init__(self, num_workers, rho):
        self.rho = rho
        super().__init__(num_workers)

    def gossip_matrix(self):
        n = self.num_workers
        ee = torch.ones([n, n], dtype=torch.float64) / n
        with torch.random.fork_rng():
            torch.manual_seed(0)
            v = torch.randn([n, 1], dtype=torch.float64)
            v -= v.mean()
            v /= v.norm()

        m = ee + (1 - self.rho) * v @ v.T
        m = m.float()
        m += torch.diag(1 - torch.sum(m, 1))  # numerical correction, should be small
        return m


class BinaryTreeTopology(Topology):
    def __init__(self, num_workers, reverse=False):
        super().__init__(num_workers=num_workers)
        self.reverse = reverse

    def neighbors(self, worker):
        if self.num_workers < 2:
            return []
        elif worker >= self.num_workers or worker < 0:
            raise ValueError(
                f"worker number {worker} is out of range [0, {self.num_workers})"
            )
        elif worker == 0 and not self.reverse:
            return [1]
        elif worker == self.num_workers - 1 and self.reverse:
            return [self.num_workers - 2]
        elif not self.reverse:
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [c for c in children if c < self.num_workers]
            return [parent, *children]
        elif self.reverse:
            worker = self.num_workers - 1 - worker
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [
                self.num_workers - 1 - c for c in children if c < self.num_workers
            ]
            parent = self.num_workers - 1 - parent
            return [parent, *children]


def spectral_gap(matrix):
    _, s, _ = torch.linalg.svd(matrix)
    abs_eigenvalues = torch.sqrt(s**2)
    return abs_eigenvalues[0] - abs_eigenvalues[1]
