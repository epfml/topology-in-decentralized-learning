import math
from functools import lru_cache
from typing import Any
from typing import Union

import numpy as np
import topologies
import torch
from matplotlib import pyplot as plt


class AveragingScheme:
    # Some defaults
    period = 1
    n = 1

    def init(self):
        return None

    @property
    def state_size(self):
        return self.n

    def w(self, t=0, params=None):
        return torch.eye(1)

    def show_weights(self, params=None, **kwargs):
        fig, axes = plt.subplots(ncols=self.period)
        if self.period == 1:
            axes = [axes]
        for t in range(self.period):
            axes[t].set_title(f"t={t}")
            axes[t].matshow(self.w(params=params, t=t), **kwargs)

        for ax in axes:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        return fig


class MetropolisHastings(AveragingScheme):
    def __init__(self, graph: Union[torch.Tensor, topologies.Topology]):
        super().__init__()

        if isinstance(graph, topologies.Topology):
            graph = graph.gossip_matrix()

        self.connectivity = graph > 0
        self.n = len(graph)

    def w(self, t=0, params=None):
        num_neighbors = self.connectivity.sum(1, True)
        normalization = 1 / torch.max(num_neighbors, num_neighbors.T)
        m = self.connectivity * normalization
        m += torch.diag(1 - m.sum(1))
        return m


class Matrix(AveragingScheme):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix = matrix
        self.n = len(matrix)

    def w(self, t=0, params=None):
        return self.matrix


class TimeVaryingExponential(AveragingScheme):
    def __init__(self, n, device):
        super().__init__()
        self.n = n
        self.d = int(math.log(n, 2))
        self.period = self.d
        self.device = device
        assert 2**self.d == self.n

    def w(self, t=0, params=None):
        offset = 2 ** (t % self.d)
        return self._w(offset)

    @lru_cache(maxsize=10)
    def _w(self, offset):
        w = torch.eye(self.n)
        w = (w + torch.roll(w, -offset, 0)) / 2
        w = w.to(self.device)
        return w


class SymmetricDoublyStochastic(AveragingScheme):
    def __init__(self, connectivity):
        super().__init__()
        self.connectivity = connectivity > 0
        self.n = len(connectivity)
        self.max_neighbors = self.connectivity.sum(1).max()

    def init(self):
        return torch.rand([self.n, self.n]) / self.max_neighbors * 2

    def w(self, t=0, params=None):
        m = make_doubly_stochastic(self.connectivity * params)
        return m


class DoublyStochastic(AveragingScheme):
    def __init__(self, connectivity):
        super().__init__()
        self.connectivity = connectivity > 0
        self.n = len(connectivity)
        self.max_neighbors = self.connectivity.sum(1).max()

    def init(self):
        return torch.rand([self.n, self.n]) / self.max_neighbors * 2

    def w(self, t=0, params=None, num_iterations=20):
        m = sinkhorn_normalization(
            self.connectivity * params, num_iterations=num_iterations
        )
        return m


class Delays(AveragingScheme):
    def __init__(self, connectivity, max_delay):
        super().__init__()
        self.connectivity = connectivity > 0
        self.max_delay = max_delay
        self.n = len(connectivity)
        self._state_size = self.n * (max_delay + 1)
        self.max_neighbors = self.connectivity.sum(1).max()

    @property
    def state_size(self):
        return self._state_size

    def init(self):
        return (
            torch.rand([self.n * self.n * (self.max_delay + 2)])
            / self.max_neighbors
            * 2
        )

    def w(self, t=0, params=None):
        m = make_doubly_stochastic(
            self.connectivity * params[: self.n * self.n].view(self.n, self.n)
        )

        weights = torch.softmax(
            params[self.n * self.n :].view(self.max_delay + 1, self.n * self.n), dim=0
        )

        w = torch.zeros([self.state_size, self.state_size])
        for i, weight in enumerate(weights):
            w[: self.n, self.n * i : self.n * (i + 1)] = weight.view(*m.shape) * m
        w[self.n :, : self.n * self.max_delay] = torch.eye(self.max_delay * self.n)
        return w


class TimeVaryingSymmetricDoublyStochastic(AveragingScheme):
    def __init__(self, connectivity, period):
        super().__init__()
        self.connectivity = connectivity > 0
        self.period = period
        self.n = len(connectivity)
        self.max_neighbors = self.connectivity.sum(1).max()

    def init(self):
        return torch.rand([self.period, self.n, self.n]) / self.max_neighbors * 2

    def w(self, t=0, params=None):
        return make_doubly_stochastic(params[t % self.period] * self.connectivity)


class TimeVaryingDoublyStochastic(AveragingScheme):
    def __init__(self, connectivity, period):
        super().__init__()
        self.connectivity = connectivity > 0
        self.period = period
        self.n = len(connectivity)
        self.max_neighbors = self.connectivity.sum(1).max()

    def init(self):
        return torch.rand([self.period, self.n, self.n]) / self.max_neighbors * 2

    def w(self, t=0, params=None):
        return sinkhorn_normalization(params[t % self.period] * self.connectivity)


class TunedScheme(AveragingScheme):
    def __init__(self, base_scheme, max_neighbors=2):
        super().__init__()
        self.base_scheme = base_scheme
        self.period = self.base_scheme.period
        self.n = self.base_scheme.n
        self.max_neighbors = max_neighbors

    def init(self):
        return torch.rand([self.period, self.n, self.n]) / self.max_neighbors

    def w(self, t=0, params=None):
        connectivity = self.base_scheme.w(t=t, params=self.base_scheme.init()) > 0
        return sinkhorn_normalization(params[t % self.period] * connectivity)


class TimeVaryingRing(AveragingScheme):
    def __init__(self, n):
        assert n % 2 == 0
        self.n = n
        self.period = 2

    def w(self, t=0, params=None):
        n = self.n

        # Construct an n x n gossip matrix
        w = torch.zeros([n, n])

        # Divide workers in pairs of two
        for i in range(0, n, 2):
            w[i : i + 2, i : i + 2] = torch.ones([2, 2]) * 2 / 3

        # At odd time steps, shift the groups
        if t % 2 == 1:
            w = torch.roll(w, [1, 1], [0, 1])

        # Correct the weights on the diagonal
        w += torch.diag(1 - torch.sum(w, dim=1))

        return w


def make_doubly_stochastic(matrix):
    # Make symmetric
    triu = torch.triu(matrix, diagonal=1)
    m = triu + triu.T
    # m = (matrix + matrix.T)/2
    # Make stochastic
    m += torch.diag(1 - m.sum(1))
    return m


def sinkhorn_normalization(matrix: torch.Tensor, num_iterations=10):
    matrix = matrix.clone()
    for i in range(num_iterations):
        total = matrix.sum(0, keepdims=True)
        total[total < 1e-16] = 1e-16
        matrix /= total
        total = matrix.sum(1, keepdims=True)
        total[total < 1e-16] = 1e-16
        matrix /= total
    return matrix


def topology_from_string(topology, num_workers):
    if topology == "Ring":
        return topologies.RingTopology(num_workers)
    elif topology == "Uni-ring":
        return topologies.UnidirectionalRingTopology(num_workers)
    elif topology == "2D Torus":
        n = int(np.sqrt(num_workers))
        assert np.power(n, 2) == num_workers
        return topologies.TorusTopology(n, n)
    elif topology == "Torus (4x8)":
        return topologies.TorusTopology(4, 8)
    elif topology == "Torus (8x8)":
        return topologies.TorusTopology(8, 8)
    elif topology == "Torus (2x16)":
        return topologies.TorusTopology(2, 16)
    elif topology == "Binary tree":
        return topologies.BinaryTreeTopology(num_workers)
    elif topology == "Two cliques":
        return topologies.TwoCliquesTopology(num_workers)
    elif topology == "Hypercube":
        return topologies.HyperCubeTopology(num_workers)
    elif topology == "Star":
        return topologies.StarTopology(num_workers)
    elif topology == "Social network":
        assert num_workers == 32
        return topologies.SocialNetworkTopology()
    elif topology == "Fully connected":
        return topologies.FullyConnectedTopology(num_workers)
    elif topology == "Adversarial 2":
        return topologies.ArtificialAdversarialTopology(num_workers, 0.2)
    elif topology == "Adversarial 4":
        return topologies.ArtificialAdversarialTopology(num_workers, 0.4)
    elif topology == "Adversarial 6":
        return topologies.ArtificialAdversarialTopology(num_workers, 0.6)
    elif topology == "Adversarial 8":
        return topologies.ArtificialAdversarialTopology(num_workers, 0.8)
    elif topology == "Nice 2":
        return topologies.ArtificialNiceTopology(num_workers, 0.2)
    elif topology == "Nice 4":
        return topologies.ArtificialNiceTopology(num_workers, 0.4)
    elif topology == "Nice 6":
        return topologies.ArtificialNiceTopology(num_workers, 0.6)
    elif topology == "Nice 8":
        return topologies.ArtificialAdversarialTopology(num_workers, 0.8)
    elif topology == "Solo":
        return topologies.DisconnectedTopology(num_workers)
    else:
        raise ValueError(f"Unknown topology {topology}")


def scheme_for_string(topology: str, num_workers: int, device: Any) -> AveragingScheme:
    try:
        W = topology_from_string(topology, num_workers).gossip_matrix().to(device)
        return Matrix(W)
    except ValueError as e:
        if topology == "Time-varying exponential":
            return TimeVaryingExponential(num_workers, device)
        else:
            raise e
