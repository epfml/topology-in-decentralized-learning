from typing import Optional

import torch
from random_walks import effective_number_of_neighbors
from topologies import AveragingScheme

from utils import top_eigenvalue


def random_quadratic_rate(
    scheme: AveragingScheme, zeta: float, learning_rate: float, max_iters: int = 100
):
    """
    Compute the convergence rate for the random quadratic problem
    based on Section 3.1 of the paper.
    Returns a constant r for which `E x[t]^2 = (1 - r) E x[t-1]^2` or `None`.
    """
    eta = learning_rate
    r = 0
    for _ in range(max_iters):  # fixed-point iteration
        gamma = (1 - eta) ** 2 / (1 - r)
        if gamma > 1 or abs(gamma - 1) < 1e-16:
            break
        n = effective_number_of_neighbors(scheme, gamma, mode="worst")
        r, prev_r = 1 - (1 - eta) ** 2 - (zeta - 1) * eta**2 / n, r
        if abs(r - prev_r) < 1e-8:
            break
        if r >= 1 or r < 0:
            return None
    return r


def random_quadratic_transition_matrix(
    scheme: AveragingScheme, zeta: float, learning_rate: float
) -> torch.Tensor:
    """
    Computes a transition matrix of size (n x n, n x n) that takes a flattened
    list of all inner-product expectations between workers, and produces the same vector for the next time step.
    """
    n = scheme.n
    product = torch.empty([])
    lr = learning_rate

    for i in range(scheme.period):
        gossip_matrix = scheme.w(i)

        w = torch.kron(gossip_matrix, gossip_matrix)
        w *= (lr - 1) ** 2

        if lr != 1:
            correction_factor = (zeta * lr**2 + 1 - 2 * lr) / ((lr - 1) ** 2)
        else:
            correction_factor = 1
        w.view(n, n, n, n)[:, :, torch.arange(n), torch.arange(n)] *= correction_factor

        if i == 0:
            product = w
        else:
            product = w @ product

    return product


def random_quadratic_rate_precise(
    scheme: AveragingScheme,
    zeta: float,
    learning_rate: float,
) -> Optional[torch.Tensor]:
    """
    Based on spectral norm of the covariance transition matrix
    This is precise for irregular topologies, while `random_quadratic_rate` is not guaranteed to be.
    """
    matrix = random_quadratic_transition_matrix(
        scheme, zeta=zeta, learning_rate=learning_rate
    )
    rate = top_eigenvalue(matrix) ** (1 / scheme.period)  # type: ignore
    return 1 - rate if 0 < rate < 1 else None


def zeta(d: int, batch_size: int = 1) -> float:
    """The constant L such that E 𝄁Hx𝄁^2 = L E 𝄁x𝄁^2"""
    return (1 + batch_size + d) / batch_size


def tuned_centralized_lr(d: int, batch_size: int = 1, num_workers: int = 1) -> float:
    """Optimal centralized (= mini-batch) learning rate"""
    return 1 / zeta(d, batch_size * num_workers)
