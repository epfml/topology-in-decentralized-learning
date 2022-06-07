from math import sqrt

import torch
from topologies import AveragingScheme

from utils import solve_discrete_lyapunov


def simulate_random_walk(
    scheme: AveragingScheme, gamma: float, num_steps: int, num_reps: int
) -> torch.Tensor:
    x = torch.zeros([scheme.n, num_reps])

    for t in range(num_steps):
        x = scheme.w(t) @ (sqrt(gamma) * x + torch.randn_like(x))

    return x


def effective_number_of_neighbors(
    scheme: AveragingScheme, gamma: float, t: int = 0, mode="mean"
):
    var_per_worker = random_walk_covariance(scheme, gamma, t).diag()
    if mode == "mean":
        return 1 / (1 - gamma) / var_per_worker.mean()
    elif mode == "worst":
        return 1 / (1 - gamma) / var_per_worker.max()
    elif mode == "all":
        return 1 / (1 - gamma) / var_per_worker
    else:
        raise ValueError("Unknown mode")


def random_walk_covariance_static(
    gossip_matrix: torch.Tensor, gamma: float, start_at: int = 1
) -> torch.Tensor:
    """
    Asymptotic covariance `E[x x^T]` in the random walk process
    `x <- W @ (sqrt(gamma) x + n)`,
    where `x` is a vector containing one scalar per worker
    and `n` is i.i.d. standard normal noise.
    """
    W = gossip_matrix
    w_is_symmetric = W.allclose(W.T)
    if w_is_symmetric:
        L, Q = torch.linalg.eigh(W)
        numerator = L.square() if start_at == 1 else 1
        diag = numerator / (1 - gamma * L.square())
        return (Q * diag) @ Q.T  # = Q @ torch.diag(diag) @ Q.T
    else:
        rhs = W @ W.T if start_at == 1 else torch.eye(len(W))
        return solve_discrete_lyapunov(sqrt(gamma) * W, rhs)


def random_walk_covariance(
    scheme: AveragingScheme, gamma: float, t: int = 0, start_at: int = 1
) -> torch.Tensor:
    """
    Asymptotic covariance `E[x x^T]` in the (periodically time-varying) random walk process
    `x <- W[t] @ (sqrt(gamma) x + n)`,
    where `x` is a vector containing one scalar per worker
    and `n` is i.i.d. standard normal noise.
    The covariance is evaluated after averaging with `W[n * period + t]`.
    """
    if scheme.period == 1:
        return random_walk_covariance_static(
            gossip_matrix=scheme.w(0), gamma=gamma, start_at=start_at
        )

    period = scheme.period
    n = scheme.n

    # We split all the terms by their length mod period,
    # and compute contributions from each of them
    cumulative_cov = torch.zeros([n, n])
    for len_mod_period in range(period):
        # Compute the transition matrix
        T = torch.eye(n)
        for s in range(
            t - period - len_mod_period - start_at, t - len_mod_period - start_at
        ):
            T = scheme.w(s) @ T

        cov = random_walk_covariance_static(
            gossip_matrix=T, gamma=gamma**period, start_at=0
        )
        for s in range(t - len_mod_period - start_at, t):
            cov = scheme.w(s) @ cov @ scheme.w(s).T
        cumulative_cov += cov * gamma ** (len_mod_period)

    return cumulative_cov
