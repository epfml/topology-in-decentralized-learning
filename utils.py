from collections.abc import Callable

import scipy.linalg
import torch


def solve_discrete_lyapunov(
    A: torch.Tensor, Q: torch.Tensor, method=None
) -> torch.Tensor:
    """
    Solves the discrete Lyapunov equation `A X A^T - X + Q = 0`.
    This is a wrapper around scipy that accepts torch tensors.
    """
    device = A.device
    assert Q.device == A.device
    A = A.cpu().numpy()
    Q = Q.cpu().numpy()
    out = scipy.linalg.solve_discrete_lyapunov(A, Q, method)
    return torch.from_numpy(out).to(device)


def top_eigenvalue(
    matrix: torch.Tensor,
    threshold: float = 1e-5,
    max_iters: int = 1000,
    only_sub_1: bool = True,
    return_eigenvector: bool = False,
):
    """
    Find the top eigenvalue of a matrix through power iteration

    """
    n, m = matrix.shape
    assert n == m

    # special case for diagonal matrices
    diag = matrix.diag()
    if diag.diag().allclose(matrix):
        return diag.abs().max()

    return top_eigenvalue_of_operator(
        operator=lambda x: matrix @ x,
        n=n,
        threshold=threshold,
        max_iters=max_iters,
        only_sub_1=only_sub_1,
        return_eigenvector=return_eigenvector,
    )


def top_eigenvalue_of_operator(
    operator: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    threshold: float = 1e-5,
    max_iters: int = 1000,
    only_sub_1: bool = True,
    return_eigenvector: bool = False,
):
    """
    Find the top eigenvalue of a matrix through power iteration
    """

    x = torch.ones([n, 1])
    norm = torch.linalg.norm(x)
    prev_norm = 0

    for i in range(max_iters):
        x /= norm
        x = operator(x)
        norm, prev_norm, prevprev_norm = torch.linalg.norm(x), norm, prev_norm
        if torch.abs(norm - prev_norm) < threshold:
            break
        if norm > prev_norm and torch.abs(norm - prevprev_norm) < threshold:
            break
        if only_sub_1 and norm > 1 and not return_eigenvector:
            return torch.ones([])

    if return_eigenvector:
        return norm, x / torch.linalg.norm(x)
    else:
        return norm
