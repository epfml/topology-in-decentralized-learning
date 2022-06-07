import torch
from matplotlib import pyplot as plt
from random_isotropic_quadratics import random_quadratic_rate
from random_isotropic_quadratics import random_quadratic_rate_precise
from topologies import scheme_for_string


def main():
    torch.set_default_dtype(torch.float64)
    lrs = torch.logspace(-3, 0, 100)

    fig, ax = plt.subplots()
    for topology in [
        "Ring",
        "Time-varying exponential",
        "Star",
        "Binary tree",
        "Fully connected",
        "Hypercube",
    ]:
        scheme = scheme_for_string(topology, num_workers=32)
        ax.semilogx(
            lrs,
            [
                random_quadratic_rate(scheme, zeta=50, learning_rate=lr.item())
                for lr in lrs
            ],
            label=topology,
        )
    ax.set(xlabel="Learning rate", ylabel="Convergence rate")
    ax.legend()
    fig


if __name__ == "__main__":
    main()
