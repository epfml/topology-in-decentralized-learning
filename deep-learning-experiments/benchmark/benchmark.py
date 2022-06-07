from typing import OrderedDict
import pax
import torch
from tasks.cifar import CifarTask
from utils.timer import Timer
from tasks.api import Task
from utils.communication import pack, unpack
import topologies

config = {
    "seed": 1,
    "data_split_method": "dirichlet",
    "model_name": "VGG-11",
    "non_iid_alpha": 0.01,
    "num_data_splits": None,
    "task": "Cifar",
    "learning_rate": 0.05,
    "batch_size": 32,
    "weight_decay": 1e-4,
    "num_workers": 64,
}


def main():
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timer = Timer()
    task = CifarTask(model_name=config["model_name"], weight_decay=config["weight_decay"])
    train_splits = task.split_data(config["data_split_method"], config["num_workers"], config["non_iid_alpha"], config["seed"] + 14)

    params, _ = task.initialize(config["seed"] + 34)
    param_vector, shapes = pack(params.values())
    param_keys = params.keys()
    params = torch.tile(param_vector[None, :], [config["num_workers"], 1])

    def unpack_params(buffer):
        return OrderedDict(zip(param_keys, unpack(buffer, shapes)))

    topo = topologies.RingTopology(config["num_workers"])
    gossip_matrix = topo.gossip_matrix().to(device)

    iterators = [s.iterator(batch_size=config["batch_size"], shuffle=True, repeat=True, num_workers=1, pin_memory=True) for s in train_splits]

    def worker_loss(params, batch):
        pred = task._forward(unpack_params(params), batch._x)
        return torch.nn.functional.cross_entropy(pred, batch._y)

    # for _ in range(10):
    #     with timer("10x data loading"):
    #         for i, batches in enumerate(zip(*iterators)):
    #             if i > 10:
    #                 break

    # for _ in range(10):
    #     with timer("10x dp-sgd loop"):
    #         for step in range(10):
    #             for worker in topo.workers:
    #                 epoch, batch = next(iterators[worker])
    #                 params[worker].add_(pax.grad(worker_loss)(params[worker], batch), alpha=-config["learning_rate"])
    #             params = gossip_matrix @ params

    # for _ in range(10):
    #     with timer("10x local sgd loop"):
    #         for step in range(10):
    #             for worker in topo.workers:
    #                 epoch, batch = next(iterators[worker])
    #                 params[worker].add_(pax.grad(worker_loss)(params[worker], batch), alpha=-config["learning_rate"])

    # batches = [next(iterators[worker])[1] for worker in topo.workers]
    # for _ in range(10):
    #     with timer("10x local sgd loop w/o loading"):
    #         for step in range(10):
    #             for worker in topo.workers:
    #                 epoch, batch = next(iterators[worker])
    #                 params[worker].add_(pax.grad(worker_loss)(params[worker], batches[worker]), alpha=-config["learning_rate"])


    # batches = [next(iterators[worker])[1] for worker in topo.workers]
    # for _ in range(10):
    #     with timer("10x dp-sgd loop w/o loading"):
    #         for step in range(10):
    #             for worker in topo.workers:
    #                 params[worker].add_(pax.grad(worker_loss)(params[worker], batches[worker]), alpha=-config["learning_rate"])
    #             params = gossip_matrix @ params

    # x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    # for _ in range(10):
    #     with timer("gradient: n x forward and n backward"):
    #         for worker in topo.workers:
    #             x[worker].add_(pax.grad(worker_loss)(x[worker], batches[worker][1]), alpha=-config["learning_rate"])

    # del x
    # del batches
    
    for _ in range(10):
        with timer("10x gossip averaging"):
            for step in range(10):
                params = gossip_matrix @ params


    # sparse_gossip_matrix = gossip_matrix.to_sparse()
    # x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    # for i in range(10):
    #     with timer("sparse matmul"):
    #         x = torch.mm(sparse_gossip_matrix, x)
    # del x

    x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    for i in range(10):
        with timer("10x manual loop (single matrix)"):
            for _ in range(10):
                y = torch.zeros_like(x)
                for worker in topo.workers:
                    for neighbor in [worker] + topo.neighbors(worker):
                        y[worker].add_(x[neighbor], alpha=gossip_matrix[worker, neighbor])
                x = y
            del y
    del x
    
    x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    for i in range(10):
        with timer("10x manual loop (single matrix) v2"):
            for _ in range(10):
                y = torch.empty_like(x)
                for worker in topo.workers:
                    y[worker] = (1 - 2/3) * x[worker] + 1/3 * x[(worker - 1) % 64] + 1/3 * x[(worker + 1) % 64]
                x = y
            del y
    del x
    
    # x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    # for i in range(10):
    #     with timer("optimized for ring (torch.roll)"):
    #         y = (x + torch.roll(x, [1], [0]) + torch.roll(-x, [-1], [0])) / 3
    #         x = y
    #         del y
    
    # params = [[x.clone() for x in pax.get_params(task._model).values()] for w in topo.workers]
    # for _ in range(10):
    #     with timer("manual loop (individual tensors)"):
    #         new_params = []
    #         for worker in topo.workers:
    #             p = []
    #             new_params.append(p)
    #             for i in range(len(params[worker])):
    #                 p.append(params[worker][i].mul(gossip_matrix[worker, worker]))
    #                 for neighbor in topo.neighbors(worker):
    #                     p[-1].add_(params[neighbor][i], alpha=gossip_matrix[worker, neighbor])
    #         params = new_params
    #         del new_params
    # del params

    params = [param_vector.clone() for worker in topo.workers]
    for _ in range(10):
        with timer("10x manual loop (individual packed tensors)"):
            for i in range(10):
                new_params = []
                for worker in topo.workers:
                    p = params[worker].mul(gossip_matrix[worker, worker])
                    for neighbor in topo.neighbors(worker):
                        p.add_(params[neighbor], alpha=gossip_matrix[worker, neighbor])
                    new_params.append(p)
                params = new_params
                del new_params
    # del params


    x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    @torch.jit.script
    def average(x):
        results = []
        for worker in range(64):
            results.append((1 - 2/3) * x[worker] + 1/3 * x[(worker - 1) % 64] + 1/3 * x[(worker + 1) % 64])
        return torch.stack(results)
    for i in range(10):
        with timer("10x jitted manual loop (single matrix)"):
            for _ in range(10):
                x = average(x)
    del x

    x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    @torch.jit.script
    def average(x):
        out = torch.empty_like(x)
        for worker in range(64):
            out[worker] = (1 - 2/3) * x[worker] + 1/3 * x[(worker - 1) % 64] + 1/3 * x[(worker + 1) % 64]
        return out
    for i in range(10):
        with timer("10x jitted manual loop (single matrix) v2"):
            for _ in range(10):
                x = average(x)
    del x


    print(timer.summary())


if __name__ == "__main__":
    main()