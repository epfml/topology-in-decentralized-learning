import sys
sys.path.append("..")

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import pax
import torch
from typing import OrderedDict
from utils.timer import Timer
from utils.communication import pack, unpack
from tasks.cifar.models.vgg import vgg11
from typing import NamedTuple
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


class Batch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


def main():
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timer = Timer()

    net = vgg11().to(device)
    forward = pax.functional_module(net)
    params = pax.get_params(net)
    param_vector, shapes = pack(params.values())
    param_keys = list(params.keys())
    params = torch.tile(param_vector[None, :], [config["num_workers"], 1])

    def unpack_params(buffer):
        return OrderedDict(zip(param_keys, unpack(buffer, shapes)))

    topo = topologies.RingTopology(config["num_workers"])
    gossip_matrix = topo.gossip_matrix().to(device)

    builder = tfds.builder("cifar10")

    builder.download_and_prepare()

    ds = builder.as_dataset(split=tfds.even_splits("train", config["num_workers"]), shuffle_files=True)

    def prepare(dataset):
        d = (
            tf.data.Dataset.shuffle(dataset, 1000, reshuffle_each_iteration=True)
            .repeat()
            .map(lambda data: {**data, "image": tf.image.random_crop(tf.pad(data["image"], [[4, 4], [4, 4], [0, 0]]), size=(32, 32, 3))}, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(32)
            .map(lambda data: {**data, "image": tf.transpose(tf.image.random_flip_left_right(data["image"]), [0, 3, 1, 2])}, num_parallel_calls=tf.data.AUTOTUNE)
        )
        return d


    iterator = tf.data.Dataset.zip(tuple(prepare(d) for d in ds)).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    data_mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device).view(1, 3, 1, 1)
    data_stddev = torch.tensor([0.2023, 0.1994, 0.2010]).to(device).view(1, 3, 1, 1)

    def worker_loss(params, batch):
        pred = forward(unpack_params(params), batch.x)
        return torch.nn.functional.cross_entropy(pred, batch.y)

    # for _ in range(10):
    #     with timer("1x data loading"):
    #         for i, batches in enumerate(iterator):
    #             for worker in topo.workers:
    #                 x = torch.from_numpy(batches[worker]["image"]).to(device).float() / 255
    #                 (x - data_mean) / data_stddev
    #             if i > 1:
    #                 break

    for _ in range(10):
        with timer("1x dp-sgd loop"):
            for step in range(1):
                batches = next(iterator)
                for worker in topo.workers:
                    y = torch.from_numpy(batches[worker]["label"]).to(device)
                    x = torch.from_numpy(batches[worker]["image"]).to(device).float() / 255
                    x = (x - data_mean) / data_stddev
                    grad = pax.grad(worker_loss)(params[worker], Batch(x, y))
                    # print(grad.shape)
                    params[worker].add_(grad, alpha=-config["learning_rate"])
                params = gossip_matrix @ params


    batches = next(iterator)
    y = torch.from_numpy(batches[worker]["label"]).to(device)
    x = torch.from_numpy(batches[worker]["image"]).to(device).float() / 255
    x = (x - data_mean) / data_stddev
    for _ in range(10):
        with timer("just param update"):
            for worker in topo.workers:
                grad = pax.grad(worker_loss)(params[worker], Batch(x, y))
                # print(grad.shape)
                params[worker].add_(grad, alpha=-config["learning_rate"])
                
    batches = next(iterator)
    y = torch.from_numpy(batches[worker]["label"]).to(device)
    x = torch.from_numpy(batches[worker]["image"]).to(device).float() / 255
    x = (x - data_mean) / data_stddev
    streams = [torch.cuda.Stream() for worker in topo.workers]
    for _ in range(10):
        with timer("just param update with streams"):
            for worker, stream in zip(topo.workers, streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    grad = pax.grad(worker_loss)(params[worker], Batch(x, y))
                # print(grad.shape)
                params[worker].add_(grad, alpha=-config["learning_rate"])

            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
                
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
    
    # for _ in range(10):
    #     with timer("10x gossip averaging"):
    #         for step in range(10):
    #             params = gossip_matrix @ params


    # sparse_gossip_matrix = gossip_matrix.to_sparse()
    # x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    # for i in range(10):
    #     with timer("sparse matmul"):
    #         x = torch.mm(sparse_gossip_matrix, x)
    # del x

    # x = torch.tile(param_vector[None, :], [config["num_workers"], 1])
    # for i in range(10):
    #     with timer("manual loop (single matrix)"):
    #         y = torch.zeros_like(x)
    #         for worker in topo.workers:
    #             for neighbor in [worker] + topo.neighbors(worker):
    #                 y[worker].add_(x[neighbor], alpha=gossip_matrix[worker, neighbor])
    #         x = y
    #         del y
    # del x
    
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

    # param_vector, shapes = pack(pax.get_params(task._model).values())
    # params = [param_vector.clone() for worker in topo.workers]
    # for _ in range(10):
    #     with timer("manual loop (individual packed tensors)"):
    #         new_params = []
    #         for worker in topo.workers:
    #             p = params[worker].mul(gossip_matrix[worker, worker])
    #             for neighbor in topo.neighbors(worker):
    #                 p.add_(params[neighbor], alpha=gossip_matrix[worker, neighbor])
    #             new_params.append(p)
    #         params = new_params
    #         del new_params
    # del params


    print(timer.summary())


if __name__ == "__main__":
    main()