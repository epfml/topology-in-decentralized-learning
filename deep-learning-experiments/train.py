import os
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pax
import pyexr
import torch.nn
import torch.utils.data
import torchvision
from torch.utils.data import BatchSampler
from torch.utils.data import RandomSampler

from schemes import scheme_for_string
from tasks.cifar.models.resnet20 import ResNet20
from tasks.cifar.models.vgg import vgg11
from utils.accumulators import running_avg_step
from utils.communication import pack
from utils.communication import unpack
from utils.timer import Timer

config = {
    "seed": 1,
    "model_name": "MLP",
    "non_iid_alpha": None,
    "num_data_splits": None,
    "task": "FashionMNIST",
    "topology": "Ring",
    "learning_rate": 0.05,
    "num_epochs": 50,
    "batch_size": 32,
    "step_decay": [
        [1, 1]
    ],  # [[1, 0.75], [0.1, .25]] means "full learning rate for 75%, 0.1x for 25%"
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 32,
    "num_workers_to_eval": 2,
    "eval_batch_size": 2000,
    "eval_alpha": 0.75,  # in [0.5, 1]. 0.5 means uniform, larger numbers focus alpha % of evaluation on the first half of each time window
    "eval_budget": 100,
    "ema_gamma": 0.0,
    "inter_worker_distances": "last iterate",
}


output_dir = "output.tmp"


def main():
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timer = Timer()

    forward, params = configure_model(device)

    # Broadcast the parameters over the workers
    param_matrix = torch.tile(
        params[None, :], [config["num_workers"], 1]
    )  # shape [n, d], where d is the size of the parameter vector
    param_matrix_ema = param_matrix.clone()

    def consensus_distance():
        return torch.var(param_matrix, 0, unbiased=False).sum()

    # Define the loss of a single worker and its gradient
    def loss(params, x, y):
        prediction = forward(params, x)
        return torch.nn.functional.cross_entropy(prediction, y)

    # Create the network topology
    workers = list(range(config["num_workers"]))
    gossip_matrix: Callable = scheme_for_string(
        config["topology"], config["num_workers"], device
    ).w

    # CUDA streams for each worker so they can work more in parallel
    worker_streams = [torch.cuda.Stream() for worker in workers]
    main_stream = torch.cuda.default_stream()

    data_root = os.path.join(os.getenv("DATA", ""), "data")
    if config["task"] == "Cifar":
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        dataset = torchvision.datasets.CIFAR10
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
    elif config["task"] == "FashionMNIST":
        transform_train = transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.2860, 0.3530),
            ]
        )
        dataset = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Unknown task")
    training_set = dataset(
        root=data_root, train=True, download=True, transform=transform_train
    )
    train_test_set = dataset(
        root=data_root, train=True, download=True, transform=transform_test
    )
    test_set = dataset(
        root=data_root, train=False, download=True, transform=transform_test
    )
    train_iterator = iter(
        torch.utils.data.DataLoader(
            training_set,
            batch_size=config["batch_size"] * config["num_workers"],
            sampler=RandomSampler(
                training_set,
                replacement=True,
                num_samples=(
                    int(
                        len(training_set) * config["num_epochs"] * 4 * 100
                    )  # plenty / never stop
                ),
            ),
            drop_last=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(config["seed"]),
            pin_memory=True,
        )
    )

    test_images = torch.stack(list(test_set[i][0] for i in range(len(test_set)))).to(
        device
    )
    test_labels = torch.stack(
        list(torch.tensor(test_set[i][1]) for i in range(len(test_set)))
    ).to(device)
    train_test_images = torch.stack(
        list(train_test_set[i][0] for i in range(len(train_test_set)))
    ).to(device)
    train_test_labels = torch.stack(
        list(torch.tensor(train_test_set[i][1]) for i in range(len(train_test_set)))
    ).to(device)

    # Optimizer state
    optim_state = SGDState(momentum=torch.zeros_like(param_matrix))

    # Training loop
    num_batches_per_epoch = int(
        len(training_set) // (config["batch_size"] * config["num_workers"])
    )
    num_iterations = int(config["num_epochs"] * num_batches_per_epoch)
    batch_num = -1
    for lr_factor, fraction_of_training in config["step_decay"]:
        local_num_iterations = int(fraction_of_training * num_iterations)
        local_eval_budget = int(fraction_of_training * config["eval_budget"])
        for local_batch_num in range(local_num_iterations):
            batch_num += 1
            epoch = batch_num / num_batches_per_epoch

            with timer("data loading"):
                all_images, all_labels = next(train_iterator)
                all_images, all_labels = all_images.to(device), all_labels.to(device)
                xs = all_images.view(
                    config["num_workers"], config["batch_size"], *all_images.shape[1:]
                )
                ys = all_labels.view(
                    config["num_workers"], config["batch_size"], *all_labels.shape[1:]
                )

            with timer("model update"):
                for worker, worker_stream, x, y, params, momentum in zip(
                    workers, worker_streams, xs, ys, param_matrix, optim_state.momentum
                ):
                    worker_stream.wait_stream(main_stream)
                    with torch.cuda.stream(worker_stream):
                        grad = pax.grad(loss)(params, x, y)
                        if config["weight_decay"] > 0:
                            grad.add_(params, alpha=config["weight_decay"])
                        if config["momentum"] > 0:
                            momentum.mul_(config["momentum"]).add_(grad)
                        else:
                            momentum[:] = grad
                        del grad

                        params.sub_(momentum, alpha=config["learning_rate"] * lr_factor)

                for stream in worker_streams:
                    main_stream.wait_stream(stream)

            with timer("gossip averaging"):
                param_matrix = gossip_matrix(batch_num) @ param_matrix

            with timer("update moving average"):
                if config["ema_gamma"] > 0:
                    param_matrix_ema.mul_(config["ema_gamma"]).add_(
                        param_matrix, alpha=(1 - config["ema_gamma"])
                    )
                else:
                    param_matrix_ema = param_matrix

            if should_eval(
                local_batch_num,
                config["eval_alpha"],
                local_eval_budget,
                local_num_iterations,
            ):
                log_info({"state.progress": batch_num / num_iterations})

                assert config["inter_worker_distances"] == "last iterate"
                # pyexr.write(
                #     os.path.join(output_dir, f"distance_matrix_{batch_num:07d}.exr"),
                #     inter_worker_distances(param_matrix).cpu().numpy(),
                # )

                current_consensus_distance = consensus_distance()
                log_metric(
                    "consensus_distance",
                    {
                        "value": current_consensus_distance,
                        "epoch": epoch,
                        "step": batch_num,
                    },
                )

                def sample_workers(n=config["num_workers_to_eval"]):
                    return torch.randperm(len(workers))[:n]

                for worker in sample_workers():
                    with timer("eval on test"):
                        worker_model = lambda images: forward(
                            param_matrix_ema[worker], images
                        )
                        for key, value in evaluate_classifier(
                            worker_model, test_images, test_labels
                        ).items():
                            exit_if_nan(value)
                            if (
                                key == "loss"
                                and config["task"] == "Cifar"
                                and value > 2.6
                            ):
                                raise RuntimeError("Diverged (loss too high)")
                            log_metric(
                                key,
                                {"value": value, "epoch": epoch, "step": batch_num},
                                {"split": "test", "worker": worker},
                            )

                # for worker in sample_workers():
                #     with timer("eval on local train"):
                #         worker_model = lambda images: forward(
                #             param_matrix_ema[worker], images
                #         )
                #         data_iterator = train_data.worker_iterator(
                #             worker, batch_size=config["eval_batch_size"], drop_last=False
                #         )
                #         for key, value in evaluate_classifier(
                #             worker_model, data_iterator
                #         ).items():
                #             exit_if_nan(value)
                #             log_metric(
                #                 key,
                #                 {"value": value, "epoch": epoch, "step": batch_num},
                #                 {"split": "local_train", "worker": worker},
                #             )
                #         del data_iterator

                for worker in sample_workers():
                    with timer("eval on global train"):
                        worker_model = lambda images: forward(
                            param_matrix_ema[worker], images
                        )
                        for key, value in evaluate_classifier(
                            worker_model, train_test_images, train_test_labels
                        ).items():
                            exit_if_nan(value)
                            log_metric(
                                key,
                                {"value": value, "epoch": epoch, "step": batch_num},
                                {"split": "train", "worker": worker},
                            )

                # mean_params = param_matrix_ema.mean(0)

                # with timer("eval mean on test"):
                #     worker_model = lambda images: forward(mean_params, images)
                #     for key, value in evaluate_classifier(
                #         worker_model, test_iterator
                #     ).items():
                #         exit_if_nan(value)
                #         log_metric(
                #             key,
                #             {"value": value, "epoch": epoch, "step": batch_num},
                #             {"split": "test", "worker": "mean"},
                #         )

                # with timer("eval mean on global train"):
                #     worker_model = lambda images: forward(mean_params, images)
                #     for key, value in evaluate_classifier(
                #         worker_model, train_test_iterator
                #     ).items():
                #         exit_if_nan(value)
                #         log_metric(
                #             key,
                #             {"value": value, "epoch": epoch, "step": batch_num},
                #             {"split": "train", "worker": "mean"},
                #         )

                # del mean_params

                print(timer.summary())

    # Store timing results
    for entry in timer.transcript():
        log_runtime(entry["event"], entry["mean"], entry["std"], entry["instances"])

    # We're done
    log_info({"state.progress": 1.0})


def configure_model(device):
    if config["model_name"] == "VGG-11":
        net = vgg11().to(device)
    elif config["model_name"] == "ResNet20":
        net = ResNet20().to(device)
    elif config["model_name"] == "MLP":
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 10),
        ).to(device)
    else:
        raise ValueError("Unknown model_name")
    orig_forward = pax.functional_module(net)
    params = pax.get_params(net)

    # Modify the forward function to take a single parameter vector instead of a dictionary
    buffer, metadata = pack_params(params)

    def forward(param_buffer, images):
        params = unpack_params(param_buffer, metadata)
        return orig_forward(params, images)

    return forward, buffer


def evaluate_classifier(model, images, labels):
    with torch.no_grad():
        mean_stats = None
        chunk_size = 2000
        for start_idx in range(0, len(images), chunk_size):
            span = slice(start_idx, start_idx + chunk_size)
            x, y = images[span], labels[span]
            batch_size = len(y)
            output = model(x)
            stats = dict(
                accuracy=torch.argmax(output, 1).eq(y).float().mean(),
                loss=torch.nn.functional.cross_entropy(output, y),
            )
            mean_stats = running_avg_step(mean_stats, stats, weight=batch_size)
        return mean_stats.avg


def pack_params(
    params: OrderedDict[str, torch.Tensor]
) -> tuple[torch.Tensor, tuple[list, list]]:
    buffer, shapes = pack(params.values())
    return buffer, (shapes, list(params.keys()))


def unpack_params(
    buffer: torch.Tensor, metadata: tuple[list, list]
) -> OrderedDict[str, torch.Tensor]:
    shapes, keys = metadata
    return OrderedDict(zip(keys, unpack(buffer, shapes)))


def should_eval(step: int, alpha: float, budget: int, num_steps: int) -> bool:
    """
    `budget`: total number of planned evaluations
    `alpha`: in [0.5, 1]. 0.5 means uniform, larger numbers focus alpha % of evaluation on the first half of each time window
    `num_steps`: how long we are going to train
    """
    t = step / num_steps
    p = -torch.log(torch.tensor(alpha)) / torch.log(torch.tensor(2.0))
    prob = min(1, budget / num_steps * p * t ** (p - 1))
    return torch.rand([]).item() < prob


def exit_if_nan(value):
    if torch.isnan(value).sum() > 0:
        raise RuntimeError("Diverged (nan)")


class SGDState(NamedTuple):
    momentum: torch.Tensor


def log_info(info_dict):
    """Add any information to MongoDB
    This function will be overwritten when called through run.py"""
    pass


def inter_worker_distances(param_matrix, stride=8192):
    """Squared norm of the difference of worker parameters"""
    total = 0
    for start in range(0, param_matrix.shape[-1], stride):
        total += (
            (
                param_matrix[:, None, start : start + stride]
                - param_matrix[None, :, start : start + stride]
            )
            .square()
            .sum(dim=-1)
        )
    return total


def log_metric(name, values, tags={}):
    """Log timeseries data
    This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print(f"{name:30s} - {values} ({tags})")


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


if __name__ == "__main__":
    main()
