import os
from typing import NamedTuple

import tensorflow as tf
import torch
import torchvision

from tasks.datasets.utils import distribute_data_dirichlet
from tasks.datasets.utils import distribute_data_random


CIFAR_STATS = {
    "mean": torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1),
    "stddev": torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1),
}
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Batch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


class DistributedCifarDataset:
    def __init__(
        self,
        segment="train",
        num_splits=1,
        dataset="CIFAR10",
        split_method="random",
        non_iid_alpha=None,
        data_root=os.path.join(os.getenv("DATA"), "data"),
        seed=0,
        use_augmentation=True,
    ):
        dataset_class = getattr(torchvision.datasets, dataset)
        self._dataset = dataset_class(
            root=data_root, train=(segment == "train"), download=True
        )

        self._x = torch.from_numpy(self._dataset.data)
        self._y = torch.tensor(self._dataset.targets)

        self._use_augmentation = use_augmentation

        if split_method == "dirichlet":
            self._split_indices = distribute_data_dirichlet(
                self._y, non_iid_alpha, num_splits, seed=seed
            )
        elif split_method == "random":
            self._split_indices = distribute_data_random(
                len(self._y), num_splits, seed=seed
            )
        elif split_method == "shared-data":
            self._split_indices = [
                list(range(len(self._y))) for worker in range(num_splits)
            ]
        else:
            raise ValueError(f"Unknown split method {split_method}")

        self.mean_data_per_split = len(self._y) / num_splits

    def worker_iterator(
        self,
        worker_idx,
        batch_size,
        shuffle=True,
        repeat=False,
        drop_last=True,
        seed=0,
        subset_size=-1,
    ):
        return TfDatasetIterator(
            self._worker_dataset(
                worker_idx, batch_size, repeat, drop_last, seed, shuffle
            )
            .take(subset_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def multi_worker_iterator(
        self,
        worker_indices,
        batch_size,
        shuffle=True,
        repeat=False,
        drop_last=True,
        seed=0,
    ):
        datasets = tuple(
            self._worker_dataset(idx, batch_size, repeat, drop_last, seed, shuffle)
            for idx in worker_indices
        )
        return TfDatasetIterator(
            tf.data.Dataset.zip(datasets).prefetch(tf.data.AUTOTUNE)
        )

    def _worker_dataset(
        self, worker_idx, batch_size, repeat=False, drop_last=True, seed=0, shuffle=True
    ):
        indices = self._split_indices[worker_idx]
        x = self._x[indices]
        y = self._y[indices]
        ds = tf.data.Dataset.from_tensor_slices({"x": x, "y": y})

        if shuffle:
            n = len(self._split_indices)
            ds = ds.shuffle(
                buffer_size=1000,
                reshuffle_each_iteration=True,
                seed=seed * n + worker_idx,
            )

        if repeat:
            ds = ds.repeat()

        if self._use_augmentation:
            ds = ds.map(
                lambda data: {
                    **data,
                    "x": _tf_image_crop(data["x"], (32, 32), padding=4),
                },
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.batch(batch_size, drop_remainder=drop_last)

        if self._use_augmentation:
            ds = ds.map(
                lambda data: {
                    **data,
                    "x": _tf_bhwc_to_bchw(tf.image.random_flip_left_right(data["x"])),
                },
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda data: {**data, "x": _tf_bhwc_to_bchw(data["x"])},
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return ds


def preprocess(tf_batch, device=DEFAULT_DEVICE):
    # Ensure the statistics are on the right device
    if CIFAR_STATS["mean"].get_device() != device:
        CIFAR_STATS["mean"] = CIFAR_STATS["mean"].to(device)
        CIFAR_STATS["stddev"] = CIFAR_STATS["stddev"].to(device)

    x = torch.from_numpy(tf_batch["x"]).to(device).float() / 255
    y = torch.from_numpy(tf_batch["y"]).to(device)
    x = (x - CIFAR_STATS["mean"]) / CIFAR_STATS["stddev"]
    return Batch(x, y)


def _tf_image_crop(image, size, padding=0):
    p = padding
    return tf.image.random_crop(
        tf.pad(image, [[p, p], [p, p], [0, 0]]), size=size + (3,)
    )


def _tf_bhwc_to_bchw(batch):
    return tf.transpose(batch, [0, 3, 1, 2])


class TfDatasetIterator:
    def __init__(self, dataset: tf.data.Dataset):
        self._dataset = dataset

    def __iter__(self):
        return self._dataset.as_numpy_iterator()


if __name__ == "__main__":
    data = DistributedCifarDataset(num_splits=4)
    for batch in data.multi_worker_iterator([2, 3], batch_size=3):
        print(batch.keys())
