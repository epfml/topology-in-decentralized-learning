import math
from typing import List

import numpy as np


def distribute_data_random(dataset_size: int, num_workers: int, seed: int = 0) -> List[List[int]]:
    indices = list(np.arange(dataset_size))
    np.random.RandomState(seed).shuffle(indices)
    return np.array_split(indices, num_workers)



def distribute_data_dirichlet(
    targets: int, non_iid_alpha: float, num_workers: int, seed: int = 0, num_auxiliary_workers: int = 10
) -> List[List[int]]:
    """Code adapted from Tao Lin (partition_data.py)"""
    random_state = np.random.RandomState(seed=seed)

    num_indices = len(targets)
    num_classes = len(np.unique(targets))

    indices2targets = np.array(list(enumerate(targets)))
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(num_workers / num_auxiliary_workers)
    split_num_workers = [
        num_auxiliary_workers
        if idx < num_splits - 1
        else num_workers - num_auxiliary_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_num_workers / num_workers for _num_workers in split_num_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(num_auxiliary_workers / num_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _num_workers = min(num_auxiliary_workers, num_workers)
        num_workers = num_workers - num_auxiliary_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _num_workers):
            _idx_batch = [[] for _ in range(_num_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _num_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _num_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch
