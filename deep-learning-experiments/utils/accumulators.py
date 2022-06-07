from typing import Any, NamedTuple

import pax
import torch


class RunningAvgState(NamedTuple):
    avg: Any
    count: int

def running_avg_step(state: RunningAvgState, tree: Any, weight: int = 1) -> RunningAvgState:
    if state is None:
        return RunningAvgState(tree, weight)
    else:
        avg, count = state
        new_count = count + weight
        update = lambda mean, x: mean + (x - mean) / (new_count)
        return RunningAvgState(pax.tree_map(update, avg, tree), new_count)


class EmaState(NamedTuple):
    avg: Any
    weight: float


def ema_init(tree: Any) -> EmaState:
    return EmaState(pax.tree_map(torch.zeros_like, tree), 0)

def ema_step(state: EmaState, tree: Any, gamma) -> EmaState:
    if state is None:
        return EmaState(
            pax.tree_map(lambda x: x * (1-gamma), tree), 
            1 - gamma
        )
    elif gamma == 0:
        return RunningAvgState(tree, 1)
    else:
        avg, weight = state
        new_weight = weight * gamma + (1 - gamma)
        update = lambda a, b: gamma * a + (1 - gamma) * b
        return RunningAvgState(pax.tree_map(update,  avg, tree), new_weight)


def ema_read(state: EmaState, use_correction=False):
    if use_correction:
        return pax.tree_map(lambda x: x / state.weight, state.avg)
    else:
        return state.avg
