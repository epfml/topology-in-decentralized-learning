# Beyond spectral gap: The role of the topology in decentralized learning

Abstract: In data-parallel optimization of machine learning models, workers collaborate to improve their estimates of the model: more accurate gradients allow them to use larger learning rates and optimize faster.
We consider the setting in which all workers sample from the same dataset,
and communicate over a sparse graph (decentralized).
In this setting, current theory fails to capture important aspects of real-world behavior.
First, the ‘spectral gap’ of the communication graph is not predictive of its empirical performance in (deep) learning.
Second, current theory does not explain that collaboration enables *larger* learning rates than training alone.
In fact, it prescribes *smaller* learning rates, which further decrease as graphs become larger, failing to explain convergence in infinite graphs.
This paper aims to paint an accurate picture of sparsely-connected distributed optimization when workers share the same data distribution.
We quantify how the graph topology influences convergence in a quadratic toy problem and provide theoretical results for general smooth and (strongly) convex objectives.
Our theory matches empirical observations in deep learning, and accurately describes the relative merits of different graph topologies.

- [Paper (pre-print)](#)

## Reference
If you use this code, please cite the following paper

```
@article{vogels2022bsg,
  author    = {Thijs Vogels and Hadrien Hendrikx and Martin Jaggi},
  title     = {Beyond spectral gap: The role of the topology in decentralized learning},
  journal   = {CoRR},
  volume    = {abs/TODO},
  year      = {2022},
  ee        = {http://arxiv.org/abs/TODO},
}
```
