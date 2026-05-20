from typing import List

import torch
import torch.nn as nn


class PointwiseHead(nn.Module):
    """Time-invariant per-timestep MLP, implemented as a stack of 1x1 Conv1ds.

    A ``Conv1d(in_c, out_c, kernel_size=1)`` applied to a ``(B, in_c, T)``
    tensor is mathematically identical to applying a single shared
    ``nn.Linear(in_c, out_c)`` to every time slice independently and
    restacking — same weights at every day, no mixing across the time
    dimension. Stacking several such 1x1 convs with ReLU between them
    therefore implements a small MLP that is applied pointwise (per
    timestep) with shared parameters across the sequence.

    The 1x1-conv formulation is preferred over ``nn.Linear`` here only
    because it accepts channels-first tensors directly, which is the
    layout produced by Conv1d / LSTM encoders.

    Shape:
        input:  ``(B, in_size, T)``
        output: ``(B, out_size, T)``

    A depth of 1 collapses to a single 1x1 ``Conv1d`` (a plain
    per-timestep linear projection); deeper stacks interleave
    ``hidden_size``-channel 1x1 convs with ReLU activations.
    """

    def __init__(
        self,
        num_layers: int,
        in_size: int,
        hidden_size: int,
        out_size: int,
    ) -> None:
        assert num_layers >= 1

        super().__init__()

        if num_layers == 1:
            self._net: nn.Module = nn.Conv1d(in_size, out_size, kernel_size=1)
        else:
            layers: List[nn.Module] = [
                nn.Conv1d(in_size, hidden_size, kernel_size=1),
                nn.ReLU(),
            ]
            for _ in range(num_layers - 2):
                layers += [
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                    nn.ReLU(),
                ]
            layers.append(nn.Conv1d(hidden_size, out_size, kernel_size=1))
            self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)
