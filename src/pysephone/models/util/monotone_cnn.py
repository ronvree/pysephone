"""Pointwise monotone network via softplus-constrained Conv1d weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneCNN(nn.Module):
    """Pointwise monotone-increasing network.

    Uses kernel_size=1 convolutions with softplus-constrained weights so that
    all weights are strictly positive. Combined with sigmoid activations (which
    are themselves monotone), the full network is guaranteed monotone increasing.

    Input/output shape: ``(B, 1, T)`` — applied independently at each timestep.
    Output is in ``(0, 1)``.

    Args:
        hidden_size: Number of hidden channels.
        num_layers:  Total number of layers (including input and output).
    """

    def __init__(self, hidden_size: int = 16, num_layers: int = 3) -> None:
        super().__init__()
        assert num_layers >= 1

        sizes = [1] + [hidden_size] * (num_layers - 1) + [1]
        self._raw_weights = nn.ParameterList([
            nn.Parameter(torch.randn(sizes[i + 1], sizes[i], 1) * 0.1)
            for i in range(len(sizes) - 1)
        ])
        self._biases = nn.ParameterList([
            nn.Parameter(torch.zeros(sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(B, 1, T)``
        Returns:
            ``(B, 1, T)`` in ``(0, 1)``, monotone increasing in the input.
        """
        for i, (raw_w, b) in enumerate(zip(self._raw_weights, self._biases)):
            w = F.softplus(raw_w)
            x = F.conv1d(x, w, b)
            x = torch.sigmoid(x)
        return x
