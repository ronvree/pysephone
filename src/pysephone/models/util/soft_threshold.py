import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class SoftThreshold(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
        slope: float = 10.0,
        threshold_requires_grad: bool = False,
        slope_requires_grad: bool = False,
        threshold_positive: bool = False,
        slope_positive: bool = False,
        logit_min: float = -40.0,
        logit_max: float = 40.0,
        eps: float = 1e-12,
        b0: float | None = None,
        b1: float | None = None,
        bk: float = 1.,
    ):
        super().__init__()
        assert 0 <= bk <= 1

        self.normalize_inputs = (b0 is not None) and (b1 is not None)
        self.eps = eps

        if self.normalize_inputs:
            self.register_buffer("b0", torch.tensor(float(b0)))
            self.register_buffer("b1", torch.tensor(float(b1)))
            self.register_buffer("bk", torch.tensor(float(bk)))
        else:
            self.b0 = None
            self.b1 = None
            self.bk = bk

        self.positive_threshold = threshold_positive
        self.positive_slope = slope_positive
        self.logit_min = logit_min
        self.logit_max = logit_max

        self._threshold = nn.Parameter(
            torch.tensor(float(threshold)),
            requires_grad=threshold_requires_grad,
        )
        self._slope = nn.Parameter(
            torch.tensor(float(slope)),
            requires_grad=slope_requires_grad,
        )

        self._softplus_shift = math.log(2.0)

    def _pos(self, p: torch.Tensor) -> torch.Tensor:
        return (F.softplus(p) - self._softplus_shift).clamp_min(0.0)

    @property
    def threshold(self) -> torch.Tensor:
        return self._pos(self._threshold) if self.positive_threshold else self._threshold

    @property
    def slope(self) -> torch.Tensor:
        return self._pos(self._slope) if self.positive_slope else self._slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            b0 = self.b0.to(dtype=x.dtype, device=x.device)
            b1 = self.b1.to(dtype=x.dtype, device=x.device)
            x = (x - b0) / ((b1 - b0 + self.eps) * self.bk)

        threshold = self.threshold
        slope = self.slope

        z = (slope * (x - threshold)).clamp(self.logit_min, self.logit_max)
        return torch.sigmoid(z)
