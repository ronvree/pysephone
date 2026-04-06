"""
Hybrid phenology model with a unimodal chilling stage and a GDD forcing stage.

Architecture:
  1. A learnable optimal chilling temperature ``T_opt`` and a monotone-increasing
     decay network map distance from the optimum to a chilling contribution:
     ``tt_contrib = 1 − decay(|T_norm − T_opt_norm|)``
     which is unimodal in temperature by construction, with peak at ``T_opt``.
  2. Cumulative sum of contributions is soft-thresholded → chilling fulfillment
     mask ``tt_reached`` in [0,1].
  3. Daily GDD (relu(T − t_base)) is weighted by ``tt_reached``.
  4. Cumulative masked GDD is soft-thresholded → per-day bloom probability
     ``ps`` in [0,1].
  5. Predicted day = argmax of first difference of ``ps``.

Training uses binary cross-entropy against a soft step-function label
(0 before the event, 1 from the event day onwards).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysephone.constants import KEY_FEATURES
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs
from pysephone.models.util.func_phenology_torch import TorchGDD
from pysephone.models.util.monotone_cnn import MonotoneCNN
from pysephone.models.util.soft_threshold import SoftThreshold


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class UnimodalHybridModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`UnimodalHybridModel`.

    Attributes:
        temperature_key:    Feature key for temperature (raw, un-normalised).
        feature_statistics: Optional ``{key: (mean, std)}`` for normalisation.
                            ``None`` → use
                            :meth:`~BaseTorchModel.get_default_norm_params`.
        hidden_size:        Hidden channel count of each monotone network.
        num_layers:         Number of layers in each monotone network.
        learn_t_base:       Whether to learn the GDD base temperature.
        t_base_init:        Initial value for the GDD base temperature.
        learn_thresholds:   Whether to learn the soft-threshold locations.
    """
    temperature_key: str = 'temperature_2m_mean'
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    hidden_size: int = 16
    num_layers: int = 3
    learn_t_base: bool = False
    t_base_init: float = 5.0
    learn_thresholds: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UnimodalHybridModel(BaseTorchModel):
    """Hybrid phenology model with unimodal chilling stage (monotone networks).

    The chilling contribution at each timestep is computed as::

        dist(t)        = |T_norm(t) − T_opt_norm|
        tt_contrib(t)  = 1 − decay(dist(t))

    where ``T_opt_norm`` is a learnable optimal chilling temperature (in
    normalised space) and ``decay`` is a monotone-increasing network with
    softplus-constrained weights.  Because ``decay`` increases with distance
    from the optimum, ``tt_contrib`` is bell-shaped (unimodal) in temperature
    by construction, with its peak at ``T_opt``.

    Args:
        temperature_key:    Feature key for temperature.
        feature_statistics: ``{key: (mean, std)}`` for input normalisation.
        hidden_size:        Hidden size of the decay network.
        num_layers:         Number of layers in the decay network.
        learn_t_base:       Learn the GDD base temperature.
        t_base_init:        Initial GDD base temperature.
        learn_thresholds:   Learn the soft-threshold locations.
    """

    def __init__(
        self,
        temperature_key: str = 'temperature_2m_mean',
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden_size: int = 16,
        num_layers: int = 3,
        learn_t_base: bool = False,
        t_base_init: float = 5.0,
        learn_thresholds: bool = True,
    ) -> None:
        super().__init__()

        self._temperature_key = temperature_key
        self._feature_statistics: Dict[str, Tuple[float, float]] = (
            feature_statistics if feature_statistics is not None
            else self.__class__.get_default_norm_params()
        )

        # Learnable optimal chilling temperature (in normalised space)
        self._T_opt_norm = nn.Parameter(torch.tensor(0.0))
        # Monotone-increasing decay: maps distance from T_opt → (0, 1)
        # tt_contrib = 1 - decay(|T_norm - T_opt_norm|) is unimodal by construction
        self._decay = MonotoneCNN(hidden_size=hidden_size, num_layers=num_layers)

        self._tt_unit_threshold = SoftThreshold(
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=200.0,
        )

        self._gdd_threshold = SoftThreshold(
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=500.0,
        )

        self._gdd = TorchGDD(t_base=t_base_init, t_base_req_grad=learn_t_base)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Returns:
            ``(ixs, info)`` where *ixs* is ``(B,)`` predicted day indices and
            *info* contains intermediate series for inspection.
        """
        temp = xs[KEY_FEATURES][self._temperature_key]   # (B, T), raw

        # Normalise temperature
        t_mean, t_std = self._feature_statistics[self._temperature_key]
        temp_norm = torch.nan_to_num((temp - t_mean) / t_std)  # (B, T)

        # Distance from learnable optimal chilling temperature
        dist = (temp_norm - self._T_opt_norm).abs()      # (B, T), ≥ 0

        # Unimodal chilling contribution: 1 − decay(dist)
        # decay is monotone increasing → contribution is bell-shaped around T_opt
        tt_contrib = 1.0 - self._decay(dist.unsqueeze(1))[:, 0, :]  # (B, T)

        # Cumulative chilling → soft threshold → fulfillment mask
        tt_units   = torch.cumsum(tt_contrib, dim=-1)
        tt_reached = self._tt_unit_threshold(tt_units.unsqueeze(1)).squeeze(1)

        # Masked GDD accumulation
        gdd_daily        = F.relu(temp - self._gdd._tb)
        gdd_daily_masked = gdd_daily * tt_reached
        gdd_cum          = torch.cumsum(gdd_daily_masked, dim=-1)

        # Soft threshold on cumulative GDD → bloom probability curve
        ps = self._gdd_threshold(gdd_cum.unsqueeze(1)).squeeze(1)

        # Predicted day: argmax of first difference
        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs  = torch.argmax(diff, dim=-1).clamp(0, ps.size(-1) - 1)

        return ixs.float(), {
            'ps':               ps,
            'tt_contrib':       tt_contrib,
            'tt_units':         tt_units,
            'tt_reached':       tt_reached,
            'gdd_daily':        gdd_daily,
            'gdd_daily_masked': gdd_daily_masked,
            'gdd_cum':          gdd_cum,
        }

    # ------------------------------------------------------------------
    # Loss — BCE with soft step-function labels
    # ------------------------------------------------------------------

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Binary cross-entropy against a soft step-function target."""
        ys_pred, info = self(xs)
        ps = info['ps']

        season_starts = xs['season_start']
        ys_true_list = []
        for i, season_start in enumerate(season_starts):
            sample = {k: v[i] if isinstance(v, (list, torch.Tensor)) else v
                      for k, v in xs.items()}
            target_dt = np.datetime64(target_fn(sample), 'D')
            start_dt  = np.datetime64(season_start, 'D')
            ix = int((target_dt - start_dt) / np.timedelta64(1, 'D'))
            ys_true_list.append(float(ix))

        ys_true = torch.tensor(ys_true_list, dtype=ps.dtype, device=ps.device)
        T = ps.size(-1)
        t_range = torch.arange(T, device=ps.device).unsqueeze(0)
        labels  = (t_range >= ys_true.view(-1, 1)).to(ps.dtype)

        loss = F.binary_cross_entropy(ps, labels)
        return loss, {
            'forward_pass': info,
            'ys_pred':      ys_pred,
            'ys_true':      ys_true,
        }

    # ------------------------------------------------------------------
    # fit_from_args
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset,
        model_args: UnimodalHybridModelArgs,
        model: Optional['UnimodalHybridModel'] = None,
        **kwargs,
    ) -> Tuple['UnimodalHybridModel', Dict[str, Any]]:
        """Fit from a :class:`UnimodalHybridModelArgs` instance."""
        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_args.model_kwargs,
            num_epochs=model_args.num_epochs,
            batch_size=model_args.batch_size,
            val_period=model_args.val_period,
            plot_period=model_args.plot_period,
            scheduler_step_size=model_args.scheduler_step_size,
            scheduler_decay=model_args.scheduler_decay,
            clip_gradient=model_args.clip_gradient,
            optimizer=model_args.optimizer,
            optimizer_kwargs=model_args.optimizer_kwargs,
            early_stopping=model_args.early_stopping,
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_min_delta=model_args.early_stopping_min_delta,
            early_stopping_rerun=model_args.early_stopping_rerun,
            device=torch.device(model_args.device),
            num_workers=model_args.num_workers,
            pin_memory=model_args.pin_memory,
            seed=model_args.seed,
            verbose=model_args.verbose,
            **kwargs,
        )
