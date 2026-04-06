"""
Hybrid phenology model combining a learned chilling stage (TTCNN) with a
learned forcing stage (GDD with learnable base temperature).

Architecture:
  1. TTCNN processes daily features → per-day chilling contribution in [0,1]
  2. Cumulative sum of contributions is soft-thresholded → chilling fulfillment
     mask ``tt_reached`` in [0,1]
  3. Daily GDD (relu(T - t_base)) is weighted by ``tt_reached``
  4. Cumulative masked GDD is soft-thresholded → per-day bloom probability
     ``ps`` in [0,1]
  5. Predicted day = argmax of first difference of ``ps``

Training uses binary cross-entropy against a soft step-function label
(0 before the event, 1 from the event day onwards) — the same loss as
:class:`~pysephone.models.lstm.LSTMModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs
from pysephone.models.util.func_phenology_torch import TorchGDD
from pysephone.models.util.soft_threshold import SoftThreshold
from pysephone.models.util.ttcnn import TTCNN
from pysephone.utils.func_torch import create_left_mask


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class HybridModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`HybridModel`.

    Attributes:
        data_keys:          Weather variable names used as TTCNN input features.
        temperature_key:    Which feature key to use as raw temperature for GDD.
        obs_features:       Optional list of observation-index keys appended as
                            binary mask features to the TTCNN input.
        feature_statistics: Optional ``{key: (mean, std)}`` for normalisation.
                            ``None`` → use
                            :meth:`~BaseTorchModel.get_default_norm_params`.
        hidden_size:        TTCNN hidden channel count.
        kernel_size:        TTCNN causal convolution kernel size.
        num_layers:         Number of TTCNN layers.
        use_dilations:      Use exponentially growing dilations in TTCNN.
        l1_lambda:          L1 regularisation coefficient applied to TTCNN
                            weights.  ``0.0`` disables it.
    """
    data_keys: List[str] = field(default_factory=lambda: ['temperature_2m_mean'])
    temperature_key: str = 'temperature_2m_mean'
    obs_features: Optional[List[str]] = None
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    hidden_size: int = 16
    kernel_size: int = 7
    num_layers: int = 4
    use_dilations: bool = False
    ttcnn_output_lambda: float = 0.0
    learn_t_base: bool = True
    learn_thresholds: bool = True
    apply_chilling_mask: bool = False
    t_chilling_low: float = 1.4
    t_chilling_high: float = 15.9


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridModel(BaseTorchModel):
    """Hybrid two-stage phenology model (TTCNN chilling + GDD forcing).

    Args:
        data_keys:          Feature keys from ``sample['features']``.
        temperature_key:    Key for raw (un-normalised) temperature used in GDD.
        obs_features:       Observation-index keys to include as binary masks.
        feature_statistics: ``{key: (mean, std)}`` for TTCNN input normalisation.
        hidden_size:        TTCNN hidden channel count.
        kernel_size:        TTCNN causal convolution kernel size.
        num_layers:         Number of TTCNN layers.
        use_dilations:      Use exponentially growing dilations in TTCNN.
    """

    def __init__(
        self,
        data_keys: List[str] = None,
        temperature_key: str = 'temperature_2m_mean',
        obs_features: Optional[List[str]] = None,
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden_size: int = 16,
        kernel_size: int = 7,
        num_layers: int = 4,
        use_dilations: bool = False,
        ttcnn_output_lambda: float = 0.0,
        learn_t_base: bool = True,
        t_base_init: float = 1.0,
        learn_thresholds: bool = True,
        apply_chilling_mask: bool = False,
        t_chilling_low: float = 1.4,
        t_chilling_high: float = 15.9,
    ) -> None:
        super().__init__()

        self._ttcnn_output_lambda = ttcnn_output_lambda
        self._apply_chilling_mask = apply_chilling_mask
        self._t_chilling_low = t_chilling_low
        self._t_chilling_high = t_chilling_high
        self._data_keys = list(data_keys or ['temperature_2m_mean'])
        self._temperature_key = temperature_key
        self._obs_features = obs_features
        self._feature_statistics: Dict[str, Tuple[float, float]] = (
            feature_statistics if feature_statistics is not None
            else self.__class__.get_default_norm_params()
        )

        num_input_features = len(self._data_keys) + (
            0 if obs_features is None else len(obs_features)
        )

        self._ttcnn = TTCNN(
            num_channels_in=num_input_features,
            num_channels_out=1,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            num_layers=num_layers,
            final_activation=nn.Sigmoid(),
            use_dilations=use_dilations,
        )

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
    # Feature assembly
    # ------------------------------------------------------------------

    def _build_feature_tensor(self, xs: Dict[str, Any]) -> torch.Tensor:
        """Return ``(B, C, T)`` normalised feature tensor for the TTCNN."""
        features = xs[KEY_FEATURES]
        features = {
            k: (v - self._feature_statistics[k][0]) / self._feature_statistics[k][1]
            for k, v in features.items()
        }

        fs = [features[k] for k in self._data_keys]

        if self._obs_features is not None:
            obs_index = xs[KEY_OBSERVATIONS_INDEX]
            season_length = fs[0].size(1)
            for key in self._obs_features:
                fs.append(create_left_mask(season_length, obs_index[key].long()).float())

        x = torch.cat([f.unsqueeze(-1) for f in fs], dim=-1)  # (B, T, C)
        x = torch.nan_to_num(x)
        return x.permute(0, 2, 1)                              # (B, C, T)

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
        feats_bct = self._build_feature_tensor(xs)   # (B, C, T)

        # TTCNN → per-day chilling contribution (B, T)
        tt_contrib = self._ttcnn(feats_bct)[:, 0, :]

        # Optional Utah-style output mask: zero contribution outside [t_low, t_high]
        if self._apply_chilling_mask:
            temp_raw = xs[KEY_FEATURES][self._temperature_key]
            in_range = (temp_raw >= self._t_chilling_low) & (temp_raw <= self._t_chilling_high)
            tt_contrib = tt_contrib * in_range.float()

        # Cumulative chilling → soft threshold → fulfillment mask
        tt_units   = torch.cumsum(tt_contrib, dim=-1)
        tt_reached = self._tt_unit_threshold(tt_units.unsqueeze(1)).squeeze(1)

        # Masked GDD accumulation
        temp = xs[KEY_FEATURES][self._temperature_key]        # (B, T), raw
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
    # Loss — BCE with soft step-function labels (same as LSTMModel)
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
        if self._ttcnn_output_lambda > 0.0:
            loss = loss + self._ttcnn_output_lambda * info['tt_contrib'].abs().mean()
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
        model_args: HybridModelArgs,
        model: Optional['HybridModel'] = None,
        **kwargs,
    ) -> Tuple['HybridModel', Dict[str, Any]]:
        """Fit from a :class:`HybridModelArgs` instance."""
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
