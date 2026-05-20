"""
1D-CNN phenology model.

Convolutional counterpart to :class:`pysephone.models.lstm.LSTMModel`.  A full
season of daily meteorological features is encoded by a stack of *causal* 1-D
convolutions and projected through a pointwise linear head to a per-day
sigmoid probability curve.  The predicted event day is the day with the
largest increase in probability (i.e. the argmax of the first difference).

Causal padding keeps the same left-to-right semantics as the LSTM: the
probability at day ``t`` depends only on inputs at days ``<= t``.  This is
what makes the soft step-function BCE label and the first-difference
prediction rule meaningful — a non-causal encoder would let day ``t`` peek
at the future, which would distort the interpretation of the probability
curve.

Training uses binary cross-entropy against a soft step-function label: 0
before the observed event day, 1 from that day onwards (identical to the
LSTM model's training target).

Example::

    from pysephone.models.cnn_1d import CNN1DModel

    model, info = CNN1DModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(
            data_keys=['temperature_2m_mean'],
            hidden_size=64,
            num_layers=4,
            kernel_size=7,
        ),
        num_epochs=50,
        batch_size=32,
        val_period=5,
    )
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
from pysephone.models.util.causal_cnn import CausalConv1d
from pysephone.models.util.pointwise_head import PointwiseHead
from pysephone.utils.func_torch import create_left_mask


@dataclass
class CNN1DModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`CNN1DModel`.

    Attributes:
        data_keys:           Weather variable names to use as input features.
        hidden_size:         Channel count of each causal conv layer.
        num_layers:          Number of stacked causal conv layers.
        kernel_size:         Kernel size of each causal conv layer.
        num_layers_lin:      Depth of the pointwise linear head (>= 1).
        feature_statistics:  Optional ``{key: (mean, std)}`` dict for input
                             normalisation.  ``None`` -> use
                             :meth:`~BaseTorchModel.get_default_norm_params`.
        obs_features:        Optional list of observation-index keys whose
                             within-season index is encoded as a binary mask
                             feature (1 from that day onwards, 0 before).
    """
    data_keys: List[str] = field(default_factory=lambda: ['temperature_2m_mean'])
    hidden_size: int = 64
    num_layers: int = 4
    kernel_size: int = 7
    num_layers_lin: int = 2
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    obs_features: Optional[List[str]] = None


class CNN1DModel(BaseTorchModel):
    """1D-CNN phenology model (causal, undilated).

    Architecture:

    * **Input**: per-day meteorological features (plus optional binary mask
      features for observed prior events), normalised with *feature_statistics*.
    * **Encoder**: stack of ``num_layers`` causal 1-D convolutions
      (``kernel_size``-wide, ``hidden_size`` channels, dilation 1) with ReLU
      activations between layers.  Causal padding ensures the activation at
      day ``t`` depends only on inputs at days ``<= t``.
    * **Head**: pointwise linear stack (``num_layers_lin`` 1x1 convs) mapping
      channel activations to a single sigmoid probability per day.
    * **Prediction**: day with the largest first-difference in the probability
      curve (argmax of ``p[t] - p[t-1]``).

    Args:
        data_keys:          Feature keys from ``sample['features']``.
        hidden_size:        Channel count of each causal conv layer.
        num_layers:         Number of stacked causal conv layers.
        kernel_size:        Kernel size of each causal conv layer.
        num_layers_lin:     Depth of the pointwise linear head (>= 1).
        feature_statistics: ``{key: (mean, std)}`` for input normalisation.
                            ``None`` -> use
                            :meth:`~BaseTorchModel.get_default_norm_params`.
        obs_features:       List of observation-index keys to include as binary
                            mask features.  Each adds one channel to the input.
    """

    def __init__(
        self,
        data_keys: List[str],
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 7,
        num_layers_lin: int = 2,
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        obs_features: Optional[List[str]] = None,
    ) -> None:
        assert hidden_size > 0
        assert num_layers > 0
        assert kernel_size > 0
        assert num_layers_lin > 0

        super().__init__()

        self._data_keys = list(data_keys)
        self._feature_statistics: Dict[str, Tuple[float, float]] = (
            feature_statistics if feature_statistics is not None
            else self.__class__.get_default_norm_params()
        )
        self._obs_features: Optional[List[str]] = obs_features

        num_input_features = len(self._data_keys) + (
            0 if obs_features is None else len(obs_features)
        )

        layers: List[nn.Module] = []
        in_c = num_input_features
        for _ in range(num_layers):
            layers.append(CausalConv1d(in_c, hidden_size, kernel_size, dilation=1))
            layers.append(nn.ReLU())
            in_c = hidden_size
        self._encoder = nn.Sequential(*layers)

        self._lin = PointwiseHead(
            num_layers=num_layers_lin,
            in_size=hidden_size,
            hidden_size=hidden_size,
            out_size=1,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            xs: Collated batch dict from :meth:`~BaseTorchModel.collate_fn`.

        Returns:
            ``(ixs, {'ps': ps})`` where *ixs* is ``(B,)`` predicted day
            indices (float) and *ps* is ``(B, T)`` per-day sigmoid
            probabilities.
        """
        features: Dict[str, torch.Tensor] = xs[KEY_FEATURES]

        features = {
            k: (v - self._feature_statistics[k][0]) / self._feature_statistics[k][1]
            for k, v in features.items()
        }

        fs: List[torch.Tensor] = [features[k] for k in self._data_keys]

        if self._obs_features is not None:
            obs_index: Dict[str, torch.Tensor] = xs[KEY_OBSERVATIONS_INDEX]
            season_length = fs[0].size(1)
            for key in self._obs_features:
                obs_ixs = obs_index[key].long()
                fs.append(create_left_mask(season_length, obs_ixs).float())

        # (B, T, num_features)
        x = torch.cat([f.unsqueeze(-1) for f in fs], dim=-1)
        x = torch.nan_to_num(x)

        # Causal conv encoder expects channels-first: (B, num_features, T) -> (B, hidden_size, T)
        x = x.permute(0, 2, 1)
        x = self._encoder(x)

        # Pointwise head: (B, hidden_size, T) -> (B, 1, T) -> (B, T)
        ps = torch.sigmoid(self._lin(x)).squeeze(1)

        # Predicted day: argmax of first difference
        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs = torch.argmax(diff, dim=-1).clamp(0, ps.size(-1) - 1)

        return ixs.float(), {'ps': ps}

    # ------------------------------------------------------------------
    # Loss - BCE with soft step-function labels
    # ------------------------------------------------------------------

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Binary cross-entropy against a soft step-function target.

        The target for day *t* is 1 if ``t >= observed_event_day``, else 0.
        """
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
        labels = (t_range >= ys_true.view(-1, 1)).to(ps.dtype)

        loss = F.binary_cross_entropy(ps, labels)

        return loss, {
            'forward_pass': info,
            'ys_pred': ys_pred,
            'ys_true': ys_true,
        }
