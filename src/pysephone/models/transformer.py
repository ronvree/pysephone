"""
Transformer-based phenology model.

Self-attention counterpart to :class:`pysephone.models.lstm.LSTMModel`.  A full
season of daily meteorological features is projected to a model dimension,
combined with sinusoidal positional encoding, processed by a stack of
TransformerEncoder layers with a causal attention mask, and decoded through a
pointwise linear head into a per-day sigmoid probability curve.  The predicted
event day is the day with the largest increase in probability (i.e. the argmax
of the first difference).

The causal mask is essential: it forces the probability at day ``t`` to depend
only on inputs at days ``<= t``, which is what makes the soft step-function
BCE label and the first-difference prediction rule meaningful.  Without it,
day ``t`` could attend to future weather and the per-day probability would
no longer be interpretable as "event has occurred by day t given observations
so far".

Training uses binary cross-entropy against a soft step-function label: 0 before
the observed event day, 1 from that day onwards.

Example::

    from pysephone.models.transformer import TransformerModel

    model, info = TransformerModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(
            data_keys=['temperature_2m_mean'],
            hidden_size=64,
            num_layers=2,
            nhead=4,
            dim_feedforward=128,
        ),
        num_epochs=50,
        batch_size=32,
        val_period=5,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs
from pysephone.models.util.pointwise_head import PointwiseHead
from pysephone.utils.func_torch import create_left_mask


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Added to the input embedding before the encoder.  No learnable parameters.
    The buffer is sized to ``max_len`` and the forward pass slices it to the
    actual sequence length, so the same module handles variable-length seasons.
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) so it broadcasts over the batch
        self.register_buffer('_pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self._pe[:, : x.size(1), :]


@dataclass
class TransformerModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`TransformerModel`.

    Attributes:
        data_keys:           Weather variable names to use as input features.
        hidden_size:         Model dimension (``d_model``).  Must be divisible
                             by ``nhead``.
        num_layers:          Number of stacked TransformerEncoder layers.
        nhead:               Number of self-attention heads per layer.
        dim_feedforward:     Hidden dimension of the per-position feedforward
                             sub-layer inside each encoder block.
        dropout:             Dropout probability inside the encoder layers.
        num_layers_lin:      Depth of the pointwise linear head (>= 1).
        max_len:             Maximum supported sequence length for the
                             positional encoding buffer.
        feature_statistics:  Optional ``{key: (mean, std)}`` dict for input
                             normalisation.  ``None`` -> use
                             :meth:`~BaseTorchModel.get_default_norm_params`.
        obs_features:        Optional list of observation-index keys whose
                             within-season index is encoded as a binary mask
                             feature (1 from that day onwards, 0 before).
    """
    data_keys: List[str] = field(default_factory=lambda: ['temperature_2m_mean'])
    hidden_size: int = 64
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_layers_lin: int = 2
    max_len: int = 1024
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    obs_features: Optional[List[str]] = None


class TransformerModel(BaseTorchModel):
    """Transformer phenology model (causal self-attention).

    Architecture:

    * **Input**: per-day meteorological features (plus optional binary mask
      features for observed prior events), normalised with *feature_statistics*.
    * **Embedding**: linear projection of input channels to ``hidden_size``,
      plus sinusoidal positional encoding.
    * **Encoder**: stack of ``num_layers`` ``TransformerEncoderLayer`` blocks
      with ``nhead`` self-attention heads, ``dim_feedforward``-wide
      feedforward sub-layer, and a causal attention mask that prevents day
      ``t`` from attending to days ``> t``.
    * **Head**: pointwise linear stack (``num_layers_lin`` 1x1 convs) mapping
      encoder outputs to a single sigmoid probability per day.
    * **Prediction**: day with the largest first-difference in the probability
      curve (argmax of ``p[t] - p[t-1]``).

    Args:
        data_keys:          Feature keys from ``sample['features']``.
        hidden_size:        Model dimension (``d_model``).
        num_layers:         Number of stacked encoder blocks.
        nhead:              Number of self-attention heads per block.
        dim_feedforward:    Feedforward sub-layer hidden dimension.
        dropout:            Dropout probability inside encoder layers.
        num_layers_lin:     Depth of the pointwise linear head (>= 1).
        max_len:            Max sequence length for the positional encoding
                            buffer (larger seasons require a larger value).
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
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_layers_lin: int = 2,
        max_len: int = 1024,
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        obs_features: Optional[List[str]] = None,
    ) -> None:
        assert hidden_size > 0
        assert num_layers > 0
        assert nhead > 0
        assert dim_feedforward > 0
        assert num_layers_lin > 0
        assert hidden_size % nhead == 0, (
            f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})"
        )

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

        self._embed = nn.Linear(num_input_features, hidden_size)
        self._pos_enc = _SinusoidalPositionalEncoding(hidden_size, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

        # Embed -> (B, T, hidden_size), add positional encoding
        x = self._embed(x)
        x = self._pos_enc(x)

        # Causal mask: position t cannot attend to positions > t
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self._encoder(x, mask=causal_mask, is_causal=True)

        # Pointwise head: (B, hidden_size, T) -> (B, 1, T) -> (B, T)
        x = x.permute(0, 2, 1)
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
