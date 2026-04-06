"""
Hybrid wheat phenology model — learned analog of PVTT.

Architecture:
  A single TTCNN with two output channels processes daily weather features.
  Each channel's output is cumulated (from sowing) and soft-thresholded.
  The product of the two soft-thresholded accumulators gives the heading
  probability curve:

      contributions  = TTCNN(features)          # (B, 2, T), in [0, 1]
      contributions *= sow_mask                  # zero before sowing
      cumsum_0       = cumsum(channel_0)
      cumsum_1       = cumsum(channel_1)
      ps             = SoftThreshold(cumsum_0) × SoftThreshold(cumsum_1)
      predicted_day  = argmax(first_diff(ps))

  The product means *both* stages must be satisfied before the heading
  probability rises — the same logical structure as PVTT where vernalization
  must be complete before PVTT can accumulate.  The TTCNN learns what each
  channel should capture from the input features (temperature, daylight).

Training: BCE against a soft step-function label, same as UnimodalHybridModel.
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
from pysephone.models.util.soft_threshold import SoftThreshold
from pysephone.models.util.ttcnn import TTCNN
from pysephone.utils.func_torch import create_left_mask


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class WheatHybridModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`WheatHybridModel`.

    Attributes:
        data_keys:          Feature keys fed to the TTCNN.
        sow_key:            Observation-index key for the sowing date (BBCH_0).
        feature_statistics: Optional ``{key: (mean, std)}`` for normalisation.
        hidden_size:        TTCNN hidden channel count.
        kernel_size:        TTCNN causal convolution kernel size.
        num_layers:         Number of TTCNN layers.
        use_dilations:      Use exponentially growing dilations in TTCNN.
        learn_thresholds:   Whether to learn the soft-threshold locations.
    """
    data_keys: List[str] = field(
        default_factory=lambda: ['temperature_2m_mean', 'daylight_duration']
    )
    sow_key: str = 'BBCH_0'
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    hidden_size: int = 32
    kernel_size: int = 7
    num_layers: int = 4
    use_dilations: bool = False
    learn_thresholds: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WheatHybridModel(BaseTorchModel):
    """Hybrid wheat phenology model — learned analog of PVTT.

    Uses a single TTCNN with two output channels.  Both channels are
    cumulated from the sowing date and soft-thresholded; the product of the
    two thresholded accumulators forms the heading probability:

        ps = SoftThreshold(cumsum_0) × SoftThreshold(cumsum_1)

    This encodes the two-stage requirement (vernalization + forcing) as an
    architectural constraint rather than a hard-coded functional form.

    Args:
        data_keys:          Feature keys from ``sample['features']``.
        sow_key:            Observation-index key for the sowing date.
        feature_statistics: ``{key: (mean, std)}`` for input normalisation.
        hidden_size:        TTCNN hidden channel count.
        kernel_size:        TTCNN causal convolution kernel size.
        num_layers:         Number of TTCNN layers.
        use_dilations:      Use exponentially growing dilations in TTCNN.
        learn_thresholds:   Learn the soft-threshold locations.
    """

    def __init__(
        self,
        data_keys: List[str] = None,
        sow_key: str = 'BBCH_0',
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden_size: int = 32,
        kernel_size: int = 7,
        num_layers: int = 4,
        use_dilations: bool = False,
        learn_thresholds: bool = True,
    ) -> None:
        super().__init__()

        self._data_keys = list(data_keys or ['temperature_2m_mean', 'daylight_duration'])
        self._sow_key = sow_key
        self._feature_statistics: Dict[str, Tuple[float, float]] = (
            feature_statistics if feature_statistics is not None
            else self.__class__.get_default_norm_params()
        )

        self._ttcnn = TTCNN(
            num_channels_in=len(self._data_keys),
            num_channels_out=2,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            num_layers=num_layers,
            final_activation=nn.Sigmoid(),
            use_dilations=use_dilations,
        )

        # Each channel accumulates up to ~365 units; normalise in [0, 365]
        self._th0 = SoftThreshold(
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=365.0,
        )
        self._th1 = SoftThreshold(
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=365.0,
        )

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _build_feature_tensor(self, xs: Dict[str, Any]) -> torch.Tensor:
        """Return ``(B, C, T)`` normalised feature tensor for the TTCNN."""
        features = xs[KEY_FEATURES]
        fs = []
        for k in self._data_keys:
            mean, std = self._feature_statistics[k]
            fs.append((features[k] - mean) / std)
        x = torch.stack(fs, dim=-1)  # (B, T, C)
        x = torch.nan_to_num(x)
        return x.permute(0, 2, 1)   # (B, C, T)

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
        feats_bct = self._build_feature_tensor(xs)     # (B, C, T)
        B, _, T   = feats_bct.shape

        # Sowing mask: zero contributions before BBCH_0
        sow_ix   = xs[KEY_OBSERVATIONS_INDEX][self._sow_key].long()  # (B,)
        sow_mask = create_left_mask(T, sow_ix).float()               # (B, T)

        # TTCNN → per-day contributions for two channels, in [0, 1]
        contrib = self._ttcnn(feats_bct)                # (B, 2, T)
        c0 = contrib[:, 0, :] * sow_mask               # (B, T)
        c1 = contrib[:, 1, :] * sow_mask               # (B, T)

        # Cumulative accumulation
        cs0 = torch.cumsum(c0, dim=-1)                 # (B, T)
        cs1 = torch.cumsum(c1, dim=-1)                 # (B, T)

        # Product of soft-thresholds → heading probability
        ps0 = self._th0(cs0.unsqueeze(1)).squeeze(1)   # (B, T)
        ps1 = self._th1(cs1.unsqueeze(1)).squeeze(1)   # (B, T)
        ps  = ps0 * ps1                                # (B, T)

        # Predicted day: argmax of first difference
        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs  = torch.argmax(diff, dim=-1).clamp(0, T - 1)

        return ixs.float(), {
            'ps':       ps,
            'ps0':      ps0,
            'ps1':      ps1,
            'c0':       c0,
            'c1':       c1,
            'cs0':      cs0,
            'cs1':      cs1,
            'sow_mask': sow_mask,
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
        model_args: WheatHybridModelArgs,
        model: Optional['WheatHybridModel'] = None,
        **kwargs,
    ) -> Tuple['WheatHybridModel', Dict[str, Any]]:
        """Fit from a :class:`WheatHybridModelArgs` instance."""
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
