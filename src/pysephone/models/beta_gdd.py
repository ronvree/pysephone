"""
Differentiable chilling-forcing model with a beta-distribution chilling
temperature response and GDD forcing.

Architecture:
  1. Raw temperature is normalised to ``[0, 1]`` over a fixed range
     ``[t_low, t_high]``.
  2. The per-day chilling contribution is computed from the (unnormalised) beta
     PDF evaluated at the normalised temperature and rescaled to peak at 1.0::

         log_contrib  = (α − 1) log(T_norm) + (β − 1) log(1 − T_norm)
         mode         = (α − 1) / (α + β − 2)
         log_max      = (α − 1) log(mode) + (β − 1) log(1 − mode)
         tt_contrib   = exp(log_contrib − log_max)  ∈ [0, 1]

     The effective optimal chilling temperature is
     ``T_opt = t_low + (t_high − t_low) × mode``.

  3. Unimodality is enforced by construction: ``α > 1`` and ``β > 1`` are
     guaranteed via ``α = 1 + softplus(raw_α)``.  The beta distribution on
     ``(0, 1)`` is unimodal if and only if both shape parameters exceed 1.

  4. α and β may be **global** learnable scalars (one set for the whole dataset)
     or **contextually predicted** from a per-sample context vector via a small
     MLP — see :class:`CtxBetaGDDModel` and its concrete subclasses.

  5. Cumulative chilling is soft-thresholded → fulfillment mask ``tt_reached``.

  6. Daily GDD ``relu(T − t_base)`` is weighted by ``tt_reached``.

  7. Cumulative masked GDD is soft-thresholded → per-day bloom probability
     ``ps ∈ [0, 1]``.

  8. Predicted day = argmax of the first difference of ``ps``.

Training uses binary cross-entropy against a soft step-function label
(0 before the event, 1 from the event day onwards).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysephone.constants import KEY_DATA_SOURCE, KEY_FEATURES, KEY_SPECIES_ID

_AE_EMBED_DIM: int = 64  # AlphaEarth V1 embedding dimensionality
from pysephone.models.base import ModelException
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs, _OPTIMIZERS
from pysephone.models.util.func_phenology_torch import TorchGDD
from pysephone.models.util.soft_threshold import SoftThreshold


SpeciesKey = Tuple[str, int]

_SOFTPLUS_SHIFT = 0.6931471805599453  # log(2), so softplus(0) = log(2) ≈ 0.693


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_positive(raw: torch.Tensor) -> torch.Tensor:
    """Map raw parameter to (0, ∞) via softplus, centred so _to_positive(0) ≈ 0.693."""
    return F.softplus(raw)


def _raw_alpha_beta_to_shape(raw: torch.Tensor) -> torch.Tensor:
    """Convert raw (unconstrained) value to a beta shape parameter > 1."""
    return 1.0 + _to_positive(raw)


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class BetaGDDModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`GlobalBetaGDDModel`.

    Attributes:
        temperature_key:    Feature key for temperature (raw, un-normalised).
        feature_statistics: Optional ``{key: (mean, std)}`` for normalisation.
                            Not used for the beta computation itself (which
                            operates on raw Celsius values), but kept for API
                            consistency with other hybrid models.
        t_low:              Lower bound of the chilling temperature range (°C).
                            Temperatures at or below this map to 0 in [0, 1].
        t_high:             Upper bound of the chilling temperature range (°C).
        learn_t_base:       Whether to learn the GDD base temperature.
        t_base_init:        Initial GDD base temperature (°C).
        learn_thresholds:   Whether to learn the soft-threshold locations.
        alpha_init:         Initial beta shape parameter α (must be > 1).
        beta_init:          Initial beta shape parameter β (must be > 1).
        learn_alpha_beta:   Whether to learn α and β as global parameters.
        learn_bounds:       Whether to learn t_low and t_high.
        bounds_reg_lambda:  Regularization weight for (t_high - t_low) penalty.
                            Ignored when ``learn_bounds=False``.
        bounds_min_width:   Minimum allowed width ``t_high - t_low`` (°C).
    """
    temperature_key: str = 'temperature_2m_mean'
    feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None
    t_low: float = -5.0
    t_high: float = 20.0
    learn_t_base: bool = True
    t_base_init: float = 5.0
    learn_thresholds: bool = True
    alpha_init: float = 2.0
    beta_init: float = 2.0
    learn_alpha_beta: bool = True
    learn_bounds: bool = False
    bounds_reg_lambda: float = 0.0
    bounds_min_width: float = 5.0
    early_stopping_patience: int = 10


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BetaGDDModel(BaseTorchModel, ABC):
    """Abstract base: differentiable beta-chilling + GDD forcing.

    Subclasses implement :meth:`get_alpha_beta` to supply per-sample (or
    global) shape parameters ``α > 1`` and ``β > 1``.

    Args:
        temperature_key:    Feature key for raw temperature.
        t_low:              Lower bound of the chilling temperature range (°C).
        t_high:             Upper bound of the chilling temperature range (°C).
        learn_t_base:       Learn the GDD base temperature.
        t_base_init:        Initial GDD base temperature (°C).
        learn_thresholds:   Learn the soft-threshold locations.
        learn_bounds:       Learn t_low and t_high by gradient descent.
        bounds_reg_lambda:  L1 regularization weight on (t_high - t_low).
        bounds_min_width:   Minimum allowed width t_high - t_low (°C).
    """

    _EPS: float = 1e-6
    _BOUNDS_STEEPNESS: float = 2.0  # sigmoid steepness for soft boundary mask (°C⁻¹)

    def __init__(
        self,
        temperature_key: str = 'temperature_2m_mean',
        t_low: float = -5.0,
        t_high: float = 20.0,
        learn_t_base: bool = True,
        t_base_init: float = 5.0,
        learn_thresholds: bool = True,
        learn_bounds: bool = False,
        bounds_reg_lambda: float = 0.0,
        bounds_min_width: float = 5.0,
    ) -> None:
        super().__init__()

        self._temperature_key = temperature_key
        self._bounds_reg_lambda = float(bounds_reg_lambda)
        self._bounds_min_width = float(bounds_min_width)

        # t_low is stored unconstrained; t_high = t_low + min_width + softplus(raw_gap)
        self._raw_t_low = nn.Parameter(
            torch.tensor(float(t_low)),
            requires_grad=learn_bounds,
        )
        gap_init = max(float(t_high) - float(t_low) - float(bounds_min_width), 1e-3)
        # invert softplus: raw = log(exp(gap) - 1)
        raw_gap_init = float(np.log(np.exp(gap_init) - 1.0)) if gap_init > 1e-3 else -5.0
        self._raw_t_gap = nn.Parameter(
            torch.tensor(raw_gap_init),
            requires_grad=learn_bounds,
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
    # Learnable bounds
    # ------------------------------------------------------------------

    @property
    def t_low_eff(self) -> torch.Tensor:
        """Effective lower bound of chilling temperature range (scalar tensor)."""
        return self._raw_t_low

    @property
    def t_high_eff(self) -> torch.Tensor:
        """Effective upper bound; always >= t_low_eff + bounds_min_width."""
        return self._raw_t_low + self._bounds_min_width + F.softplus(self._raw_t_gap)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_cf_parameters(self, xs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return per-sample chilling-forcing parameters as a dict.

        Required keys — all tensors of shape ``(B,)``:

        * ``'alpha'``  — beta shape parameter > 1 (unimodality).
        * ``'beta'``   — beta shape parameter > 1 (unimodality).
        * ``'t_low'``  — lower bound of chilling temperature window (°C).
        * ``'t_high'`` — upper bound of chilling temperature window (°C);
                         must satisfy ``t_high > t_low``.

        Use ``_raw_alpha_beta_to_shape(raw)`` to map unconstrained raw values
        to shape parameters > 1.
        """

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _beta_chilling_contrib(
        self,
        temp: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        t_low: torch.Tensor,
        t_high: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-day chilling contributions from beta PDF.

        Args:
            temp:   ``(B, T)`` raw temperature tensor.
            alpha:  ``(B,)`` shape parameter > 1.
            beta:   ``(B,)`` shape parameter > 1.
            t_low:  ``(B,)`` lower bound of chilling range (°C).
            t_high: ``(B,)`` upper bound of chilling range (°C).

        Returns:
            ``(B, T)`` tensor of chilling contributions in ``[0, 1]``,
            peaking at 1.0 at the mode temperature.
        """
        eps = self._EPS
        # Unsqueeze to (B, 1) for broadcasting over time
        t_l = t_low.unsqueeze(1)
        t_h = t_high.unsqueeze(1)
        t_range = t_h - t_l + eps

        # Soft boundary mask: differentiable w.r.t. t_low and t_high, giving
        # gradient signal to push the bounds toward temperatures that improve fit.
        # At k=2 the mask is ~0.007 at 5°C outside the range (effectively zero)
        # but smooth at the boundary — unlike the hard step which has zero gradient.
        k = self._BOUNDS_STEEPNESS
        soft_mask = torch.sigmoid(k * (temp - t_l)) * torch.sigmoid(k * (t_h - temp))

        # Clamp only to keep log() numerically safe
        T_norm = ((temp - t_l) / t_range).clamp(eps, 1.0 - eps)          # (B, T)

        # Expand shape parameters for broadcasting: (B, 1)
        a = alpha.unsqueeze(1)   # (B, 1)
        b = beta.unsqueeze(1)    # (B, 1)

        # Log of unnormalized beta PDF
        log_contrib = (a - 1.0) * torch.log(T_norm) + (b - 1.0) * torch.log(1.0 - T_norm)

        # Mode of beta(α, β) for α > 1, β > 1: (α − 1) / (α + β − 2)
        mode = (a - 1.0) / (a + b - 2.0)                                 # (B, 1)
        mode = mode.clamp(eps, 1.0 - eps)

        # Log PDF at mode (the maximum of the unnormalized PDF)
        log_max = (a - 1.0) * torch.log(mode) + (b - 1.0) * torch.log(1.0 - mode)

        # 1.0 at mode, ~0 at and beyond boundaries
        return torch.exp(log_contrib - log_max) * soft_mask               # (B, T)

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Returns:
            ``(ixs, info)`` where *ixs* is ``(B,)`` predicted day indices and
            *info* contains intermediate series for inspection.
        """
        temp = xs[KEY_FEATURES][self._temperature_key]   # (B, T), raw °C

        cf = self.get_cf_parameters(xs)
        alpha  = cf['alpha']   # (B,) > 1
        beta   = cf['beta']    # (B,) > 1
        t_low  = cf['t_low']   # (B,)
        t_high = cf['t_high']  # (B,)

        # Beta-distribution chilling contribution in [0, 1], peak at mode temp
        tt_contrib = self._beta_chilling_contrib(temp, alpha, beta, t_low, t_high)  # (B, T)

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
            'alpha':            alpha,
            'beta':             beta,
            't_low':            t_low,
            't_high':           t_high,
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
        if self._bounds_reg_lambda > 0.0:
            t_low  = info['t_low']
            t_high = info['t_high']
            loss = loss + self._bounds_reg_lambda * (t_high - t_low).mean()
        return loss, {
            'forward_pass': info,
            'ys_pred':      ys_pred,
            'ys_true':      ys_true,
        }


# ---------------------------------------------------------------------------
# Concrete: global (non-contextual) alpha and beta
# ---------------------------------------------------------------------------

class GlobalBetaGDDModel(BetaGDDModel):
    """Beta-GDD model with a single global pair of shape parameters.

    Both ``α`` and ``β`` are learnable scalars shared across all samples.
    Unimodality is enforced by parameterising them as
    ``1 + softplus(raw)``, guaranteeing both remain > 1.

    Args:
        alpha_init:       Initial value for α (must be > 1).
        beta_init:        Initial value for β (must be > 1).
        learn_alpha_beta: Whether α and β are learned by gradient descent.
        ...               Remaining kwargs forwarded to :class:`BetaGDDModel`.
    """

    def __init__(
        self,
        temperature_key: str = 'temperature_2m_mean',
        t_low: float = -5.0,
        t_high: float = 20.0,
        learn_t_base: bool = True,
        t_base_init: float = 5.0,
        learn_thresholds: bool = True,
        alpha_init: float = 2.0,
        beta_init: float = 2.0,
        learn_alpha_beta: bool = True,
        learn_bounds: bool = False,
        bounds_reg_lambda: float = 0.0,
        bounds_min_width: float = 5.0,
    ) -> None:
        super().__init__(
            temperature_key=temperature_key,
            t_low=t_low,
            t_high=t_high,
            learn_t_base=learn_t_base,
            t_base_init=t_base_init,
            learn_thresholds=learn_thresholds,
            learn_bounds=learn_bounds,
            bounds_reg_lambda=bounds_reg_lambda,
            bounds_min_width=bounds_min_width,
        )
        if alpha_init <= 1.0 or beta_init <= 1.0:
            raise ValueError(
                f'alpha_init and beta_init must be > 1 for unimodality; '
                f'got alpha_init={alpha_init}, beta_init={beta_init}'
            )

        # Invert softplus so that 1 + softplus(raw) == init value
        def _inv_softplus_plus1(v: float) -> float:
            u = v - 1.0  # > 0
            return float(np.log(np.exp(u) - 1.0))

        self._raw_alpha = nn.Parameter(
            torch.tensor(_inv_softplus_plus1(alpha_init)),
            requires_grad=learn_alpha_beta,
        )
        self._raw_beta = nn.Parameter(
            torch.tensor(_inv_softplus_plus1(beta_init)),
            requires_grad=learn_alpha_beta,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current α value (> 1, unimodality guaranteed)."""
        return _raw_alpha_beta_to_shape(self._raw_alpha)

    @property
    def beta(self) -> torch.Tensor:
        """Current β value (> 1, unimodality guaranteed)."""
        return _raw_alpha_beta_to_shape(self._raw_beta)

    def get_cf_parameters(self, xs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        B = xs[KEY_FEATURES][self._temperature_key].size(0)
        return {
            'alpha':  self.alpha.expand(B),
            'beta':   self.beta.expand(B),
            't_low':  self.t_low_eff.expand(B),
            't_high': self.t_high_eff.expand(B),
        }

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset,
        model_args: BetaGDDModelArgs,
        model: Optional['GlobalBetaGDDModel'] = None,
        **kwargs,
    ) -> Tuple['GlobalBetaGDDModel', Dict[str, Any]]:
        """Fit from a :class:`BetaGDDModelArgs` instance."""
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


# ---------------------------------------------------------------------------
# Abstract base: contextually-parameterized alpha and beta
# ---------------------------------------------------------------------------

@dataclass
class CtxBetaGDDModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`CtxBetaGDDModel` subclasses.

    Attributes:
        temperature_key:  Feature key for temperature (raw, un-normalised).
        t_low:            Lower bound of the chilling temperature range (°C).
                          Defaults to 1.4 °C — the Utah model's lower positive-
                          chilling threshold.
        t_high:           Upper bound of the chilling temperature range (°C).
                          Defaults to 15.9 °C — the Utah model's upper positive-
                          chilling threshold.
        learn_t_base:     Whether to learn the GDD base temperature.
        t_base_init:      Initial GDD base temperature (°C).
        learn_thresholds: Whether to learn the soft-threshold locations.
        ctx_hidden:       Hidden size of the context MLP that predicts α, β.
        ctx_reg_lambda:   L2 ridge penalty weight on context MLP parameters.
                          Added to the BCE loss each forward pass.  Set to 0
                          to disable (default).
    """
    temperature_key: str = 'temperature_2m_mean'
    t_low: float = 1.4    # Utah model lower positive-chilling threshold
    t_high: float = 15.9  # Utah model upper positive-chilling threshold
    learn_t_base: bool = True
    t_base_init: float = 5.0
    learn_thresholds: bool = True
    ctx_hidden: int = 16
    ctx_reg_lambda: float = 0.0
    early_stopping_patience: int = 10


class CtxBetaGDDModel(BetaGDDModel, ABC):
    """Beta-GDD where α and β are predicted per-sample from a context vector.

    Temperature bounds ``[t_low, t_high]`` are fixed (default: Utah-model-
    derived 1.4 – 15.9 °C).  A small two-layer MLP maps a ``(B, ctx_dim)``
    context tensor to ``(raw_α, raw_β)``, and then
    ``α = 1 + softplus(raw_α)`` enforces unimodality.

    Weight decay is disabled for the context MLP parameters to prevent α/β
    from collapsing toward 1 during training.

    Subclasses implement :meth:`get_context_vectors`.

    Args:
        ctx_dim:          Dimensionality of the per-sample context vector.
        ctx_hidden:       Hidden size of the context MLP.
        ctx_reg_lambda:   L2 ridge penalty on context MLP parameters (default 0).
        ...               Remaining kwargs forwarded to :class:`BetaGDDModel`.
    """

    def __init__(
        self,
        ctx_dim: int,
        ctx_hidden: int = 16,
        ctx_reg_lambda: float = 0.0,
        temperature_key: str = 'temperature_2m_mean',
        t_low: float = 1.4,
        t_high: float = 15.9,
        learn_t_base: bool = True,
        t_base_init: float = 5.0,
        learn_thresholds: bool = True,
    ) -> None:
        if ctx_dim <= 0:
            raise ValueError(f'ctx_dim must be > 0, got {ctx_dim}')
        self._ctx_reg_lambda = float(ctx_reg_lambda)
        super().__init__(
            temperature_key=temperature_key,
            t_low=t_low,
            t_high=t_high,
            learn_t_base=learn_t_base,
            t_base_init=t_base_init,
            learn_thresholds=learn_thresholds,
            learn_bounds=False,  # bounds are fixed (Utah-model-derived)
        )
        self._ctx_dim = int(ctx_dim)

        # MLP: context → (raw_α, raw_β); bounds are fixed above
        self._ctx_net = nn.Sequential(
            nn.Linear(ctx_dim, ctx_hidden),
            nn.ReLU(),
            nn.Linear(ctx_hidden, 2),
        )

        # Initialise so MLP starts at α=β≈2.0 (unimodal, non-degenerate).
        # inv_softplus(1.0) = log(e − 1) ≈ 0.541
        _ab_init = float(np.log(np.exp(1.0) - 1.0))
        with torch.no_grad():
            self._ctx_net[-1].bias[0].fill_(_ab_init)
            self._ctx_net[-1].bias[1].fill_(_ab_init)

    @abstractmethod
    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        """Return per-sample context tensor of shape ``(B, ctx_dim)``."""

    def get_cf_parameters(self, xs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        p = next(self.parameters())
        ctx = self.get_context_vectors(xs).to(dtype=p.dtype, device=p.device)
        raw = self._ctx_net(ctx)                    # (B, 2)

        alpha = _raw_alpha_beta_to_shape(raw[:, 0]) # (B,), > 1
        beta  = _raw_alpha_beta_to_shape(raw[:, 1]) # (B,), > 1
        B = raw.size(0)
        return {
            'alpha':  alpha,
            'beta':   beta,
            't_low':  self.t_low_eff.expand(B),     # fixed Utah-derived bounds
            't_high': self.t_high_eff.expand(B),
        }

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss, info = super().loss(xs, target_fn)
        if self._ctx_reg_lambda > 0.0:
            ridge = sum(p.pow(2).sum() for p in self._ctx_net.parameters())
            loss = loss + self._ctx_reg_lambda * ridge
        return loss, info

    @classmethod
    def _make_optimizer(
        cls, model: 'BaseTorchModel', name: str, kwargs: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        if name not in _OPTIMIZERS:
            raise ModelException(
                f"Unknown optimizer {name!r}. Choose from: {list(_OPTIMIZERS)}"
            )
        # Zero out weight decay for the context MLP to prevent α/β collapse.
        ctx_ids = {id(p) for p in model._ctx_net.parameters()}
        physics_params = [p for p in model.parameters() if id(p) not in ctx_ids]
        ctx_params     = [p for p in model.parameters() if id(p) in ctx_ids]
        return _OPTIMIZERS[name]([
            {'params': physics_params, **kwargs},
            {'params': ctx_params, **{**kwargs, 'weight_decay': 0.0}},
        ])

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset,
        model_args: CtxBetaGDDModelArgs,
        model: Optional['CtxBetaGDDModel'] = None,
        **kwargs,
    ) -> Tuple['CtxBetaGDDModel', Dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Concrete contextual: one-hot species encoding
# ---------------------------------------------------------------------------

class OneHotSpeciesBetaGDDModel(CtxBetaGDDModel):
    """Beta-GDD with per-species chilling bounds predicted from a one-hot species context.

    Each unique ``(src, species_id)`` pair is encoded as a one-hot vector.
    The context MLP maps this to per-species ``(t_low, t_high)`` temperature
    bounds.  Shape parameters α and β are shared global scalars.

    Args:
        species_keys: Ordered list of ``(src, species_id)`` tuples.
        unknown:      ``'zero'`` (emit zero context, falls back to network
                      bias) or ``'error'`` (raise :class:`KeyError`).
        ctx_hidden:   Hidden size of the context MLP.
        ...           Remaining kwargs forwarded to :class:`CtxBetaGDDModel`.
    """

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        unknown: str = 'zero',
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if not species_keys:
            raise ValueError('species_keys must be non-empty')

        super().__init__(
            ctx_dim=len(species_keys),
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )
        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> 'OneHotSpeciesBetaGDDModel':
        """Build from the unique ``(src, species_id)`` pairs in *dataset*."""
        seen: Dict[SpeciesKey, None] = {}
        for ix in dataset.iter_index():
            key = (str(ix[0]), int(ix[3]))
            if key not in seen:
                seen[key] = None
        species_keys = sorted(seen)
        return cls(
            species_keys=species_keys,
            unknown=unknown,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = next(self.parameters()).device

        ctx = torch.zeros(len(srcs), self._ctx_dim, device=device)
        for i, (s, sid) in enumerate(zip(srcs, sids)):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in species_keys '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
            else:
                ctx[i, row] = 1.0
        return ctx


# ---------------------------------------------------------------------------
# Concrete contextual: phylogenetic MDS embedding
# ---------------------------------------------------------------------------

class PhylogeneticBetaGDDModel(CtxBetaGDDModel):
    """Beta-GDD with per-species α, β predicted from a phylogenetic embedding.

    Each ``(src, species_id)`` pair is represented by a row of MDS coordinates
    (typically derived from a phylogenetic distance matrix).  Coordinates are
    z-score normalised per dimension using statistics computed from *mds_coords*
    before being passed to the context MLP.

    Args:
        species_keys: Ordered list of ``(src, species_id)`` tuples — one per
                      row of *mds_coords*.
        mds_coords:   ``(N_species, k)`` float array of MDS coordinates.
        unknown:      ``'zero'`` or ``'error'`` (same semantics as
                      :class:`OneHotSpeciesBetaGDDModel`).
        ctx_hidden:   Hidden size of the context MLP.
        ...           Remaining kwargs forwarded to :class:`CtxBetaGDDModel`.
    """

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        mds_coords: np.ndarray,
        unknown: str = 'zero',
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if len(species_keys) != mds_coords.shape[0]:
            raise ValueError(
                f'species_keys ({len(species_keys)}) and mds_coords '
                f'({mds_coords.shape[0]}) row count must match'
            )

        ctx_dim = int(mds_coords.shape[1])
        super().__init__(
            ctx_dim=ctx_dim,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )
        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }
        mds = np.asarray(mds_coords, dtype=np.float32)
        mds_mean = mds.mean(axis=0)
        mds_std  = mds.std(axis=0)
        mds_std[mds_std < 1e-8] = 1.0   # avoid division by zero for constant dims
        self.register_buffer('_mds_table', torch.from_numpy(mds))
        self.register_buffer('_mds_mean',  torch.from_numpy(mds_mean))
        self.register_buffer('_mds_std',   torch.from_numpy(mds_std))

    @classmethod
    def from_phylogeny_features(
        cls,
        phylo,
        unknown: str = 'zero',
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> 'PhylogeneticBetaGDDModel':
        """Build from a fitted ``PhylogenyFeatures`` instance."""
        if phylo.mds_coords is None:
            raise ValueError(
                'PhylogenyFeatures must be fitted (with "mds" output) '
                'before constructing a PhylogeneticBetaGDDModel.'
            )
        return cls(
            species_keys=list(phylo.species_keys),
            mds_coords=np.asarray(phylo.mds_coords),
            unknown=unknown,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = self._mds_table.device

        rows = []
        for s, sid in zip(srcs, sids):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in fitted phylogeny '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
                rows.append(None)
            else:
                rows.append(row)

        ctx = torch.zeros(len(rows), self._ctx_dim, device=device)
        for i, r in enumerate(rows):
            if r is not None:
                ctx[i] = self._mds_table[r]
        return (ctx - self._mds_mean) / self._mds_std


# ---------------------------------------------------------------------------
# Concrete contextual: AlphaEarth satellite embedding
# ---------------------------------------------------------------------------

class AlphaEarthBetaGDDModel(CtxBetaGDDModel):
    """Beta-GDD with α, β predicted from an AlphaEarth satellite embedding.

    The 64-D annual satellite embedding is read from
    ``xs['features'][alphaearth_key]`` (shape ``(B, 64)``), which is
    produced by :class:`~pysephone.dataset.util.alphaearth.AlphaEarthFeatures`
    when added to the dataset.  Because the embedding is site- and year-specific
    rather than species-specific, the chilling curve adapts to local habitat
    rather than to taxonomy.

    Embeddings are z-score normalised using *ae_mean* and *ae_std* before being
    passed to the context MLP.  These statistics should be computed from the
    training set (e.g. per-dimension mean and std over all training samples).

    Args:
        ae_mean:        ``(alphaearth_dim,)`` array of per-dimension means.
        ae_std:         ``(alphaearth_dim,)`` array of per-dimension std devs.
        alphaearth_key: Feature key under which the embedding is stored
                        (default: ``'alphaearth_embedding'``).
        alphaearth_dim: Dimensionality of the embedding (default: 64).
        ctx_hidden:     Hidden size of the context MLP.
        ...             Remaining kwargs forwarded to :class:`CtxBetaGDDModel`.
    """

    def __init__(
        self,
        ae_mean: np.ndarray,
        ae_std: np.ndarray,
        alphaearth_key: str = 'alphaearth_embedding',
        alphaearth_dim: int = _AE_EMBED_DIM,
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(
            ctx_dim=alphaearth_dim,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )
        self._alphaearth_key = alphaearth_key
        ae_std = np.asarray(ae_std, dtype=np.float32).copy()
        ae_std[ae_std < 1e-8] = 1.0
        self.register_buffer('_ae_mean', torch.from_numpy(np.asarray(ae_mean, dtype=np.float32)))
        self.register_buffer('_ae_std',  torch.from_numpy(ae_std))

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        emb = xs[KEY_FEATURES][self._alphaearth_key]   # (B, 64)
        emb = torch.nan_to_num(emb, nan=0.0).to(dtype=self._ae_mean.dtype, device=self._ae_mean.device)
        return (emb - self._ae_mean) / self._ae_std


# ---------------------------------------------------------------------------
# Concrete contextual: phylogenetic embedding + AlphaEarth concatenated
# ---------------------------------------------------------------------------

class PhyloAlphaEarthBetaGDDModel(CtxBetaGDDModel):
    """Beta-GDD with α, β predicted from phylogenetic + AlphaEarth context.

    The context vector is the concatenation of:

    - A per-species phylogenetic MDS embedding (species-level, from a
      pre-computed distance matrix), and
    - A per-site AlphaEarth satellite embedding (location- and year-level).

    Each part is z-score normalised independently before concatenation:
    MDS statistics are computed from *mds_coords*; AEE statistics must be
    passed in explicitly.

    Args:
        species_keys:   Ordered list of ``(src, species_id)`` tuples — one per
                        row of *mds_coords*.
        mds_coords:     ``(N_species, k)`` float array of MDS coordinates.
        ae_mean:        ``(alphaearth_dim,)`` per-dimension AEE means.
        ae_std:         ``(alphaearth_dim,)`` per-dimension AEE std devs.
        unknown:        ``'zero'`` or ``'error'`` for unseen species.
        alphaearth_key: Feature key for the AlphaEarth embedding
                        (default: ``'alphaearth_embedding'``).
        alphaearth_dim: Dimensionality of the AlphaEarth embedding (default: 64).
        ctx_hidden:     Hidden size of the context MLP.
        ...             Remaining kwargs forwarded to :class:`CtxBetaGDDModel`.
    """

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        mds_coords: np.ndarray,
        ae_mean: np.ndarray,
        ae_std: np.ndarray,
        unknown: str = 'zero',
        alphaearth_key: str = 'alphaearth_embedding',
        alphaearth_dim: int = _AE_EMBED_DIM,
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if len(species_keys) != mds_coords.shape[0]:
            raise ValueError(
                f'species_keys ({len(species_keys)}) and mds_coords '
                f'({mds_coords.shape[0]}) row count must match'
            )

        mds_dim = int(mds_coords.shape[1])
        super().__init__(
            ctx_dim=mds_dim + alphaearth_dim,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )
        self._unknown = unknown
        self._mds_dim = mds_dim
        self._alphaearth_key = alphaearth_key
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }
        mds = np.asarray(mds_coords, dtype=np.float32)
        mds_mean = mds.mean(axis=0)
        mds_std  = mds.std(axis=0)
        mds_std[mds_std < 1e-8] = 1.0
        ae_std_arr = np.asarray(ae_std, dtype=np.float32).copy()
        ae_std_arr[ae_std_arr < 1e-8] = 1.0
        self.register_buffer('_mds_table', torch.from_numpy(mds))
        self.register_buffer('_mds_mean',  torch.from_numpy(mds_mean))
        self.register_buffer('_mds_std',   torch.from_numpy(mds_std))
        self.register_buffer('_ae_mean',   torch.from_numpy(np.asarray(ae_mean, dtype=np.float32)))
        self.register_buffer('_ae_std',    torch.from_numpy(ae_std_arr))

    @classmethod
    def from_phylogeny_features(
        cls,
        phylo,
        ae_mean: np.ndarray,
        ae_std: np.ndarray,
        unknown: str = 'zero',
        alphaearth_key: str = 'alphaearth_embedding',
        alphaearth_dim: int = _AE_EMBED_DIM,
        ctx_hidden: int = 16,
        **base_kwargs: Any,
    ) -> 'PhyloAlphaEarthBetaGDDModel':
        """Build from a fitted ``PhylogenyFeatures`` instance."""
        if phylo.mds_coords is None:
            raise ValueError(
                'PhylogenyFeatures must be fitted (with "mds" output) '
                'before constructing a PhyloAlphaEarthBetaGDDModel.'
            )
        return cls(
            species_keys=list(phylo.species_keys),
            mds_coords=np.asarray(phylo.mds_coords),
            ae_mean=ae_mean,
            ae_std=ae_std,
            unknown=unknown,
            alphaearth_key=alphaearth_key,
            alphaearth_dim=alphaearth_dim,
            ctx_hidden=ctx_hidden,
            **base_kwargs,
        )

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = self._mds_table.device

        # Phylogenetic part: look up MDS row, then z-score normalise
        phylo_ctx = torch.zeros(len(srcs), self._mds_dim, device=device)
        for i, (s, sid) in enumerate(zip(srcs, sids)):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in fitted phylogeny '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
            else:
                phylo_ctx[i] = self._mds_table[row]
        phylo_ctx = (phylo_ctx - self._mds_mean) / self._mds_std

        # AlphaEarth part: read from features, z-score normalise
        ae_ctx = torch.nan_to_num(
            xs[KEY_FEATURES][self._alphaearth_key].to(device=device),
            nan=0.0,
        )
        ae_ctx = (ae_ctx - self._ae_mean) / self._ae_std

        return torch.cat([phylo_ctx, ae_ctx], dim=-1)   # (B, mds_dim + ae_dim)
