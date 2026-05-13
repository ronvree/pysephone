"""
Differentiable chilling-forcing model with a **cubic B-spline** chilling
temperature response and GDD forcing.

Mirrors :mod:`pysephone.models.beta_gdd`, but the per-day chilling contribution
is parameterised as a non-negative linear combination of cardinal cubic
B-spline basis functions placed on a uniform knot grid over ``[t_low, t_high]``::

    f(T) = (1/peak) · soft_mask(T) · Σ_i softplus(c_i) · N_i(T)

where ``N_i`` is the cardinal cubic B-spline basis function centred at the
``i``-th uniform knot and ``c_i`` is a learnable control point.  Non-negativity
is enforced by ``softplus``; ``peak`` rescales so the curve peaks at 1.0; and
a soft sigmoid mask suppresses the response outside the temperature window.

The forcing stage is identical to :mod:`beta_gdd` — cumulative
``relu(T − t_base)`` after the chilling soft-threshold opens — and training
uses BCE against soft step-function labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_FEATURES,
    KEY_LOC_ID,
    KEY_SPECIES_ID,
)
from pysephone.models.base import ModelException
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs, _OPTIMIZERS
from pysephone.models.util.func_phenology_torch import TorchGDD
from pysephone.models.util.soft_threshold import SoftThreshold


SpeciesKey = Tuple[str, int]
LocationKey = Tuple[str, str]   # (src, str(loc_id)) — loc_ids may be ints or station codes
_AE_EMBED_DIM: int = 64  # AlphaEarth V1 embedding dimensionality


# ---------------------------------------------------------------------------
# Cardinal cubic B-spline basis
# ---------------------------------------------------------------------------

def _cardinal_cubic_bspline(u: torch.Tensor) -> torch.Tensor:
    """Cardinal cubic B-spline basis function ``N(u)`` supported on ``[0, 4]``.

    Evaluated piecewise:

    * ``N(u) = u^3 / 6``                                             for 0 ≤ u < 1
    * ``N(u) = (-3u^3 + 12u^2 − 12u + 4) / 6``                       for 1 ≤ u < 2
    * ``N(u) = (3u^3 − 24u^2 + 60u − 44) / 6``                       for 2 ≤ u < 3
    * ``N(u) = (4 − u)^3 / 6``                                       for 3 ≤ u ≤ 4
    * ``N(u) = 0``                                                    otherwise

    Continuous and ``C²`` everywhere, non-negative, peaks at ``u = 2`` with
    value ``2/3``.  Sum over integer translates is identically 1 (partition
    of unity).
    """
    zero = torch.zeros_like(u)
    in_support = (u >= 0.0) & (u <= 4.0)
    uc = torch.clamp(u, 0.0, 4.0)

    seg0 = (uc ** 3) / 6.0
    seg1 = (-3.0 * uc ** 3 + 12.0 * uc ** 2 - 12.0 * uc + 4.0) / 6.0
    seg2 = (3.0 * uc ** 3 - 24.0 * uc ** 2 + 60.0 * uc - 44.0) / 6.0
    seg3 = ((4.0 - uc) ** 3) / 6.0

    val = torch.where(uc < 1.0, seg0,
          torch.where(uc < 2.0, seg1,
          torch.where(uc < 3.0, seg2, seg3)))
    return torch.where(in_support, val, zero)


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class BSplineGDDModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`GlobalBSplineGDDModel`.

    Attributes:
        temperature_key:    Feature key for raw temperature.
        t_low:              Lower bound of chilling temperature range (°C).
        t_high:             Upper bound of chilling temperature range (°C).
        n_basis:            Number of cubic B-spline basis functions.  Must be
                            ≥ 4 (the minimum for cubic B-splines).
        learn_t_base:       Whether to learn the GDD base temperature.
        t_base_init:        Initial GDD base temperature (°C).
        learn_thresholds:   Whether to learn the soft-threshold locations.
        chill_threshold_init: Initial chilling requirement in natural chill
                            units (default 100).
        chill_threshold_max: Normalisation scale for the chilling threshold —
                            the SoftThreshold operates on
                            ``cum_chill / chill_threshold_max`` (default 200).
        gdd_threshold_init: Initial forcing requirement in natural heat units
                            (default 250).
        gdd_threshold_max:  Normalisation scale for the GDD threshold
                            (default 500).
        learn_bounds:       Whether to learn ``t_low``/``t_high``.
        bounds_reg_lambda:  L1 regularisation weight on ``t_high − t_low``.
        bounds_min_width:   Minimum allowed chilling-window width (°C).
        controls_init:      Initial value used for every control point before
                            softplus (default 0.541 → softplus ≈ 1.0).
        controls_reg_lambda: Optional L2 ridge penalty on raw control points
                            to shrink magnitudes toward zero.
        smoothness_reg_lambda: P-spline penalty (Eilers & Marx 1996) on the
                            squared 2nd differences of the (softplus-applied)
                            control points: ``Σᵢ (cᵢ₋₁ − 2cᵢ + cᵢ₊₁)²``.
                            Penalises curvature in the control-point sequence,
                            discouraging wiggly / bimodal response shapes
                            without imposing a parametric form.  Linear and
                            constant control-point sequences pay zero penalty.
        normalize_peak:     If True, rescale the response to peak at 1.0 so
                            only the *shape* is learned.  When False, control
                            magnitudes carry chilling-rate information — useful
                            when the chilling threshold is held fixed.
        zero_boundary:      If True, fix the two outermost control points on
                            each side (``c[0], c[1], c[n−2], c[n−1]``) to 0,
                            which guarantees ``f(t_low) = f(t_high) = 0`` with
                            zero first and second derivatives at the bounds.
                            Only the ``n_basis − 4`` interior controls are
                            learnable.  Requires ``n_basis ≥ 5``.
    """
    temperature_key: str = 'temperature_2m_mean'
    t_low: float = -5.0
    t_high: float = 20.0
    n_basis: int = 8
    learn_t_base: bool = True
    t_base_init: float = 5.0
    learn_thresholds: bool = True
    chill_threshold_init: float = 100.0
    chill_threshold_max: float = 200.0
    gdd_threshold_init: float = 250.0
    gdd_threshold_max: float = 500.0
    learn_bounds: bool = False
    bounds_reg_lambda: float = 0.0
    bounds_min_width: float = 5.0
    controls_init: float = 0.5413248546129181  # log(e − 1): softplus(.) ≈ 1
    controls_reg_lambda: float = 0.0
    smoothness_reg_lambda: float = 0.0
    normalize_peak: bool = True
    zero_boundary: bool = False
    early_stopping_patience: int = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlobalBSplineGDDModel(BaseTorchModel):
    """Cubic B-spline chilling + GDD forcing model.

    The chilling stage uses ``n_basis`` cardinal cubic B-spline basis functions
    centred at uniform knots ``x_i = t_low + i · h`` (``h = (t_high − t_low) /
    (n_basis − 1)``).  Per-day chilling contribution is::

        f(T) = soft_mask(T) · ⟨softplus(c), N((T − t_low)/h − i + 2)⟩ / peak

    where ``peak`` is computed on a dense grid so the curve peaks at 1.0.
    Cumulative chilling drives a soft-threshold gate on the cumulative GDD
    accumulation, exactly as in :class:`GlobalBetaGDDModel`.
    """

    _EPS: float = 1e-6
    _BOUNDS_STEEPNESS: float = 2.0    # sigmoid steepness for soft window mask
    _PEAK_GRID_PER_BASIS: int = 8     # density of grid used to estimate peak

    def __init__(
        self,
        temperature_key: str = 'temperature_2m_mean',
        t_low: float = -5.0,
        t_high: float = 20.0,
        n_basis: int = 8,
        learn_t_base: bool = True,
        t_base_init: float = 5.0,
        learn_thresholds: bool = True,
        chill_threshold_init: float = 100.0,
        chill_threshold_max: float = 200.0,
        gdd_threshold_init: float = 250.0,
        gdd_threshold_max: float = 500.0,
        learn_bounds: bool = False,
        bounds_reg_lambda: float = 0.0,
        bounds_min_width: float = 5.0,
        controls_init: float = 0.5413248546129181,
        controls_reg_lambda: float = 0.0,
        smoothness_reg_lambda: float = 0.0,
        normalize_peak: bool = True,
        zero_boundary: bool = False,
    ) -> None:
        super().__init__()
        if n_basis < 4:
            raise ValueError(f'n_basis must be >= 4, got {n_basis}')
        if zero_boundary and n_basis < 5:
            raise ValueError(
                f'zero_boundary=True requires n_basis >= 5 (need at least 1 '
                f'free interior control after zeroing 4 boundary controls); '
                f'got n_basis={n_basis}'
            )
        if chill_threshold_max <= 0.0:
            raise ValueError(
                f'chill_threshold_max must be > 0, got {chill_threshold_max}'
            )
        if gdd_threshold_max <= 0.0:
            raise ValueError(
                f'gdd_threshold_max must be > 0, got {gdd_threshold_max}'
            )

        self._temperature_key = temperature_key
        self._n_basis = int(n_basis)
        self._bounds_reg_lambda = float(bounds_reg_lambda)
        self._bounds_min_width = float(bounds_min_width)
        self._controls_reg_lambda = float(controls_reg_lambda)
        self._smoothness_reg_lambda = float(smoothness_reg_lambda)
        self._normalize_peak = bool(normalize_peak)
        self._zero_boundary = bool(zero_boundary)

        # Learnable control points (mapped to non-negative via softplus).
        # When zero_boundary=True, only n_basis-4 *interior* controls are
        # learnable; c[0], c[1], c[n-2], c[n-1] are hardcoded to 0 and
        # prepended/appended in the controls property.
        n_free = (n_basis - 4) if self._zero_boundary else n_basis
        self._raw_controls = nn.Parameter(
            torch.full((n_free,), float(controls_init))
        )

        # Learnable bounds — same parameterisation as BetaGDDModel
        self._raw_t_low = nn.Parameter(
            torch.tensor(float(t_low)),
            requires_grad=learn_bounds,
        )
        gap_init = max(float(t_high) - float(t_low) - float(bounds_min_width), 1e-3)
        raw_gap_init = float(np.log(np.exp(gap_init) - 1.0)) if gap_init > 1e-3 else -5.0
        self._raw_t_gap = nn.Parameter(
            torch.tensor(raw_gap_init),
            requires_grad=learn_bounds,
        )

        # Thresholds operate on inputs normalised to ``input / threshold_max``.
        # The initial threshold value is given in natural units (e.g. 100 chill
        # units) and converted to the same normalised scale before being stored
        # as the SoftThreshold's threshold parameter.
        self._tt_unit_threshold = SoftThreshold(
            threshold=float(chill_threshold_init) / float(chill_threshold_max),
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=float(chill_threshold_max),
        )
        self._gdd_threshold = SoftThreshold(
            threshold=float(gdd_threshold_init) / float(gdd_threshold_max),
            slope=40.0,
            slope_requires_grad=False,
            threshold_requires_grad=learn_thresholds,
            slope_positive=True,
            threshold_positive=False,
            b0=0.0,
            b1=float(gdd_threshold_max),
        )
        self._gdd = TorchGDD(t_base=t_base_init, t_base_req_grad=learn_t_base)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def t_low_eff(self) -> torch.Tensor:
        return self._raw_t_low

    @property
    def t_high_eff(self) -> torch.Tensor:
        return self._raw_t_low + self._bounds_min_width + F.softplus(self._raw_t_gap)

    @property
    def controls(self) -> torch.Tensor:
        """Non-negative control points (one per basis function).

        Length always equals ``n_basis``.  When ``zero_boundary=True`` the
        first two and last two entries are exact zeros (constants, not
        learnable parameters); only the ``n_basis − 4`` interior entries
        come from softplus of the learnable raw controls.
        """
        interior = F.softplus(self._raw_controls)
        if not self._zero_boundary:
            return interior
        zeros = torch.zeros(
            2, device=interior.device, dtype=interior.dtype,
        )
        return torch.cat([zeros, interior, zeros])

    @property
    def knot_centres(self) -> torch.Tensor:
        """Temperatures (°C) at which each basis function is centred."""
        t_low = self.t_low_eff
        t_high = self.t_high_eff
        h = (t_high - t_low) / (self._n_basis - 1)
        i = torch.arange(self._n_basis, device=t_low.device, dtype=t_low.dtype)
        return t_low + i * h

    # ------------------------------------------------------------------
    # B-spline evaluation
    # ------------------------------------------------------------------

    def _bspline_chilling_contrib(
        self,
        temp: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        t_low: Optional[torch.Tensor] = None,
        t_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-day chilling contributions as a cubic B-spline of T.

        Args:
            temp:     ``(B, T)`` raw temperature tensor.
            controls: Optional control-point override.  If ``None``, the
                      model's global :attr:`controls` (shape ``(n_basis,)``)
                      are used.  Otherwise:

                      * ``(n_basis,)`` — shared global controls (same as None).
                      * ``(B, n_basis)`` — per-sample controls.

            t_low:    Optional override of the lower temperature bound.  Scalar
                      (use as global) or ``(B,)`` (per-sample).  Defaults to
                      :attr:`t_low_eff`.
            t_high:   Optional override of the upper temperature bound.  Scalar
                      or ``(B,)``.  Defaults to :attr:`t_high_eff`.

        Returns:
            ``(B, T)`` chilling contributions, non-negative; in ``[0, 1]`` when
            ``normalize_peak=True``.
        """
        if t_low is None:
            t_low = self.t_low_eff
        if t_high is None:
            t_high = self.t_high_eff
        n_basis = self._n_basis

        # Make bounds broadcast against temp (B, T):
        # scalar tensors broadcast naturally; (B,) tensors are unsqueezed once.
        if t_low.dim() == 1:
            if t_low.size(0) != temp.size(0):
                raise ValueError(
                    f'per-sample t_low batch dim {t_low.size(0)} '
                    f'does not match temperature batch dim {temp.size(0)}'
                )
            t_low_b  = t_low.unsqueeze(-1)                    # (B, 1)
            t_high_b = t_high.unsqueeze(-1)                   # (B, 1)
        else:
            t_low_b  = t_low
            t_high_b = t_high
        h_b = (t_high_b - t_low_b) / (n_basis - 1)            # scalar or (B, 1)

        if controls is None:
            controls = self.controls                          # (n_basis,)
        if controls.dim() == 2:
            if controls.size(0) != temp.size(0):
                raise ValueError(
                    f'per-sample controls batch dim {controls.size(0)} '
                    f'does not match temperature batch dim {temp.size(0)}'
                )
            if controls.size(1) != n_basis:
                raise ValueError(
                    f'per-sample controls last dim {controls.size(1)} '
                    f'!= n_basis ({n_basis})'
                )
        elif controls.dim() != 1:
            raise ValueError(
                f'controls must be 1-D (n_basis,) or 2-D (B, n_basis); '
                f'got shape {tuple(controls.shape)}'
            )

        # u_i(T) = (T − t_low)/h − i + 2  ∈ [0, 4] when basis i is supported
        Tn = (temp - t_low_b) / h_b                           # (B, T)
        i_grid = torch.arange(
            n_basis, device=temp.device, dtype=temp.dtype,
        )                                                     # (n_basis,)
        u = Tn.unsqueeze(-1) - i_grid + 2.0                   # (B, T, n_basis)
        basis = _cardinal_cubic_bspline(u)                    # (B, T, n_basis)
        if controls.dim() == 1:
            f = (basis * controls).sum(dim=-1)                # (B, T)
        else:
            f = (basis * controls.unsqueeze(1)).sum(dim=-1)   # (B, T)

        # Optional peak normalisation.  The peak depends only on the controls
        # (the grid `Tg` is in basis-index space, independent of t_low/t_high),
        # so per-sample bounds do not affect this branch.
        if self._normalize_peak:
            n_grid = self._PEAK_GRID_PER_BASIS * n_basis
            Tg = torch.linspace(0.0, n_basis - 1.0, n_grid,
                                device=temp.device, dtype=temp.dtype)
            ug = Tg.unsqueeze(-1) - i_grid + 2.0
            basis_g = _cardinal_cubic_bspline(ug)             # (n_grid, n_basis)
            if controls.dim() == 1:
                f_g = (basis_g * controls).sum(dim=-1)        # (n_grid,)
                peak = f_g.max() + self._EPS
            else:
                f_g = (basis_g.unsqueeze(0)
                       * controls.unsqueeze(1)).sum(dim=-1)   # (B, n_grid)
                peak = f_g.max(dim=-1, keepdim=True).values + self._EPS  # (B, 1)
            f = f / peak

        # Soft sigmoid window mask (matches BetaGDDModel for fair comparison)
        k = self._BOUNDS_STEEPNESS
        soft_mask = torch.sigmoid(k * (temp - t_low_b)) * torch.sigmoid(k * (t_high_b - temp))
        return f * soft_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        temp = xs[KEY_FEATURES][self._temperature_key]   # (B, T)

        tt_contrib = self._bspline_chilling_contrib(temp)
        tt_units   = torch.cumsum(tt_contrib, dim=-1)
        tt_reached = self._tt_unit_threshold(tt_units.unsqueeze(1)).squeeze(1)

        gdd_daily        = F.relu(temp - self._gdd._tb)
        gdd_daily_masked = gdd_daily * tt_reached
        gdd_cum          = torch.cumsum(gdd_daily_masked, dim=-1)
        ps               = self._gdd_threshold(gdd_cum.unsqueeze(1)).squeeze(1)

        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs  = torch.argmax(diff, dim=-1).clamp(0, ps.size(-1) - 1)

        return ixs.float(), {
            'ps':               ps,
            't_low':            self.t_low_eff.expand(temp.size(0)),
            't_high':           self.t_high_eff.expand(temp.size(0)),
            'controls':         self.controls,
            'knot_centres':     self.knot_centres,
            'tt_contrib':       tt_contrib,
            'tt_units':         tt_units,
            'tt_reached':       tt_reached,
            'gdd_daily':        gdd_daily,
            'gdd_daily_masked': gdd_daily_masked,
            'gdd_cum':          gdd_cum,
        }

    # ------------------------------------------------------------------
    # Loss — same BCE-on-soft-step recipe as BetaGDDModel
    # ------------------------------------------------------------------

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
            loss = loss + self._bounds_reg_lambda * (
                info['t_high'] - info['t_low']
            ).mean()
        if self._controls_reg_lambda > 0.0:
            loss = loss + self._controls_reg_lambda * self._raw_controls.pow(2).mean()
        if self._smoothness_reg_lambda > 0.0:
            c = self.controls                                  # (n_basis,)
            d2 = c[:-2] - 2.0 * c[1:-1] + c[2:]                # (n_basis - 2,)
            loss = loss + self._smoothness_reg_lambda * d2.pow(2).sum()
        return loss, {
            'forward_pass': info,
            'ys_pred':      ys_pred,
            'ys_true':      ys_true,
        }

    # ------------------------------------------------------------------
    # Convenience: fit_from_args
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset,
        model_args: BSplineGDDModelArgs,
        model: Optional['GlobalBSplineGDDModel'] = None,
        **kwargs,
    ) -> Tuple['GlobalBSplineGDDModel', Dict[str, Any]]:
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
# Context-conditional B-spline GDD
# ---------------------------------------------------------------------------

@dataclass
class CtxBSplineGDDModelArgs(BaseTorchModelArgs):
    """Arguments for :class:`CtxBSplineGDDModel` subclasses.

    The chilling-response *shape* (global control points) is shared across
    samples; the chilling temperature *window* (``t_low`` / ``t_high`` → knots)
    and the GDD base temperature are predicted per sample from context.

    Subclasses declare how the per-sample context vector is built (e.g.
    one-hot species, one-hot location, phylogenetic embedding, AlphaEarth
    embedding), so only the spline-side fields appear here.

    Attributes:
        temperature_key:       Feature key for raw temperature.
        t_low / t_high:        Anchor lower/upper chilling temperature bounds.
                               Per-sample bounds are produced by adding a
                               ``tanh``-bounded shift (``t_low_range``) and a
                               softplus-bounded gap on top of these anchors.
        n_basis:               Number of cubic B-spline basis functions.
        learn_t_base:          Whether the global ``t_base`` is learnable.
        t_base_init:           Initial global GDD base temperature (°C).
        learn_thresholds:      Whether to learn the soft-threshold locations.
        normalize_peak:        Rescale spline to peak at 1.0.
        zero_boundary:         If True, fix the four outermost controls to 0.
        smoothness_reg_lambda: P-spline penalty on Σ (2nd diffs of the global
                               control sequence)².
        controls_init:         Initial raw-control value (softplus(.) ≈ 1).
        controls_reg_lambda:   L2 ridge on the global raw controls.
        ctx_reg_lambda:        L2 ridge on the subclass's context-mapping
                               parameters (partial-pooling knob).
        t_low_range:           Max ``±`` shift on ``t_low`` per sample (°C).
        t_base_range:          Max ``±`` shift on ``t_base`` per sample (°C).
        bounds_min_width:      Minimum value of ``t_high − t_low`` (°C).
    """
    temperature_key: str = 'temperature_2m_mean'
    t_low: float = 0.0
    t_high: float = 15.0
    n_basis: int = 8
    learn_t_base: bool = True
    t_base_init: float = 5.0
    learn_thresholds: bool = False
    chill_threshold_init: float = 100.0
    chill_threshold_max: float = 200.0
    gdd_threshold_init: float = 250.0
    gdd_threshold_max: float = 500.0
    normalize_peak: bool = False
    zero_boundary: bool = True
    smoothness_reg_lambda: float = 0.01
    controls_init: float = 0.5413248546129181
    controls_reg_lambda: float = 0.0
    ctx_reg_lambda: float = 0.0
    t_low_range: float = 3.0
    t_base_range: float = 5.0
    bounds_min_width: float = 5.0
    early_stopping_patience: int = 10


class CtxBSplineGDDModel(GlobalBSplineGDDModel, ABC):
    """B-spline GDD where the chilling temperature window (knots) and the GDD
    base temperature are predicted per-sample from a context; the control
    points remain **global** (shared across samples).

    Per-sample parameterisation::

        raw = subclass._per_sample_raw_outputs(xs)             # (B, 3)
        raw_low, raw_gap, raw_tb = _raw_default + raw          # raw_default
                                                                # reproduces the
                                                                # global anchors

        t_low_per  = t_low_anchor  + t_low_range  · tanh(raw_low)
        t_high_per = t_low_per + bounds_min_width + softplus(raw_gap)
        t_base_per = t_base_global + t_base_range · tanh(raw_tb)

    Subclasses provide :meth:`_per_sample_raw_outputs` — typically a tiny
    learnable map from sample identity / features to ``(B, 3)``:

    * For **discrete** contexts (one-hot species, one-hot location), the map is
      a direct lookup table ``nn.Embedding(N, 3)`` — 3 parameters per category,
      no hidden layer.
    * For **continuous** contexts (phylogenetic MDS, AlphaEarth embedding), the
      map is a single ``nn.Linear(d_in, 3)`` — no hidden layer.

    Both styles default to zero offset at init so every sample reproduces the
    global anchors ``(t_low, t_high, t_base)`` before any training; per-sample
    deviations are learned only as their gradient signal justifies them.

    Args:
        ctx_reg_lambda:   L2 ridge on the subclass's context-mapping parameters
                          (the partial-pooling knob: larger → samples pulled
                          harder toward the global anchors).
        t_low_range:      Max ``±`` shift on ``t_low`` per sample (°C, default 3).
        t_base_range:     Max ``±`` shift on ``t_base`` per sample (°C, default 5).
        bounds_min_width: Minimum value of ``t_high − t_low`` (°C, default 5).
        controls_init:    Initial raw-control value (parent default 0.541 → ≈1).
        controls_reg_lambda: L2 ridge on the global raw controls (default 0).
        smoothness_reg_lambda: P-spline 2nd-difference penalty on the global
                          control sequence (default 0.01).
        ...               Remaining kwargs forwarded to :class:`GlobalBSplineGDDModel`.

    Subclass contract:

    * Each subclass must register an ``nn.Module`` attribute named ``_ctx_map``
      whose parameters constitute the context-mapping parameters (used by the
      optimiser to disable weight_decay on them — pooling is controlled
      explicitly through ``ctx_reg_lambda`` instead).
    * Each subclass must implement :meth:`_per_sample_raw_outputs` returning a
      ``(B, 3)`` offset from :attr:`_raw_default`.
    """

    def __init__(
        self,
        ctx_reg_lambda: float = 0.0,
        t_low_range: float = 3.0,
        t_base_range: float = 5.0,
        bounds_min_width: float = 5.0,
        temperature_key: str = 'temperature_2m_mean',
        t_low: float = 0.0,
        t_high: float = 15.0,
        n_basis: int = 8,
        learn_t_base: bool = True,
        t_base_init: float = 5.0,
        learn_thresholds: bool = False,
        chill_threshold_init: float = 100.0,
        chill_threshold_max: float = 200.0,
        gdd_threshold_init: float = 250.0,
        gdd_threshold_max: float = 500.0,
        normalize_peak: bool = False,
        zero_boundary: bool = True,
        smoothness_reg_lambda: float = 0.01,
        controls_init: float = 0.5413248546129181,
        controls_reg_lambda: float = 0.0,
    ) -> None:
        super().__init__(
            temperature_key=temperature_key,
            t_low=t_low,
            t_high=t_high,
            n_basis=n_basis,
            learn_t_base=learn_t_base,
            t_base_init=t_base_init,
            learn_thresholds=learn_thresholds,
            chill_threshold_init=chill_threshold_init,
            chill_threshold_max=chill_threshold_max,
            gdd_threshold_init=gdd_threshold_init,
            gdd_threshold_max=gdd_threshold_max,
            learn_bounds=False,                  # global bounds are anchors;
            bounds_reg_lambda=0.0,               # per-sample shifts come from ctx
            bounds_min_width=bounds_min_width,
            controls_init=controls_init,
            controls_reg_lambda=controls_reg_lambda,
            smoothness_reg_lambda=smoothness_reg_lambda,
            normalize_peak=normalize_peak,
            zero_boundary=zero_boundary,
        )
        # Global learnable control points stay (parent's ``_raw_controls``).
        # Only knots and t_base are ctx-predicted.

        self._ctx_reg_lambda = float(ctx_reg_lambda)
        self._t_low_range = float(t_low_range)
        self._t_base_range = float(t_base_range)

        # ``_raw_default`` is the (raw_low, raw_gap, raw_tb) triple that, when
        # added with zero offset, reproduces the global anchors:
        #   raw_low = 0  → tanh(0) = 0  → t_low_per  = t_low_anchor
        #   raw_tb  = 0  → tanh(0) = 0  → t_base_per = t_base_global
        #   raw_gap : softplus(raw_gap_init) = (t_high − t_low) − bounds_min_width
        init_gap = max(
            float(self.t_high_eff - self.t_low_eff - self._bounds_min_width),
            1e-3,
        )
        raw_gap_init = float(np.log(np.exp(init_gap) - 1.0)) if init_gap > 1e-3 else -5.0
        self.register_buffer(
            '_raw_default',
            torch.tensor([0.0, raw_gap_init, 0.0]),
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _per_sample_raw_outputs(self, xs: Dict[str, Any]) -> torch.Tensor:
        """Return per-sample ``(B, 3)`` *offset* from :attr:`_raw_default`.

        The three columns correspond to ``(raw_low, raw_gap, raw_tb)``.
        Offsets of zero mean "use the global anchor"; this method should
        return zero for unknown/unseen samples.
        """

    # ------------------------------------------------------------------
    # Per-sample knots + t_base
    # ------------------------------------------------------------------

    def _per_sample_params(
        self, xs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample ``(t_low, t_high, t_base)``, each shape ``(B,)``."""
        offset = self._per_sample_raw_outputs(xs)               # (B, 3)
        raw = self._raw_default.unsqueeze(0) + offset           # (B, 3)

        t_low_per  = self.t_low_eff + self._t_low_range * torch.tanh(raw[:, 0])
        t_gap_per  = self._bounds_min_width + F.softplus(raw[:, 1])
        t_high_per = t_low_per + t_gap_per
        t_base_per = self._gdd._tb + self._t_base_range * torch.tanh(raw[:, 2])
        return t_low_per, t_high_per, t_base_per

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        temp = xs[KEY_FEATURES][self._temperature_key]
        t_low_per, t_high_per, t_base_per = self._per_sample_params(xs)

        tt_contrib = self._bspline_chilling_contrib(
            temp, t_low=t_low_per, t_high=t_high_per,
        )
        tt_units   = torch.cumsum(tt_contrib, dim=-1)
        tt_reached = self._tt_unit_threshold(tt_units.unsqueeze(1)).squeeze(1)

        gdd_daily        = F.relu(temp - t_base_per.unsqueeze(-1))   # (B, T)
        gdd_daily_masked = gdd_daily * tt_reached
        gdd_cum          = torch.cumsum(gdd_daily_masked, dim=-1)
        ps               = self._gdd_threshold(gdd_cum.unsqueeze(1)).squeeze(1)

        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs  = torch.argmax(diff, dim=-1).clamp(0, ps.size(-1) - 1)

        # Per-sample knot centres for downstream inspection / plotting
        i_grid = torch.arange(
            self._n_basis, device=temp.device, dtype=temp.dtype,
        )
        h_per = (t_high_per - t_low_per) / (self._n_basis - 1)
        knot_centres = t_low_per.unsqueeze(-1) + i_grid * h_per.unsqueeze(-1)

        return ixs.float(), {
            'ps':               ps,
            't_low':            t_low_per,        # (B,)
            't_high':           t_high_per,       # (B,)
            't_base':           t_base_per,       # (B,)
            'controls':         self.controls,    # (n_basis,) — shared global
            'knot_centres':     knot_centres,     # (B, n_basis) — per-sample
            'tt_contrib':       tt_contrib,
            'tt_units':         tt_units,
            'tt_reached':       tt_reached,
            'gdd_daily':        gdd_daily,
            'gdd_daily_masked': gdd_daily_masked,
            'gdd_cum':          gdd_cum,
        }

    # ------------------------------------------------------------------
    # Loss — parent BCE + global-control smoothness/ridge + ctx ridge
    # ------------------------------------------------------------------

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Parent's loss handles BCE + smoothness on global self.controls + the
        # optional L2 ridge on self._raw_controls.  We add an explicit L2 on
        # the subclass's context-mapping parameters to control pooling
        # strength.  (Weight decay on _ctx_map is disabled in the optimiser so
        # this is the sole shrinkage acting on them.)
        loss, info = super().loss(xs, target_fn)
        if self._ctx_reg_lambda > 0.0 and hasattr(self, '_ctx_map'):
            ridge = sum(p.pow(2).sum() for p in self._ctx_map.parameters())
            loss = loss + self._ctx_reg_lambda * ridge
        return loss, info

    # ------------------------------------------------------------------
    # Optimizer — disable weight decay on the context-mapping parameters
    # so partial-pooling strength is controlled solely via ctx_reg_lambda.
    # ------------------------------------------------------------------

    @classmethod
    def _make_optimizer(
        cls, model: 'BaseTorchModel', name: str, kwargs: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        if name not in _OPTIMIZERS:
            raise ModelException(
                f"Unknown optimizer {name!r}. Choose from: {list(_OPTIMIZERS)}"
            )
        if not hasattr(model, '_ctx_map'):
            return _OPTIMIZERS[name](model.parameters(), **kwargs)
        ctx_ids = {id(p) for p in model._ctx_map.parameters()}
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
        model_args: CtxBSplineGDDModelArgs,
        model: Optional['CtxBSplineGDDModel'] = None,
        **kwargs,
    ) -> Tuple['CtxBSplineGDDModel', Dict[str, Any]]:
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

class OneHotSpeciesBSplineGDDModel(CtxBSplineGDDModel):
    """B-spline GDD with per-species knot/t_base offsets stored as a direct
    lookup table (no hidden layer, no MLP).

    Each ``(src, species_id)`` key gets its own row of three learnable scalars
    ``(off_low, off_gap, off_tb)``; per-sample raw outputs are obtained by
    indexing the table.  Total: ``3 · N_species`` parameters for the per-species
    map.  L2 ridge via ``ctx_reg_lambda`` shrinks offsets toward zero, pulling
    data-sparse species back toward the global anchors (partial pooling).

    Args:
        species_keys: Ordered list of ``(src, species_id)`` tuples.
        unknown:      ``'zero'`` (emit zero offset, falls back to the global
                      anchors) or ``'error'`` (raise :class:`KeyError`).
        ...           Remaining kwargs forwarded to :class:`CtxBSplineGDDModel`.
    """

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        unknown: str = 'zero',
        **base_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if not species_keys:
            raise ValueError('species_keys must be non-empty')

        super().__init__(**base_kwargs)
        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }
        # 3 learnable scalars per species: (off_low, off_gap, off_tb).
        # Zero-init → every species starts at the global anchors.
        self._ctx_map = nn.Embedding(len(species_keys), 3)
        with torch.no_grad():
            self._ctx_map.weight.zero_()

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        **base_kwargs: Any,
    ) -> 'OneHotSpeciesBSplineGDDModel':
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
            **base_kwargs,
        )

    def _per_sample_raw_outputs(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = next(self.parameters()).device

        indices: List[int] = []
        is_valid: List[bool] = []
        for s, sid in zip(srcs, sids):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in species_keys '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
                indices.append(0)
                is_valid.append(False)
            else:
                indices.append(row)
                is_valid.append(True)

        idx  = torch.tensor(indices, device=device, dtype=torch.long)
        mask = torch.tensor(is_valid, device=device,
                            dtype=self._ctx_map.weight.dtype).unsqueeze(-1)
        return self._ctx_map(idx) * mask                       # (B, 3); 0 for unknowns


# ---------------------------------------------------------------------------
# Concrete contextual: one-hot location encoding
# ---------------------------------------------------------------------------

class OneHotLocationBSplineGDDModel(CtxBSplineGDDModel):
    """B-spline GDD with per-location knot/t_base offsets stored as a direct
    lookup table (no hidden layer, no MLP).

    Each ``(src, loc_id)`` key gets its own row of three learnable scalars
    ``(off_low, off_gap, off_tb)``; total ``3 · N_loc`` parameters.  L2 ridge
    via ``ctx_reg_lambda`` does partial pooling.

    ``loc_id`` may be int or string depending on the data source; keys are
    stored as ``(str(src), str(loc_id))``.

    Args:
        location_keys: Ordered list of ``(src, loc_id)`` tuples.
        unknown:       ``'zero'`` (emit zero offset → global anchors) or
                       ``'error'``.
        ...            Remaining kwargs forwarded to :class:`CtxBSplineGDDModel`.
    """

    def __init__(
        self,
        location_keys: Sequence[Tuple[str, Any]],
        unknown: str = 'zero',
        **base_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        location_keys = list(location_keys)
        if not location_keys:
            raise ValueError('location_keys must be non-empty')

        super().__init__(**base_kwargs)
        self._unknown = unknown
        self._location_index: Dict[LocationKey, int] = {
            (str(s), str(lid)): i for i, (s, lid) in enumerate(location_keys)
        }
        self._ctx_map = nn.Embedding(len(location_keys), 3)
        with torch.no_grad():
            self._ctx_map.weight.zero_()

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        **base_kwargs: Any,
    ) -> 'OneHotLocationBSplineGDDModel':
        """Build from the unique ``(src, loc_id)`` pairs in *dataset*."""
        seen: Dict[LocationKey, None] = {}
        for ix in dataset.iter_index():
            key = (str(ix[0]), str(ix[1]))
            if key not in seen:
                seen[key] = None
        location_keys = sorted(seen)
        return cls(
            location_keys=location_keys,
            unknown=unknown,
            **base_kwargs,
        )

    def _per_sample_raw_outputs(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        lids = xs[KEY_LOC_ID]
        device = next(self.parameters()).device

        indices: List[int] = []
        is_valid: List[bool] = []
        for s, lid in zip(srcs, lids):
            row = self._location_index.get((str(s), str(lid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Location ({s!r}, {lid!r}) not in location_keys '
                        f'(known: {list(self._location_index.keys())[:5]}...)'
                    )
                indices.append(0)
                is_valid.append(False)
            else:
                indices.append(row)
                is_valid.append(True)

        idx  = torch.tensor(indices, device=device, dtype=torch.long)
        mask = torch.tensor(is_valid, device=device,
                            dtype=self._ctx_map.weight.dtype).unsqueeze(-1)
        return self._ctx_map(idx) * mask                       # (B, 3); 0 for unknowns


# ---------------------------------------------------------------------------
# Concrete contextual: phylogenetic MDS embedding
# ---------------------------------------------------------------------------

class PhylogeneticBSplineGDDModel(CtxBSplineGDDModel):
    """B-spline GDD with per-sample knot/t_base offsets predicted from a
    phylogenetic MDS embedding via a single linear layer (no hidden layer).

    For each ``(src, species_id)`` we look up its row of MDS coordinates,
    z-score normalise, and feed through ``nn.Linear(d_mds, 3)``.  The Linear
    layer is initialised to zero, so at step 0 every species reproduces the
    global anchors; phylogenetically similar species end up near each other
    in offset space as the layer is trained (their inputs are similar, the
    map is continuous).

    Args:
        species_keys: Ordered list of ``(src, species_id)`` tuples — one per
                      row of *mds_coords*.
        mds_coords:   ``(N_species, k)`` float array of MDS coordinates.
        unknown:      ``'zero'`` (emit zero MDS → zero offset) or ``'error'``.
        ...           Remaining kwargs forwarded to :class:`CtxBSplineGDDModel`.
    """

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        mds_coords: np.ndarray,
        unknown: str = 'zero',
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

        super().__init__(**base_kwargs)
        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }
        mds = np.asarray(mds_coords, dtype=np.float32)
        mds_mean = mds.mean(axis=0)
        mds_std  = mds.std(axis=0)
        mds_std[mds_std < 1e-8] = 1.0
        self.register_buffer('_mds_table', torch.from_numpy(mds))
        self.register_buffer('_mds_mean',  torch.from_numpy(mds_mean))
        self.register_buffer('_mds_std',   torch.from_numpy(mds_std))

        mds_dim = int(mds_coords.shape[1])
        self._ctx_map = nn.Linear(mds_dim, 3)
        with torch.no_grad():
            self._ctx_map.weight.zero_()
            self._ctx_map.bias.zero_()

    @classmethod
    def from_phylogeny_features(
        cls,
        phylo,
        unknown: str = 'zero',
        **base_kwargs: Any,
    ) -> 'PhylogeneticBSplineGDDModel':
        """Build from a fitted ``PhylogenyFeatures`` instance."""
        if phylo.mds_coords is None:
            raise ValueError(
                'PhylogenyFeatures must be fitted (with "mds" output) '
                'before constructing a PhylogeneticBSplineGDDModel.'
            )
        return cls(
            species_keys=list(phylo.species_keys),
            mds_coords=np.asarray(phylo.mds_coords),
            unknown=unknown,
            **base_kwargs,
        )

    def _per_sample_raw_outputs(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = self._mds_table.device

        rows: List[Optional[int]] = []
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

        mds_dim = self._mds_table.size(1)
        ctx = torch.zeros(len(rows), mds_dim, device=device,
                          dtype=self._mds_table.dtype)
        for i, r in enumerate(rows):
            if r is not None:
                ctx[i] = self._mds_table[r]
        ctx = (ctx - self._mds_mean) / self._mds_std            # (B, d_mds)
        return self._ctx_map(ctx)                                # (B, 3)


# ---------------------------------------------------------------------------
# Concrete contextual: AlphaEarth satellite embedding
# ---------------------------------------------------------------------------

class AlphaEarthBSplineGDDModel(CtxBSplineGDDModel):
    """B-spline GDD with per-sample knot/t_base offsets predicted from an
    AlphaEarth satellite embedding via a single linear layer.

    The 64-D annual embedding is read from
    ``xs['features'][alphaearth_key]``, z-score normalised using *ae_mean* /
    *ae_std*, and mapped to a ``(B, 3)`` offset by ``nn.Linear(64, 3)``.  The
    Linear is zero-initialised so every sample reproduces the global anchors
    at step 0; sites with similar embeddings produce similar offsets as the
    layer is trained.

    Args:
        ae_mean:        ``(alphaearth_dim,)`` per-dimension means (training set).
        ae_std:         ``(alphaearth_dim,)`` per-dimension std devs (training set).
        alphaearth_key: Feature key under which the embedding is stored
                        (default: ``'alphaearth_embedding'``).
        alphaearth_dim: Dimensionality of the embedding (default: 64).
        ...             Remaining kwargs forwarded to :class:`CtxBSplineGDDModel`.
    """

    def __init__(
        self,
        ae_mean: np.ndarray,
        ae_std: np.ndarray,
        alphaearth_key: str = 'alphaearth_embedding',
        alphaearth_dim: int = _AE_EMBED_DIM,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self._alphaearth_key = alphaearth_key
        ae_std_arr = np.asarray(ae_std, dtype=np.float32).copy()
        ae_std_arr[ae_std_arr < 1e-8] = 1.0
        self.register_buffer(
            '_ae_mean',
            torch.from_numpy(np.asarray(ae_mean, dtype=np.float32)),
        )
        self.register_buffer('_ae_std', torch.from_numpy(ae_std_arr))

        self._ctx_map = nn.Linear(int(alphaearth_dim), 3)
        with torch.no_grad():
            self._ctx_map.weight.zero_()
            self._ctx_map.bias.zero_()

    def _per_sample_raw_outputs(self, xs: Dict[str, Any]) -> torch.Tensor:
        emb = xs[KEY_FEATURES][self._alphaearth_key]   # (B, 64)
        emb = torch.nan_to_num(emb, nan=0.0).to(
            dtype=self._ae_mean.dtype, device=self._ae_mean.device,
        )
        emb = (emb - self._ae_mean) / self._ae_std
        return self._ctx_map(emb)                       # (B, 3)
