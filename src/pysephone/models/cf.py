"""
Chill-Forcing (CF) process-based phenology models.

The CF model family assumes two sequential phases:

1. **Chilling phase** — chill units accumulate until *threshold_c* is reached
   (the chilling requirement is satisfied).
2. **Forcing phase** — heat units accumulate only after chilling is satisfied;
   the phenological event is predicted when *threshold_f* is first met.

Concrete subclasses implement :meth:`BaseCFModel.get_cf_features`, which returns
per-day chill and forcing unit arrays for a single sample.

Provided implementations
------------------------
* :class:`UtahGDDModel`         — Utah chill model + simple GDU forcing
* :class:`ChillingDaysGDDModel` — chilling-days proxy + GDU forcing
* :class:`DynamicGDDModel`      — Dynamic chill model (Fishman/Luedeling) + GDU forcing
"""

from __future__ import annotations

import argparse
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pysephone.constants import KEY_FEATURES
from pysephone.dataset.dataset import Dataset
from pysephone.models.process_based import BasePBModel, BasePBModelArgs
from pysephone.models.util.func_phenology import (
    func_chilling_days,
    func_dynamic_chill_daily,
    func_utah_chill,
)


# ---------------------------------------------------------------------------
# BaseCFModelArgs
# ---------------------------------------------------------------------------

@dataclass
class BaseCFModelArgs(BasePBModelArgs):
    """Shared arguments for all :class:`BaseCFModel` subclasses.

    Attributes:
        threshold_c:    Chilling requirement (chill units).
        threshold_f:    Forcing requirement (heat units).
        params_opt:     Names of parameters to optimise.
        opt_margin:     Fraction of bound range for local refinement.
        opt_max_steps:  Max NLopt evaluations per phase.
        opt_max_time:   Max wall-clock seconds per phase.
    """
    threshold_c: float = 50.0
    threshold_f: float = 200.0
    params_opt: Optional[List[str]] = None
    opt_margin: float = 0.1
    opt_max_steps: Optional[int] = None
    opt_max_time: Optional[float] = None


# ---------------------------------------------------------------------------
# BaseCFModel
# ---------------------------------------------------------------------------

class BaseCFModel(BasePBModel):
    """Abstract chill-forcing phenology model.

    Subclasses must implement :meth:`get_cf_features`.

    The chilling gate logic:

    1. Compute daily chill units ``cs`` and forcing units ``fs`` via
       :meth:`get_cf_features`.
    2. Accumulate ``cs``; mark days after ``threshold_c`` is reached as
       *chilling-satisfied*.
    3. Accumulate ``fs`` only on chilling-satisfied days.
    4. Predict the first day on which cumulative forcing reaches ``threshold_f``.

    Args:
        threshold_c:   Chilling requirement.
        threshold_f:   Forcing requirement.
        extra_params:  Additional model parameters (e.g. temperature thresholds).
        params_opt:    Names of parameters to optimise.
        opt_bounds_lower / opt_bounds_upper: Per-parameter bounds (only for
                       keys present in *params_opt*).
        opt_margin, opt_max_steps, opt_max_time: Forwarded to
                       :class:`~pysephone.models.process_based.BasePBModel`.
    """

    _DEFAULT_PARAMS_OPT: List[str] = ['th_c', 'th_f']
    _OPT_BOUNDS_LOWER: Dict[str, float] = {
        'th_c':  0.0,
        'th_f':  0.0,
    }
    _OPT_BOUNDS_UPPER: Dict[str, float] = {
        'th_c':  500.0,
        'th_f':  2000.0,
    }

    def __init__(
        self,
        threshold_c: float,
        threshold_f: float,
        extra_params: Optional[Dict[str, float]] = None,
        params_opt: Optional[List[str]] = None,
        opt_bounds_lower: Optional[Dict[str, float]] = None,
        opt_bounds_upper: Optional[Dict[str, float]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        params: Dict[str, float] = {
            'th_c': threshold_c,
            'th_f': threshold_f,
        }
        if extra_params:
            params.update(extra_params)

        if params_opt is None:
            params_opt = list(self._DEFAULT_PARAMS_OPT)

        # Merge class-level bound dicts with any caller overrides
        lb = {**self._OPT_BOUNDS_LOWER}
        ub = {**self._OPT_BOUNDS_UPPER}
        if opt_bounds_lower:
            lb.update(opt_bounds_lower)
        if opt_bounds_upper:
            ub.update(opt_bounds_upper)

        lb_filtered = {k: lb[k] for k in params_opt if k in lb}
        ub_filtered = {k: ub[k] for k in params_opt if k in ub}

        super().__init__(
            params=params,
            params_keys_opt=params_opt,
            opt_bounds_lower=lb_filtered,
            opt_bounds_upper=ub_filtered,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold_c(self) -> float:
        return self._params['th_c']

    @property
    def threshold_f(self) -> float:
        return self._params['th_f']

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_cf_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-day chill and forcing unit arrays for *sample*.

        Args:
            sample: Dict from :meth:`~pysephone.dataset.dataset.Dataset.__getitem__`.

        Returns:
            ``(chill_units, forcing_units)`` — 1-D arrays of equal length
            covering the full season.  Values must be ≥ 0.
        """

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, sample: Dict[str, Any], **kwargs
    ) -> Tuple[np.datetime64, Dict[str, Any]]:
        """Predict the phenological transition date for one sample.

        Returns:
            ``(predicted_date, info)`` where *info* contains:

            - ``'ix'``:      within-season day index of the prediction
            - ``'req_met'``: whether the forcing threshold was met before season end
            - ``'cs_eos'``:  cumulative chill units at end of season
            - ``'fs_eos'``:  cumulative forcing units at end of season
        """
        cs, fs = self.get_cf_features(sample)

        cs_sum = cs.cumsum()
        chill_gate = (cs_sum >= self.threshold_c).astype(float)

        fs_masked = fs * chill_gate
        fs_sum = fs_masked.cumsum()

        req = (fs_sum >= self.threshold_f).astype(int)
        ix = int((1 - req).sum())

        date = sample['season_start'] + np.timedelta64(int(ix), 'D')
        return date, {
            'ix':       ix,
            'req_met':  bool(req[-1]),
            'cs_eos':   float(cs_sum[-1]),
            'fs_eos':   float(fs_sum[-1]),
        }

    # ------------------------------------------------------------------
    # Vectorised batch prediction
    # ------------------------------------------------------------------

    def _predict_ixs_batch(
        self,
        samples: List[Dict[str, Any]],
        true_ixs: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Vectorised batch prediction using ``(N, T)`` numpy operations."""
        cs_list: List[np.ndarray] = []
        fs_list: List[np.ndarray] = []
        valid_true_ixs: List[float] = []

        for sample, true_ix in zip(samples, true_ixs):
            try:
                cs, fs = self.get_cf_features(sample)
                cs_list.append(cs)
                fs_list.append(fs)
                valid_true_ixs.append(true_ix)
            except Exception:
                pass

        if not cs_list:
            return [], []

        min_len = min(len(a) for a in cs_list)
        cs = np.stack([a[:min_len] for a in cs_list])   # (N, T)
        fs = np.stack([a[:min_len] for a in fs_list])   # (N, T)

        cs_sum = cs.cumsum(axis=-1)
        chill_gate = (cs_sum >= self.threshold_c).astype(float)

        fs_masked = fs * chill_gate
        fs_sum = fs_masked.cumsum(axis=-1)

        req = (fs_sum >= self.threshold_f).astype(int)
        ixs = (1 - req).sum(axis=-1).tolist()

        return ixs, valid_true_ixs

    # ------------------------------------------------------------------
    # CLI helpers (shared; subclasses extend as needed)
    # ------------------------------------------------------------------

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--threshold_c', type=float, default=50.0,
                            help='Chilling requirement (chill units).')
        parser.add_argument('--threshold_f', type=float, default=200.0,
                            help='Forcing requirement (heat units).')
        parser.add_argument('--params_opt', type=str, nargs='+', default=None,
                            help='Parameter names to optimise.')
        parser.add_argument('--opt_max_time', type=float, default=None,
                            help='Max wall-clock seconds per optimisation phase.')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> BaseCFModelArgs:
        return BaseCFModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold_c=args.threshold_c,
            threshold_f=args.threshold_f,
            params_opt=getattr(args, 'params_opt', None),
            opt_max_time=getattr(args, 'opt_max_time', None),
        )

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: BaseCFModelArgs,
        model: Optional['BaseCFModel'] = None,
        **kwargs,
    ) -> Tuple['BaseCFModel', Dict[str, Any]]:
        """Fit from a :class:`BaseCFModelArgs` instance."""
        model_kwargs = cls._model_kwargs_from_args(model_args)
        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    @classmethod
    def _model_kwargs_from_args(cls, model_args: BaseCFModelArgs) -> Dict[str, Any]:
        """Build ``model_kwargs`` dict from a :class:`BaseCFModelArgs` instance."""
        kw: Dict[str, Any] = {
            'threshold_c': model_args.threshold_c,
            'threshold_f': model_args.threshold_f,
        }
        if model_args.params_opt is not None:
            kw['params_opt'] = model_args.params_opt
        if model_args.opt_margin != 0.1:
            kw['opt_margin'] = model_args.opt_margin
        if model_args.opt_max_steps is not None:
            kw['opt_max_steps'] = model_args.opt_max_steps
        if model_args.opt_max_time is not None:
            kw['opt_max_time'] = model_args.opt_max_time
        return kw


# ---------------------------------------------------------------------------
# UtahGDDModel
# ---------------------------------------------------------------------------

@dataclass
class UtahGDDModelArgs(BaseCFModelArgs):
    """Arguments for :class:`UtahGDDModel`.

    Attributes:
        t_base: Base temperature (°C) for GDU forcing accumulation.
    """
    t_base: float = 4.0


class UtahGDDModel(BaseCFModel):
    """CF model: Utah chill units + GDU forcing.

    **Chilling**: Utah model (Richardson et al. 1974) — temperature-dependent
    chill units per day, ranging from −1 (hot) to +1 (optimal), with partial
    credit at intermediate temperatures.

    **Forcing**: simple growing degree units, ``max(T − t_base, 0)`` per day.

    Args:
        threshold_c:  Utah chill unit requirement.
        threshold_f:  GDU forcing requirement.
        t_base:       Base temperature (°C) for forcing (default: 4.0).
        params_opt:   Parameters to optimise (default: ``['th_c', 'th_f', 't_base']``).
        opt_max_time: Max wall-clock seconds per optimisation phase.
    """

    _DEFAULT_PARAMS_OPT = ['th_c', 'th_f', 't_base']
    _OPT_BOUNDS_LOWER = {
        'th_c':   0.0,
        'th_f':   0.0,
        't_base': 0.0,
    }
    _OPT_BOUNDS_UPPER = {
        'th_c':   200.0,
        'th_f':   2000.0,
        't_base':  10.0,
    }

    def __init__(
        self,
        threshold_c: float = 50.0,
        threshold_f: float = 200.0,
        t_base: float = 4.0,
        params_opt: Optional[List[str]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        super().__init__(
            threshold_c=threshold_c,
            threshold_f=threshold_f,
            extra_params={'t_base': t_base},
            params_opt=params_opt,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    @property
    def t_base(self) -> float:
        return self._params['t_base']

    def get_cf_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = sample[KEY_FEATURES]['temperature_2m_mean']
        chill = func_utah_chill(ts)
        force = np.maximum(ts - self.t_base, 0.0)
        return chill, force

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        super().configure_argparser(parser)
        parser.add_argument('--t_base', type=float, default=4.0,
                            help='Base temperature (°C) for GDU forcing.')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> UtahGDDModelArgs:
        return UtahGDDModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold_c=args.threshold_c,
            threshold_f=args.threshold_f,
            t_base=args.t_base,
            params_opt=getattr(args, 'params_opt', None),
            opt_max_time=getattr(args, 'opt_max_time', None),
        )

    @classmethod
    def _model_kwargs_from_args(cls, model_args: UtahGDDModelArgs) -> Dict[str, Any]:
        kw = super()._model_kwargs_from_args(model_args)
        kw['t_base'] = model_args.t_base
        return kw


# ---------------------------------------------------------------------------
# ChillingDaysGDDModel
# ---------------------------------------------------------------------------

@dataclass
class ChillingDaysGDDModelArgs(BaseCFModelArgs):
    """Arguments for :class:`ChillingDaysGDDModel`.

    Attributes:
        t_chill: Temperature threshold (°C) for counting a chill day (default: 7.2).
        t_base:  Base temperature (°C) for GDU forcing accumulation (default: 4.0).
    """
    t_chill: float = 7.2
    t_base: float = 4.0


class ChillingDaysGDDModel(BaseCFModel):
    """CF model: chilling-days proxy + GDU forcing.

    **Chilling**: one chill unit per day when daily mean temperature ≤ *t_chill*
    (classic 7.2 °C / 45 °F threshold by default).  This is a simple daily proxy
    for chilling hours when only mean temperature is available.

    **Forcing**: simple growing degree units, ``max(T − t_base, 0)`` per day.

    Args:
        threshold_c:  Chill-day requirement (number of days).
        threshold_f:  GDU forcing requirement.
        t_chill:      Temperature threshold for counting a chill day (default: 7.2).
        t_base:       Base temperature (°C) for forcing (default: 4.0).
        params_opt:   Parameters to optimise
                      (default: ``['th_c', 'th_f', 't_chill', 't_base']``).
        opt_max_time: Max wall-clock seconds per optimisation phase.
    """

    _DEFAULT_PARAMS_OPT = ['th_c', 'th_f', 't_chill', 't_base']
    _OPT_BOUNDS_LOWER = {
        'th_c':    0.0,
        'th_f':    0.0,
        't_chill': 0.0,
        't_base':  0.0,
    }
    _OPT_BOUNDS_UPPER = {
        'th_c':    200.0,
        'th_f':    2000.0,
        't_chill':  15.0,
        't_base':   10.0,
    }

    def __init__(
        self,
        threshold_c: float = 50.0,
        threshold_f: float = 200.0,
        t_chill: float = 7.2,
        t_base: float = 4.0,
        params_opt: Optional[List[str]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        super().__init__(
            threshold_c=threshold_c,
            threshold_f=threshold_f,
            extra_params={'t_chill': t_chill, 't_base': t_base},
            params_opt=params_opt,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    @property
    def t_chill(self) -> float:
        return self._params['t_chill']

    @property
    def t_base(self) -> float:
        return self._params['t_base']

    def get_cf_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = sample[KEY_FEATURES]['temperature_2m_mean']
        chill = func_chilling_days(ts, t_threshold=self.t_chill)
        force = np.maximum(ts - self.t_base, 0.0)
        return chill, force

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        super().configure_argparser(parser)
        parser.add_argument('--t_chill', type=float, default=7.2,
                            help='Temperature threshold (°C) for counting a chill day.')
        parser.add_argument('--t_base', type=float, default=4.0,
                            help='Base temperature (°C) for GDU forcing.')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> ChillingDaysGDDModelArgs:
        return ChillingDaysGDDModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold_c=args.threshold_c,
            threshold_f=args.threshold_f,
            t_chill=args.t_chill,
            t_base=args.t_base,
            params_opt=getattr(args, 'params_opt', None),
            opt_max_time=getattr(args, 'opt_max_time', None),
        )

    @classmethod
    def _model_kwargs_from_args(cls, model_args: ChillingDaysGDDModelArgs) -> Dict[str, Any]:
        kw = super()._model_kwargs_from_args(model_args)
        kw['t_chill'] = model_args.t_chill
        kw['t_base'] = model_args.t_base
        return kw


# ---------------------------------------------------------------------------
# DynamicGDDModel
# ---------------------------------------------------------------------------

@dataclass
class DynamicGDDModelArgs(BaseCFModelArgs):
    """Arguments for :class:`DynamicGDDModel`.

    Attributes:
        t_base:    Base temperature (°C) for GDU forcing accumulation (default: 4.0).
        amplitude: Half the synthetic daily temperature range (°C) used when
                   reconstructing hourly temperatures from daily means (default: 5.0).
    """
    t_base: float = 4.0
    amplitude: float = 5.0


class DynamicGDDModel(BaseCFModel):
    """CF model: Dynamic chill model + GDU forcing.

    **Chilling**: Fishman et al. (1987) Dynamic Model (as popularised by
    Luedeling et al.) — a two-state kinetic model that accumulates
    *chill portions*.  Because the model requires hourly temperatures, daily
    mean values are expanded to 24 synthetic hourly values using a cosine
    cycle with a configurable *amplitude*.

    **Forcing**: simple growing degree units, ``max(T − t_base, 0)`` per day.

    The physical parameters of the Dynamic Model (E0, E1, A0, A1, slope, Tf)
    are held at their literature defaults and are not optimised.

    Args:
        threshold_c:  Chill-portion requirement.
        threshold_f:  GDU forcing requirement.
        t_base:       Base temperature (°C) for forcing (default: 4.0).
        amplitude:    Half daily temperature range (°C) for the synthetic hourly
                      reconstruction (default: 5.0).
        params_opt:   Parameters to optimise
                      (default: ``['th_c', 'th_f', 't_base', 'amplitude']``).
        opt_max_time: Max wall-clock seconds per optimisation phase.
    """

    _DEFAULT_PARAMS_OPT = ['th_c', 'th_f', 't_base', 'amplitude']
    _OPT_BOUNDS_LOWER = {
        'th_c':      0.0,
        'th_f':      0.0,
        't_base':    0.0,
        'amplitude': 0.5,
    }
    _OPT_BOUNDS_UPPER = {
        'th_c':      150.0,
        'th_f':      2000.0,
        't_base':    10.0,
        'amplitude': 15.0,
    }

    def __init__(
        self,
        threshold_c: float = 50.0,
        threshold_f: float = 200.0,
        t_base: float = 4.0,
        amplitude: float = 5.0,
        params_opt: Optional[List[str]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        super().__init__(
            threshold_c=threshold_c,
            threshold_f=threshold_f,
            extra_params={'t_base': t_base, 'amplitude': amplitude},
            params_opt=params_opt,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    @property
    def t_base(self) -> float:
        return self._params['t_base']

    @property
    def amplitude(self) -> float:
        return self._params['amplitude']

    def get_cf_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = sample[KEY_FEATURES]['temperature_2m_mean']
        chill = func_dynamic_chill_daily(ts, amplitude=self.amplitude)
        force = np.maximum(ts - self.t_base, 0.0)
        return chill, force

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        super().configure_argparser(parser)
        parser.add_argument('--t_base', type=float, default=4.0,
                            help='Base temperature (°C) for GDU forcing.')
        parser.add_argument('--amplitude', type=float, default=5.0,
                            help='Half daily temperature range (°C) for hourly reconstruction.')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> DynamicGDDModelArgs:
        return DynamicGDDModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold_c=args.threshold_c,
            threshold_f=args.threshold_f,
            t_base=args.t_base,
            amplitude=args.amplitude,
            params_opt=getattr(args, 'params_opt', None),
            opt_max_time=getattr(args, 'opt_max_time', None),
        )

    @classmethod
    def _model_kwargs_from_args(cls, model_args: DynamicGDDModelArgs) -> Dict[str, Any]:
        kw = super()._model_kwargs_from_args(model_args)
        kw['t_base'] = model_args.t_base
        kw['amplitude'] = model_args.amplitude
        return kw
