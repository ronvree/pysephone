"""
Growing Degree Days (GDD) phenology model.

The model accumulates heat units above a base temperature (and optionally
below an upper cap) starting from a configurable within-season day index,
and predicts the day at which the cumulative sum first reaches a threshold.

Example::

    from pysephone.models.gdd import GDDModel, observation_start

    # Start accumulation from the day of observed BBCH_11
    model = GDDModel(threshold=500.0, t_base=5.0, ix_start_fn=observation_start('BBCH_11'))

    model, info = GDDModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(threshold=500.0, t_base=5.0),
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pysephone.models.process_based import BasePBModel, BasePBModelArgs


@dataclass
class GDDModelArgs(BasePBModelArgs):
    """Model arguments for :class:`GDDModel`.

    Attributes:
        threshold:     Thermal time requirement (degree-days).
        t_base:        Base temperature (°C).
        t_upper:       Optional upper temperature cap (°C).
        ix_start_fn:   Callable returning the within-season start index.
                       Defaults to :func:`zero_start`.
        opt_margin:    Fraction of bound range for local refinement step.
        opt_max_steps: Max NLopt evaluations per optimisation phase.
    """
    threshold: float = 0.0
    t_base: float = 0.0
    t_upper: Optional[float] = None
    ix_start_fn: Optional[Callable[[Dict[str, Any]], int]] = None
    opt_margin: float = 0.1
    opt_max_steps: Optional[int] = None
    opt_max_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Start-index functions
# ---------------------------------------------------------------------------

def zero_start() -> Callable[[Dict[str, Any]], int]:
    """Return a start-index function that always returns 0 (season start)."""
    def _fn(_: Dict[str, Any]) -> int:
        return 0
    return _fn


def observation_start(obs_key: str) -> Callable[[Dict[str, Any]], int]:
    """Return a start-index function that reads ``sample['observations_index'][obs_key]``.

    Useful for starting heat accumulation from an already-observed event, e.g.
    leaf unfolding (BBCH_11) before predicting flowering (BBCH_60).

    Args:
        obs_key: Key in ``sample['observations_index']``, e.g. ``'BBCH_11'``.
    """
    def _fn(sample: Dict[str, Any]) -> int:
        return int(sample['observations_index'][obs_key])
    return _fn


# ---------------------------------------------------------------------------
# GDD model
# ---------------------------------------------------------------------------

class GDDModel(BasePBModel):
    """Growing Degree Days phenology model.

    Accumulates ``max(min(T, t_upper) - t_base, 0)`` day by day starting from
    the index returned by *ix_start_fn*, and predicts the first day on which
    the cumulative sum meets or exceeds *threshold*.

    Args:
        threshold:    Thermal time requirement (degree-days).
        t_base:       Base temperature below which no accumulation occurs (°C).
        t_upper:      Optional upper temperature cap (°C).  If ``None``, no
                      capping is applied.
        ix_start_fn:  Callable ``(sample) -> int`` returning the within-season
                      day index from which accumulation starts.  Defaults to
                      :func:`zero_start` (season day 0).
        opt_margin:   Forwarded to :class:`~pysephone.models.process_based.BasePBModel`.
        opt_max_steps: Forwarded to :class:`~pysephone.models.process_based.BasePBModel`.
    """

    def __init__(
        self,
        threshold: float,
        t_base: float,
        t_upper: Optional[float] = None,
        ix_start_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        params: Dict[str, float] = {
            'threshold': threshold,
            't_base':    t_base,
        }
        opt_bounds_lower: Dict[str, float] = {
            'threshold': 0.0,
            't_base':    0.0,
        }
        opt_bounds_upper: Dict[str, float] = {
            'threshold': float(2 ** 12),
            't_base':    20.0,
        }

        if t_upper is not None:
            params['t_upper'] = t_upper
            opt_bounds_lower['t_upper'] = 10.0
            opt_bounds_upper['t_upper'] = 40.0

        self._ix_start_fn: Callable[[Dict[str, Any]], int] = (
            ix_start_fn if ix_start_fn is not None else zero_start()
        )

        super().__init__(
            params=params,
            params_keys_opt=None,
            opt_bounds_lower=opt_bounds_lower,
            opt_bounds_upper=opt_bounds_upper,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_upper_bound(self) -> bool:
        return 't_upper' in self._params

    @property
    def threshold(self) -> float:
        return self._params['threshold']

    @property
    def t_base(self) -> float:
        return self._params['t_base']

    @property
    def t_upper(self) -> Optional[float]:
        return self._params.get('t_upper')

    # ------------------------------------------------------------------
    # fit_from_args
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset,
        model_args: GDDModelArgs,
        model: Optional['GDDModel'] = None,
        **kwargs,
    ):
        """Fit from a :class:`GDDModelArgs` instance.

        Builds ``model_kwargs`` from the typed fields on *model_args* so callers
        don't have to construct the dict manually.
        """
        model_kwargs = dict(
            threshold=model_args.threshold,
            t_base=model_args.t_base,
        )
        if model_args.t_upper is not None:
            model_kwargs['t_upper'] = model_args.t_upper
        if model_args.ix_start_fn is not None:
            model_kwargs['ix_start_fn'] = model_args.ix_start_fn
        if model_args.opt_margin != 0.1:
            model_kwargs['opt_margin'] = model_args.opt_margin
        if model_args.opt_max_steps is not None:
            model_kwargs['opt_max_steps'] = model_args.opt_max_steps
        if model_args.opt_max_time is not None:
            model_kwargs['opt_max_time'] = model_args.opt_max_time

        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--threshold', type=float, default=500.0,
                            help='Thermal time accumulation threshold (degree-days).')
        parser.add_argument('--t_base', type=float, default=5.0,
                            help='Base temperature (°C).')
        parser.add_argument('--t_upper', type=float, default=None,
                            help='Optional upper temperature cap (°C).')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> GDDModelArgs:
        return GDDModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold=args.threshold,
            t_base=args.t_base,
            t_upper=getattr(args, 't_upper', None),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        """Predict the phenological transition date for one sample.

        Args:
            sample: Dict from :meth:`~pysephone.dataset.dataset.Dataset.__getitem__`.

        Returns:
            ``(predicted_date, {'ix': int})`` where *ix* is the within-season
            day index at which the thermal time threshold is first reached.
        """
        temps: np.ndarray = sample['features']['temperature_2m_mean']
        ix_start: int = self._ix_start_fn(sample)

        ts = temps[ix_start:]

        if self.has_upper_bound:
            ts = np.clip(ts, a_min=None, a_max=self.t_upper)

        gdd = np.clip(ts - self.t_base, a_min=0.0, a_max=None).cumsum()

        # First day where cumulative GDD meets the threshold; if never met,
        # predict the last day of the available window.
        mask = gdd >= self.threshold
        ix_relative = int((1 - mask.astype(int)).sum())  # days before threshold
        ix = ix_start + ix_relative

        predicted_date = sample['season_start'] + np.timedelta64(int(ix), 'D')
        return predicted_date, {'ix': ix}
