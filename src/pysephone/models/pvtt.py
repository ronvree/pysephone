"""
Photo-Vernalization Thermal Time (PVTT) phenology model.

Predicts the phenological transition date by accumulating thermal time
corrected for vernalization and photoperiod effects.

Example::

    from pysephone.models.pvtt import PVTTModel, PVTTModelArgs, observation_start

    model, info = PVTTModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(
            threshold_pvtt=800.0,
            threshold_vern=30.0,
            t_base=1.0,
            t_limit=32.0,
            t_upper=40.0,
            p_base=7.0,
            p_saturation=17.0,
        ),
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.dataset.dataset import Dataset
from pysephone.models.base import ModelException
from pysephone.models.process_based import BasePBModel, BasePBModelArgs
from pysephone.models.util.func_phenology import (
    func_vernalization_unit,
    func_vernalization_tres,
    func_photoperiod_factor,
    func_growing_degree_units_2,
)
from pysephone.utils.func import create_left_mask


@dataclass
class PVTTModelArgs(BasePBModelArgs):
    """Model arguments for :class:`PVTTModel`.

    Attributes:
        threshold_pvtt: Photo-vernalization thermal time requirement.
        threshold_vern: Vernalization unit requirement for full vernalization.
        t_base:         Base temperature (°C); no heat accumulation below this.
        t_limit:        Optimal temperature (°C); peak heat accumulation.
        t_upper:        Upper temperature limit (°C); no accumulation above.
        p_base:         Minimum photoperiod (h) below which photoperiod factor is 0.
        p_saturation:   Saturating photoperiod (h) above which factor is 1.
        ix_start:       Within-season day index from which accumulation begins.
        key_sow:        Key in ``observations_index`` for the sowing date.
        params_opt:     Subset of parameter names to optimise.  Defaults to
                        ``['th_pvtt', 'th_vern', 't_base', 'p_base', 'p_saturation']``.
        opt_margin:     Fraction of bound range for local refinement.
        opt_max_steps:  Max NLopt evaluations per optimisation phase.
        opt_max_time:   Max wall-clock seconds per optimisation phase.
    """
    threshold_pvtt: float = 800.0
    threshold_vern: float = 30.0
    t_base: float = 1.0
    t_limit: float = 32.0
    t_upper: float = 40.0
    p_base: float = 7.0
    p_saturation: float = 17.0
    ix_start: int = 0
    key_sow: str = 'BBCH_0'
    params_opt: Optional[List[str]] = None
    opt_margin: float = 0.1
    opt_max_steps: Optional[int] = None
    opt_max_time: Optional[float] = None


class PVTTModel(BasePBModel):
    """Photo-Vernalization Thermal Time phenology model.

    Accumulates thermal time corrected by a vernalization factor and a
    photoperiod factor, starting from the sowing date.  Predicts the day
    at which the cumulative corrected thermal time first meets *threshold_pvtt*.

    Requires ``features['temperature_2m_mean']`` and
    ``features['daylight_duration']`` (in seconds) in each sample, as well as
    ``observations_index[key_sow]`` for the sowing date index.

    Args:
        threshold_pvtt: PVTT accumulation requirement (degree-days).
        threshold_vern: Vernalization units required for full vernalization.
        t_base:         Base temperature (°C).
        t_limit:        Optimal temperature (°C) for heat accumulation.
        t_upper:        Upper temperature limit (°C) for heat accumulation.
        p_base:         Photoperiod base (h); below this PF = 0.
        p_saturation:   Photoperiod saturation (h); above this PF = 1.
        ix_start:       Season index from which accumulation starts.  Must be ≥ 0.
        key_sow:        Key for the sowing date in ``observations_index``.
        params_opt:     Parameter names to optimise.  Defaults to
                        ``['th_pvtt', 'th_vern', 't_base', 'p_base', 'p_saturation']``.
        opt_margin:     Forwarded to :class:`~pysephone.models.process_based.BasePBModel`.
        opt_max_steps:  Forwarded to :class:`~pysephone.models.process_based.BasePBModel`.
        opt_max_time:   Forwarded to :class:`~pysephone.models.process_based.BasePBModel`.
    """

    _DEFAULT_PARAMS_OPT = ['th_pvtt', 'th_vern', 't_base', 'p_base', 'p_saturation']

    _OPT_BOUNDS_LOWER = {
        'th_pvtt':       0.0,
        'th_vern':       0.0,
        't_base':        0.0,
        'p_base':        5.0,
        'p_saturation': 14.0,
    }
    _OPT_BOUNDS_UPPER = {
        'th_pvtt':        1000.0,
        'th_vern':          40.0,
        't_base':            5.0,
        'p_base':            9.0,
        'p_saturation':     20.0,
    }

    def __init__(
        self,
        threshold_pvtt: float,
        threshold_vern: float,
        t_base: float,
        t_limit: float,
        t_upper: float,
        p_base: float,
        p_saturation: float,
        ix_start: int = 0,
        key_sow: str = 'BBCH_0',
        params_opt: Optional[List[str]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
        opt_max_time: Optional[float] = None,
    ) -> None:
        if ix_start < 0:
            raise ValueError(f"ix_start must be >= 0, got {ix_start}")

        params = {
            'th_pvtt':       threshold_pvtt,
            'th_vern':       threshold_vern,
            't_base':        t_base,
            't_limit':       t_limit,
            't_upper':       t_upper,
            'p_base':        p_base,
            'p_saturation':  p_saturation,
        }

        if params_opt is None:
            params_opt = list(self._DEFAULT_PARAMS_OPT)

        opt_bounds_lower = {k: self._OPT_BOUNDS_LOWER[k] for k in params_opt}
        opt_bounds_upper = {k: self._OPT_BOUNDS_UPPER[k] for k in params_opt}

        self._ix_start = ix_start
        self._key_sow = key_sow

        super().__init__(
            params=params,
            params_keys_opt=params_opt,
            opt_bounds_lower=opt_bounds_lower,
            opt_bounds_upper=opt_bounds_upper,
            opt_margin=opt_margin,
            opt_max_steps=opt_max_steps,
            opt_max_time=opt_max_time,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold_pvtt(self) -> float:
        return self._params['th_pvtt']

    @property
    def threshold_vern(self) -> float:
        return self._params['th_vern']

    @property
    def t_base(self) -> float:
        return self._params['t_base']

    @property
    def t_limit(self) -> float:
        return self._params['t_limit']

    @property
    def t_upper(self) -> float:
        return self._params['t_upper']

    @property
    def p_base(self) -> float:
        return self._params['p_base']

    @property
    def p_saturation(self) -> float:
        return self._params['p_saturation']

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        """Predict the phenological transition date for one sample.

        Args:
            sample: Dict from :meth:`~pysephone.dataset.dataset.Dataset.__getitem__`.

        Returns:
            ``(predicted_date, info)`` where *info* contains:

            - ``'ix'``: within-season day index of the prediction
            - ``'req_met'``: whether the PVTT threshold was met before season end
            - ``'pvtt'``: daily corrected thermal time array
            - ``'pvtt_cs_eos'``: cumulative PVTT at end of season
            - ``'vf_eos'``: vernalization factor at end of season
        """
        ts = sample[KEY_FEATURES]['temperature_2m_mean']
        ps = sample[KEY_FEATURES]['daylight_duration'] / 3600.0  # seconds → hours

        ts = ts[self._ix_start:]
        ps = ps[self._ix_start:]

        obs_sow = sample[KEY_OBSERVATIONS_INDEX][self._key_sow]
        mask = create_left_mask(len(ts), obs_sow)

        vu = func_vernalization_unit(ts) * mask
        vt = func_vernalization_tres(vu.cumsum(axis=-1), threshold=self.threshold_vern)

        pf = func_photoperiod_factor(ps, p_base=self.p_base, p_sat=self.p_saturation)

        pvtt = func_growing_degree_units_2(
            ts,
            t_base=self.t_base,
            t_limit=self.t_limit,
            t_upper=self.t_upper,
        ) * mask * vt * pf

        pvtt_cs = pvtt.cumsum(axis=-1)

        req = np.where(pvtt_cs >= self.threshold_pvtt, 1, 0)
        ix_relative = int((1 - req).sum())
        ix = ix_relative + self._ix_start

        date = sample['season_start'] + np.timedelta64(int(ix), 'D')
        return date, {
            'ix':           ix,
            'req_met':      bool(req[-1]),
            'pvtt':         pvtt,
            'pvtt_cs_eos':  float(pvtt_cs[-1]),
            'vf_eos':       float(vt[-1]),
        }

    # ------------------------------------------------------------------
    # Vectorised batch prediction (overrides BasePBModel for speed)
    # ------------------------------------------------------------------

    def _predict_ixs_batch(
        self,
        samples: List[Dict[str, Any]],
        true_ixs: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Vectorised batch prediction across all samples simultaneously.

        Stacks all temperature and photoperiod arrays into ``(N, T)`` matrices
        and processes them in a single set of numpy operations, avoiding the
        per-sample Python loop of the default implementation.
        """
        ts_list, ps_list, sow_list = [], [], []
        for s in samples:
            ts_list.append(s[KEY_FEATURES]['temperature_2m_mean'][self._ix_start:])
            ps_list.append(s[KEY_FEATURES]['daylight_duration'][self._ix_start:] / 3600.0)
            sow_list.append(s[KEY_OBSERVATIONS_INDEX][self._key_sow])

        min_len = min(len(a) for a in ts_list)
        ts = np.stack([a[:min_len] for a in ts_list])   # (N, T)
        ps = np.stack([a[:min_len] for a in ps_list])   # (N, T)
        obs_sows = np.array(sow_list)                   # (N,)

        # Build per-sample mask: 0 before sow date, 1 from sow date onwards
        indices = np.arange(min_len)[np.newaxis, :]     # (1, T)
        mask = (indices >= obs_sows[:, np.newaxis]).astype(float)  # (N, T)

        vu = func_vernalization_unit(ts) * mask
        vt = func_vernalization_tres(vu.cumsum(axis=-1), threshold=self.threshold_vern)
        pf = func_photoperiod_factor(ps, p_base=self.p_base, p_sat=self.p_saturation)
        pvtt = func_growing_degree_units_2(
            ts, t_base=self.t_base, t_limit=self.t_limit, t_upper=self.t_upper,
        ) * mask * vt * pf

        pvtt_cs = pvtt.cumsum(axis=-1)
        req = (pvtt_cs >= self.threshold_pvtt).astype(int)
        ix_relative = (1 - req).sum(axis=-1)            # (N,)
        ixs = (ix_relative + self._ix_start).tolist()

        return ixs, list(true_ixs)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    @classmethod
    def configure_argparser(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--threshold_pvtt', type=float, default=800.0,
                            help='PVTT accumulation threshold.')
        parser.add_argument('--threshold_vern', type=float, default=30.0,
                            help='Vernalization unit requirement for full vernalization.')
        parser.add_argument('--t_base', type=float, default=1.0,
                            help='Base temperature (°C).')
        parser.add_argument('--t_limit', type=float, default=32.0,
                            help='Optimal temperature (°C).')
        parser.add_argument('--t_upper', type=float, default=40.0,
                            help='Upper temperature limit (°C).')
        parser.add_argument('--p_base', type=float, default=7.0,
                            help='Photoperiod base (h).')
        parser.add_argument('--p_saturation', type=float, default=17.0,
                            help='Photoperiod saturation (h).')
        parser.add_argument('--ix_start', type=int, default=0,
                            help='Within-season start index for accumulation.')
        parser.add_argument('--key_sow', type=str, default='BBCH_0',
                            help='Observation-index key for the sowing date.')
        parser.add_argument('--params_opt', type=str, nargs='+', default=None,
                            help='Parameter names to optimise (default: th_pvtt th_vern t_base p_base p_saturation).')
        parser.add_argument('--opt_max_time', type=float, default=None,
                            help='Max wall-clock seconds per optimisation phase (default: no limit).')
        return parser

    @classmethod
    def model_args_from_namespace(cls, args: argparse.Namespace) -> PVTTModelArgs:
        return PVTTModelArgs(
            model_name=getattr(args, 'model_name', None),
            threshold_pvtt=args.threshold_pvtt,
            threshold_vern=args.threshold_vern,
            t_base=args.t_base,
            t_limit=args.t_limit,
            t_upper=args.t_upper,
            p_base=args.p_base,
            p_saturation=args.p_saturation,
            ix_start=args.ix_start,
            key_sow=args.key_sow,
            params_opt=getattr(args, 'params_opt', None),
            opt_max_time=getattr(args, 'opt_max_time', None),
        )

    # ------------------------------------------------------------------
    # fit_from_args
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: PVTTModelArgs,
        model: Optional['PVTTModel'] = None,
        **kwargs,
    ) -> Tuple['PVTTModel', Dict[str, Any]]:
        """Fit from a :class:`PVTTModelArgs` instance."""
        model_kwargs = dict(
            threshold_pvtt=model_args.threshold_pvtt,
            threshold_vern=model_args.threshold_vern,
            t_base=model_args.t_base,
            t_limit=model_args.t_limit,
            t_upper=model_args.t_upper,
            p_base=model_args.p_base,
            p_saturation=model_args.p_saturation,
            ix_start=model_args.ix_start,
            key_sow=model_args.key_sow,
        )
        if model_args.params_opt is not None:
            model_kwargs['params_opt'] = model_args.params_opt
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


class CalibratedPVTTModel(PVTTModel):
    """Pre-calibrated PVTT model for winter wheat.

    Uses literature-based defaults for all parameters; only *threshold_pvtt*
    is optimised during fitting.
    """

    def __init__(self) -> None:
        super().__init__(
            threshold_pvtt=800.0,
            threshold_vern=30.0,
            t_base=1.0,
            t_limit=32.0,
            t_upper=40.0,
            p_base=7.0,
            p_saturation=17.0,
            ix_start=0,
            params_opt=['th_pvtt'],
        )
