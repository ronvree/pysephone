"""
Abstract base class for process-based phenology models optimised with NLopt.

Subclass :class:`BasePBModel`, implement :meth:`predict` (which must return
``ix`` — the predicted within-season day index — inside its *info* dict), and
the two-phase NLopt optimisation (global GN_DIRECT followed by local LN_COBYLA)
is inherited for free.

Example::

    from pysephone.models.process_based import BasePBModel

    class MyModel(BasePBModel):

        def __init__(self, T_base: float = 5.0, F_crit: float = 200.0):
            super().__init__(params={'T_base': T_base, 'F_crit': F_crit})

        def predict(self, sample, **kwargs):
            temps = sample['features']['temperature_2m_mean']
            T_base = self._params['T_base']
            F_crit = self._params['F_crit']
            forcing = np.cumsum(np.maximum(temps - T_base, 0))
            ix = int(np.searchsorted(forcing, F_crit))
            season_start = sample['season_start']
            predicted_date = season_start + np.timedelta64(ix, 'D')
            return predicted_date, {'ix': ix}

    model, info = MyModel.fit(
        target_fn=lambda s: s['observations']['BBCH_11'],
        dataset=ds_train,
    )
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import nlopt
import numpy as np

from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs, ModelException


@dataclass
class BasePBModelArgs(ModelArgs):
    """Model arguments for :class:`BasePBModel` subclasses.

    All optimisation settings (bounds, ``opt_margin``, ``opt_max_steps``) are
    constructor-level and belong in ``model_kwargs``.  This class exists as a
    named base for process-based model arg subclasses.
    """


class BasePBModel(BaseModel):
    """Abstract base for process-based models optimised with NLopt.

    Subclasses must implement :meth:`predict`.  The *info* dict returned by
    :meth:`predict` **must** include ``'ix'``: the predicted within-season day
    index (integer), which is used as the optimisation target.

    Args:
        params:            Dict of all model parameters (name → initial value).
        params_keys_opt:   Subset of *params* keys to optimise.  Defaults to
                           all keys in *params*.
        opt_bounds_lower:  Lower bounds per optimised parameter.  Defaults to
                           ``param_value − 0.5 * abs(param_value)`` for each key.
        opt_bounds_upper:  Upper bounds per optimised parameter.  Defaults to
                           ``param_value + 0.5 * abs(param_value)`` for each key.
        opt_margin:        Fraction of the bound range used to tighten bounds
                           around the global optimum before the local refinement
                           step.  Default: ``0.1``.
        opt_max_steps:     Maximum NLopt function evaluations per phase.
                           Defaults to ``10 * 2 ** n_opt_params``.
    """

    def __init__(
        self,
        params: Dict[str, float],
        params_keys_opt: Optional[List[str]] = None,
        opt_bounds_lower: Optional[Dict[str, float]] = None,
        opt_bounds_upper: Optional[Dict[str, float]] = None,
        opt_margin: float = 0.1,
        opt_max_steps: Optional[int] = None,
    ) -> None:
        if params_keys_opt is not None:
            unknown = [k for k in params_keys_opt if k not in params]
            if unknown:
                raise ValueError(f"params_keys_opt contains keys not in params: {unknown}")

        self._params = dict(params)
        self._param_keys = list(params.keys())
        self._param_keys_opt = self._param_keys if params_keys_opt is None else list(params_keys_opt)

        if opt_bounds_lower is None:
            opt_bounds_lower = {
                k: self._params[k] - 0.5 * abs(self._params[k])
                for k in self._param_keys_opt
            }
        self._opt_bounds_lower = opt_bounds_lower

        if opt_bounds_upper is None:
            opt_bounds_upper = {
                k: self._params[k] + 0.5 * abs(self._params[k])
                for k in self._param_keys_opt
            }
        self._opt_bounds_upper = opt_bounds_upper

        self._opt_margin = opt_margin

        if opt_max_steps is None:
            opt_max_steps = 10 * (2 ** len(self._param_keys_opt))
        self._opt_max_steps = opt_max_steps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> Dict[str, float]:
        return dict(self._params)

    @property
    def num_params(self) -> int:
        return len(self._params)

    @property
    def num_params_opt(self) -> int:
        return len(self._param_keys_opt)

    @property
    def param_keys(self) -> List[str]:
        return list(self._param_keys)

    @property
    def param_keys_opt(self) -> List[str]:
        return list(self._param_keys_opt)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        """Predict the phenological transition date for one sample.

        The returned *info* dict **must** contain ``'ix'`` (``int``): the
        predicted within-season day index.  This is used during fitting to
        compute the MSE objective.
        """

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['BasePBModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple['BasePBModel', Dict[str, Any]]:
        """Fit model parameters using a two-phase NLopt optimisation.

        Phase 1 — global search with ``GN_DIRECT``.
        Phase 2 — local refinement with ``LN_COBYLA`` starting from the global
        optimum, tightening the bounds to ±*opt_margin* × bound-range around it.

        The optimisation objective is MSE of within-season day index
        (``info['ix']``) against the ground-truth index derived from
        ``target_fn(sample)`` and ``sample['season_start']``.

        Args:
            target_fn:    Callable returning the ground-truth date (anything
                          castable to ``np.datetime64``).  Receives a full
                          sample dict.
            dataset:      Dataset to fit on (must have a calendar attached so
                          that ``sample['season_start']`` is available).
            model_name:   Unused; kept for API compatibility.
            model:        Existing instance to warm-start from.
            model_kwargs: Keyword arguments forwarded to ``cls(...)`` when
                          *model* is ``None``.
            **kwargs:     Ignored.

        Returns:
            ``(fitted_model, {})``

        Raises:
            TypeError:      If *dataset* is not a :class:`~pysephone.dataset.dataset.Dataset`.
            ModelException: If fitting fails.
        """
        super().fit(target_fn, dataset, model_name=model_name,
                    model=model, model_kwargs=model_kwargs, **kwargs)

        if model is None:
            model = cls(**(model_kwargs or {}))

        if len(dataset) == 0:
            return model, {}

        def _true_ix(sample: Dict[str, Any]) -> int:
            target_dt = np.datetime64(target_fn(sample), 'D')
            season_start = np.datetime64(sample['season_start'], 'D')
            return int((target_dt - season_start) / np.timedelta64(1, 'D'))

        def f_objective(x, grad):
            model._set_model_params_opt(x)
            ys_pred: List[float] = []
            ys_true: List[float] = []
            for sample in dataset.iter_items():
                try:
                    _, info = model.predict(sample)
                    ys_pred.append(float(info['ix']))
                    ys_true.append(float(_true_ix(sample)))
                except Exception:
                    pass
            if not ys_pred:
                return float('inf')
            err = np.array(ys_true) - np.array(ys_pred)
            return float(np.mean(err ** 2))

        # Phase 1: global search
        opt = nlopt.opt(nlopt.GN_DIRECT, model.num_params_opt)
        opt.set_min_objective(f_objective)
        opt.set_lower_bounds(model._get_opt_bound_lower())
        opt.set_upper_bounds(model._get_opt_bound_upper())
        opt.set_maxeval(model._opt_max_steps)
        try:
            xopt = opt.optimize(model._get_opt_init())
        except Exception as exc:
            raise ModelException(f"Global optimisation failed: {exc}") from exc

        # Phase 2: local refinement
        opt = nlopt.opt(nlopt.LN_COBYLA, model.num_params_opt)
        opt.set_min_objective(f_objective)
        lb, ub = model._get_opt_margin_bounds(xopt)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_maxeval(model._opt_max_steps)
        try:
            xopt = opt.optimize(xopt)
        except Exception as exc:
            raise ModelException(f"Local optimisation failed: {exc}") from exc

        model._set_model_params_opt(xopt)
        return model, {}

    # ------------------------------------------------------------------
    # Parameter helpers (used internally by fit)
    # ------------------------------------------------------------------

    def _get_model_params_opt(self) -> np.ndarray:
        return np.array([self._params[k] for k in self._param_keys_opt])

    def _set_model_params_opt(self, param_array: np.ndarray) -> None:
        for key, value in zip(self._param_keys_opt, param_array):
            self._params[key] = float(value)

    def _get_opt_bound_lower(self) -> np.ndarray:
        return np.array([self._opt_bounds_lower[k] for k in self._param_keys_opt])

    def _get_opt_bound_upper(self) -> np.ndarray:
        return np.array([self._opt_bounds_upper[k] for k in self._param_keys_opt])

    def _get_opt_margin_bounds(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lb = self._get_opt_bound_lower()
        ub = self._get_opt_bound_upper()
        margin = self._opt_margin * (ub - lb)
        return np.maximum(lb, params - margin), np.minimum(ub, params + margin)

    def _get_opt_init(self) -> np.ndarray:
        lb = self._get_opt_bound_lower()
        ub = self._get_opt_bound_upper()
        epsilon = 1e-3
        return lb + epsilon * (ub - lb)
