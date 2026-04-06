"""
Random Forest phenology model.

Each season is represented as a flat feature vector: daily
``temperature_2m_mean`` and ``daylight_duration`` (converted to hours)
concatenated → ``(2 × T,)`` per sample.  A ``RandomForestRegressor`` is
fitted on this representation to predict the within-season day index of the
phenological event.

Optional hyperparameter search (``RandomizedSearchCV``) over
``n_estimators``, ``max_depth``, and ``min_samples_split`` can be enabled via
``hyperparameter_search=True``.

Example::

    from pysephone.models.random_forest import RandomForestModel

    model, info = RandomForestModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(n_estimators=200, max_depth=12),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from pysephone.constants import KEY_FEATURES
from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs, ModelException


# ---------------------------------------------------------------------------
# Pre-tuned hyperparameters per dataset (optional starting points)
# ---------------------------------------------------------------------------

_MAP_KWARGS: Dict[str, Dict[str, Any]] = {
    'GMU_Cherry_Japan_YS': dict(n_estimators=300, max_depth=15, min_samples_split=4),
    'PEP725':              dict(n_estimators=200, max_depth=12, min_samples_split=5),
}

_PARAM_DIST = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [8, 10, 12, 15, 20, None],
    'min_samples_split': [2, 4, 6, 8, 10],
}


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class RandomForestModelArgs(ModelArgs):
    """Arguments for :class:`RandomForestModel`.

    Attributes:
        hyperparameter_search: Run ``RandomizedSearchCV`` to find optimal
                               hyperparameters before fitting the final model.
        n_iter_search:         Number of parameter settings sampled when
                               ``hyperparameter_search=True``.
        cv_folds:              Cross-validation folds used during search.
        random_state:          Seed for ``RandomForestRegressor`` and the
                               randomised search.
        dataset_key:           Dataset name used to look up pre-tuned kwargs
                               from the internal table.  ``None`` → ignored.
        data_keys:             Feature keys to include (order matters).
                               Defaults to ``['temperature_2m_mean',
                               'daylight_duration']``.
    """
    hyperparameter_search: bool = False
    n_iter_search: int = 20
    cv_folds: int = 5
    random_state: Optional[int] = 42
    dataset_key: Optional[str] = None
    data_keys: List[str] = field(
        default_factory=lambda: ['temperature_2m_mean', 'daylight_duration']
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_DAYLIGHT_SCALE = 1.0 / 3600.0   # seconds → hours


def _extract_features(
    sample: Dict[str, Any],
    data_keys: List[str],
) -> np.ndarray:
    """Return a flat ``(sum_k T_k,)`` feature vector for one sample.

    ``daylight_duration`` is rescaled from seconds to hours; all other keys
    are used as-is.
    """
    parts = []
    for k in data_keys:
        arr = np.asarray(sample[KEY_FEATURES][k], dtype=float)
        if k == 'daylight_duration':
            arr = arr * _DAYLIGHT_SCALE
        parts.append(arr)
    return np.concatenate(parts)


def _build_feature_matrix(
    samples: List[Dict[str, Any]],
    data_keys: List[str],
) -> np.ndarray:
    rows = [_extract_features(s, data_keys) for s in samples]
    return np.vstack(rows)


def _build_targets(
    samples: List[Dict[str, Any]],
    target_fn: Callable[[Dict[str, Any]], Any],
) -> np.ndarray:
    ys = []
    for s in samples:
        target_dt = np.datetime64(target_fn(s), 'D')
        start_dt  = np.datetime64(s['season_start'], 'D')
        ys.append(int((target_dt - start_dt) / np.timedelta64(1, 'D')))
    return np.array(ys, dtype=float)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """Random Forest phenology model.

    Features are constructed by flattening daily ``temperature_2m_mean`` and
    ``daylight_duration`` (in hours) over the season into a single vector per
    sample.  The target is the within-season day index of the event.

    Args:
        n_estimators:      Number of trees.
        max_depth:         Maximum tree depth (``None`` → unlimited).
        min_samples_split: Minimum samples to split an internal node.
        random_state:      Random seed.
        data_keys:         Ordered list of feature keys to include.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        random_state: Optional[int] = 42,
        data_keys: Optional[List[str]] = None,
    ) -> None:
        self._data_keys = data_keys or ['temperature_2m_mean', 'daylight_duration']
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        self._random_state = random_state

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        sample: Dict[str, Any],
        **kwargs,
    ) -> Tuple[np.datetime64, Dict[str, Any]]:
        x = _extract_features(sample, self._data_keys).reshape(1, -1)
        ix = int(round(float(self._rf.predict(x)[0])))
        ix = max(0, ix)
        date = sample['season_start'] + np.timedelta64(ix, 'D')
        return date, {'ix': ix}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['RandomForestModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        hyperparameter_search: bool = False,
        n_iter_search: int = 20,
        cv_folds: int = 5,
        random_state: Optional[int] = 42,
        dataset_key: Optional[str] = None,
        data_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple['RandomForestModel', Dict[str, Any]]:
        """Fit a Random Forest regressor.

        Args:
            target_fn:             Callable extracting the target date from a sample.
            dataset:               Dataset to fit on.
            model_name:            Optional model name.
            model:                 Ignored (Random Forests always fit from scratch).
            model_kwargs:          Forwarded to the ``RandomForestModel`` constructor.
            hyperparameter_search: Run ``RandomizedSearchCV`` before fitting.
            n_iter_search:         Number of parameter settings to try in the search.
            cv_folds:              Cross-validation folds for the search.
            random_state:          Seed for reproducibility.
            dataset_key:           Look up pre-tuned kwargs from internal table.
            data_keys:             Feature keys to include.

        Returns:
            ``(fitted_model, fit_info)``
        """
        super().fit(target_fn, dataset, model_name=model_name,
                    model=model, model_kwargs=model_kwargs)

        if model_kwargs is None:
            model_kwargs = {}

        if data_keys is not None:
            model_kwargs.setdefault('data_keys', data_keys)

        if dataset_key is not None and dataset_key in _MAP_KWARGS:
            for k, v in _MAP_KWARGS[dataset_key].items():
                model_kwargs.setdefault(k, v)

        if 'random_state' not in model_kwargs:
            model_kwargs['random_state'] = random_state

        effective_data_keys = model_kwargs.get(
            'data_keys', ['temperature_2m_mean', 'daylight_duration']
        )

        samples = list(dataset.iter_items())
        X = _build_feature_matrix(samples, effective_data_keys)
        y = _build_targets(samples, target_fn)

        fit_info: Dict[str, Any] = {}

        if hyperparameter_search:
            base_rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
            search = RandomizedSearchCV(
                base_rf,
                param_distributions=_PARAM_DIST,
                n_iter=n_iter_search,
                cv=cv_folds,
                scoring='neg_mean_absolute_error',
                random_state=random_state,
                n_jobs=-1,
            )
            search.fit(X, y)
            best = search.best_params_
            fit_info['best_params'] = best
            fit_info['best_cv_score'] = -search.best_score_
            for k, v in best.items():
                model_kwargs[k] = v

        instance = cls(**model_kwargs)
        instance._rf.fit(X, y)

        fit_info['n_samples'] = len(samples)
        fit_info['feature_shape'] = X.shape

        return instance, fit_info

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: RandomForestModelArgs,
        model: Optional['RandomForestModel'] = None,
        **kwargs,
    ) -> Tuple['RandomForestModel', Dict[str, Any]]:
        """Fit from a :class:`RandomForestModelArgs` instance."""
        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_args.model_kwargs,
            hyperparameter_search=model_args.hyperparameter_search,
            n_iter_search=model_args.n_iter_search,
            cv_folds=model_args.cv_folds,
            random_state=model_args.random_state,
            dataset_key=model_args.dataset_key,
            data_keys=model_args.data_keys,
            **kwargs,
        )
