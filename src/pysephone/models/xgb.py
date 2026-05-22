"""
XGBoost phenology model.

XGBoost counterpart to :class:`pysephone.models.random_forest.RandomForestModel`.
Each season is represented as a flat feature vector: daily features over the
season concatenated -> ``(sum_k T_k,)`` per sample.  An ``XGBRegressor`` is
fitted on this representation to predict the within-season day index of the
phenological event.

Optional hyperparameter search (``RandomizedSearchCV``) over ``n_estimators``,
``max_depth``, and ``learning_rate`` can be enabled via
``hyperparameter_search=True``.

Example::

    from pysephone.models.xgb import XGBoostModel

    model, info = XGBoostModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(n_estimators=300, max_depth=8, learning_rate=0.05),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs, ModelException
from pysephone.models.util.flat_features import (
    build_day_index_targets,
    build_flat_feature_matrix,
    extract_flat_features,
)


_PARAM_DIST = {
    'n_estimators':  [200, 300, 500, 800],
    'max_depth':     [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample':     [0.7, 0.85, 1.0],
}


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class XGBoostModelArgs(ModelArgs):
    """Arguments for :class:`XGBoostModel`.

    Attributes:
        hyperparameter_search: Run ``RandomizedSearchCV`` to find optimal
                               hyperparameters before fitting the final model.
        n_iter_search:         Number of parameter settings sampled when
                               ``hyperparameter_search=True``.
        cv_folds:              Cross-validation folds used during search.
        random_state:          Seed for ``XGBRegressor`` and the randomised
                               search.
        data_keys:             Feature keys to include (order matters).
                               Defaults to ``['temperature_2m_mean',
                               'daylight_duration']``.
    """
    hyperparameter_search: bool = False
    n_iter_search: int = 20
    cv_folds: int = 5
    random_state: Optional[int] = 42
    data_keys: List[str] = field(
        default_factory=lambda: ['temperature_2m_mean', 'daylight_duration']
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class XGBoostModel(BaseModel):
    """XGBoost phenology model.

    Features are constructed by flattening the daily values of *data_keys*
    over the season into a single vector per sample.  The target is the
    within-season day index of the event.

    Args:
        n_estimators:  Number of boosting rounds.
        max_depth:     Maximum tree depth.
        learning_rate: Boosting learning rate (``eta``).
        subsample:     Row subsample ratio per tree.
        random_state:  Random seed.
        data_keys:     Ordered list of feature keys to include.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        random_state: Optional[int] = 42,
        data_keys: Optional[List[str]] = None,
    ) -> None:
        self._data_keys = data_keys or ['temperature_2m_mean', 'daylight_duration']
        self._xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            n_jobs=-1,
            objective='reg:squarederror',
            tree_method='hist',
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
        x = extract_flat_features(sample, self._data_keys).reshape(1, -1)
        ix = int(round(float(self._xgb.predict(x)[0])))
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
        model: Optional['XGBoostModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        hyperparameter_search: bool = False,
        n_iter_search: int = 20,
        cv_folds: int = 5,
        cv: Optional[Any] = None,
        cv_group_by: Optional[str] = None,
        random_state: Optional[int] = 42,
        data_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple['XGBoostModel', Dict[str, Any]]:
        """Fit an XGBoost regressor.

        Args:
            target_fn:             Callable extracting the target date from a sample.
            dataset:               Dataset to fit on.
            model_name:            Optional model name.
            model:                 Ignored (XGBoost always fits from scratch).
            model_kwargs:          Forwarded to the ``XGBoostModel`` constructor.
            hyperparameter_search: Run ``RandomizedSearchCV`` before fitting.
            n_iter_search:         Number of parameter settings to try in the search.
            cv_folds:              Cross-validation folds for the search.
            random_state:          Seed for reproducibility.
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

        if 'random_state' not in model_kwargs:
            model_kwargs['random_state'] = random_state

        effective_data_keys = model_kwargs.get(
            'data_keys', ['temperature_2m_mean', 'daylight_duration']
        )

        samples = list(dataset.iter_items())
        X = build_flat_feature_matrix(samples, effective_data_keys)
        y = build_day_index_targets(samples, target_fn)

        fit_info: Dict[str, Any] = {}

        if hyperparameter_search:
            base_xgb = XGBRegressor(
                random_state=random_state,
                n_jobs=-1,
                objective='reg:squarederror',
                tree_method='hist',
            )
            cv_eff = cv if cv is not None else cv_folds
            groups = (
                np.array([s[cv_group_by] for s in samples])
                if cv_group_by is not None else None
            )
            search = RandomizedSearchCV(
                base_xgb,
                param_distributions=_PARAM_DIST,
                n_iter=n_iter_search,
                cv=cv_eff,
                scoring='neg_mean_absolute_error',
                random_state=random_state,
                n_jobs=-1,
            )
            if groups is not None:
                search.fit(X, y, groups=groups)
            else:
                search.fit(X, y)
            best = search.best_params_
            fit_info['best_params'] = best
            fit_info['best_cv_score'] = -search.best_score_
            for k, v in best.items():
                model_kwargs[k] = v

        instance = cls(**model_kwargs)
        instance._xgb.fit(X, y)

        fit_info['n_samples'] = len(samples)
        fit_info['feature_shape'] = X.shape

        return instance, fit_info

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: XGBoostModelArgs,
        model: Optional['XGBoostModel'] = None,
        **kwargs,
    ) -> Tuple['XGBoostModel', Dict[str, Any]]:
        """Fit from a :class:`XGBoostModelArgs` instance."""
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
            data_keys=model_args.data_keys,
            **kwargs,
        )
