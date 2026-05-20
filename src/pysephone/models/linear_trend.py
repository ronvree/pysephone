"""
Linear-trend baseline.

Predicts the within-season day index of a phenological event from the year
alone, ignoring all weather and location features.  The model captures the
secular climate-change trend — events drifting earlier (or later) per year —
which is a known signal in long-running phenology records.

Per :class:`LinearTrendModel`, one line is fitted per ``species_id``.  Species
with fewer than ``min_samples_per_species`` training samples (and any species
never seen at training time) fall back to a global line fitted across the
whole dataset.

Use this as a "what does year alone explain?" baseline.  A weather-aware
model should beat it; the *gap* tells you how much information the daily
weather features actually carry.

Example::

    from pysephone.models.linear_trend import LinearTrendModel

    model, info = LinearTrendModel.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
    )
    print(info['global_slope_days_per_year'])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from pysephone.constants import KEY_SPECIES_ID, KEY_YEAR
from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs
from pysephone.models.util.flat_features import build_day_index_targets


@dataclass
class LinearTrendModelArgs(ModelArgs):
    """Arguments for :class:`LinearTrendModel` (no fitting-procedure args).

    All constructor kwargs belong in :attr:`~ModelArgs.model_kwargs`.
    """


class LinearTrendModel(BaseModel):
    """Per-species linear trend of day index vs. year.

    Fits one :class:`sklearn.linear_model.LinearRegression` per species, plus
    a global fallback model fitted across all samples.  Species with fewer
    than ``min_samples_per_species`` training samples — and any species not
    seen at training time — are predicted with the global model.

    Args:
        min_samples_per_species: Minimum training samples required to fit a
                                 per-species line (must be >= 2).  Species
                                 below the threshold fall back to the global
                                 model.
    """

    def __init__(self, min_samples_per_species: int = 5) -> None:
        assert min_samples_per_species >= 2
        self._min_samples = min_samples_per_species
        self._species_models: Dict[Any, LinearRegression] = {}
        self._global_model: Optional[LinearRegression] = None

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        sample: Dict[str, Any],
        **kwargs,
    ) -> Tuple[np.datetime64, Dict[str, Any]]:
        if self._global_model is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )

        year = sample[KEY_YEAR]
        species_id = sample[KEY_SPECIES_ID]

        model = self._species_models.get(species_id, self._global_model)
        ix = int(round(float(model.predict(np.array([[year]], dtype=float))[0])))
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
        model: Optional['LinearTrendModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple['LinearTrendModel', Dict[str, Any]]:
        """Fit one trend line per species (with a global fallback).

        Args:
            target_fn:    Callable extracting the target date from a sample.
            dataset:      Dataset to fit on.
            model_name:   Optional model name.
            model:        Existing instance to refit (parameters are reset).
            model_kwargs: Forwarded to the ``LinearTrendModel`` constructor.

        Returns:
            ``(fitted_model, fit_info)`` where *fit_info* includes the global
            slope (days/year), the number of per-species lines fitted, and
            counts.
        """
        super().fit(target_fn, dataset, model_name=model_name,
                    model=model, model_kwargs=model_kwargs)

        if model is None:
            model = cls(**(model_kwargs or {}))

        if len(dataset) == 0:
            return model, {}

        samples = list(dataset.iter_items())
        y = build_day_index_targets(samples, target_fn)
        years = np.array([s[KEY_YEAR] for s in samples], dtype=float)
        species = [s[KEY_SPECIES_ID] for s in samples]

        # Global fallback
        model._global_model = LinearRegression()
        model._global_model.fit(years.reshape(-1, 1), y)

        # Per-species lines
        model._species_models = {}
        for sp_id in set(species):
            mask = np.array([s == sp_id for s in species])
            if mask.sum() < model._min_samples:
                continue
            sp_model = LinearRegression()
            sp_model.fit(years[mask].reshape(-1, 1), y[mask])
            model._species_models[sp_id] = sp_model

        fit_info: Dict[str, Any] = {
            'n_samples': len(samples),
            'n_species_total': len(set(species)),
            'n_species_models': len(model._species_models),
            'global_slope_days_per_year': float(model._global_model.coef_[0]),
            'global_intercept_day_index': float(model._global_model.intercept_),
        }
        return model, fit_info
