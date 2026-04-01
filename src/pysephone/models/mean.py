from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs, ModelException


@dataclass
class MeanModelArgs(ModelArgs):
    """Model arguments for :class:`MeanModel`."""


class MeanModel(BaseModel):
    """Baseline model that always predicts the mean observed day-of-season.

    The mean is computed over ``sample['observations_index'][target_key]``
    values seen during :meth:`fit`.
    """

    def __init__(self) -> None:
        self._mean: Optional[float] = None

    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        if self._mean is None:
            raise ModelException("MeanModel has not been fit yet")
        ix = int(round(self._mean))
        date = sample['season_start'] + np.timedelta64(ix, 'D')
        return date, {'ix': ix}

    @classmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['MeanModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple['MeanModel', Dict[str, Any]]:
        super().fit(target_fn, dataset, model_name=model_name,
                    model=model, model_kwargs=model_kwargs, **kwargs)

        if model is None:
            model = cls()

        ixs = []
        for sample in dataset.iter_items():
            target_dt = np.datetime64(target_fn(sample), 'D')
            season_start = np.datetime64(sample['season_start'], 'D')
            ixs.append(int((target_dt - season_start) / np.timedelta64(1, 'D')))

        if not ixs:
            return model, {}

        model._mean = sum(ixs) / len(ixs)
        return model, {'mean': model._mean, 'n': len(ixs)}
