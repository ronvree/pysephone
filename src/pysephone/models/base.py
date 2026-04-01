"""
Abstract base class for phenology models.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pysephone.dataset.dataset import Dataset
from pysephone.paths import get_data_root, get_model_dir


class ModelException(Exception):
    pass


@dataclass
class ModelArgs:
    """Base class for model arguments.

    Holds the constructor kwargs and an optional name.  Subclasses add
    fitting-procedure-specific fields and override :meth:`~BaseModel.fit_from_args`
    to pass them through.

    Attributes:
        model_name:   Identifier used when saving the fitted model.
                      Defaults to the class name when ``None``.
        model_kwargs: Keyword arguments forwarded to the model constructor.
    """
    model_name: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None


class BaseModel(ABC):

    @abstractmethod
    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        """Predict the moment of phenological transition for a single sample.

        Args:
            sample: Dict from ``Dataset.__getitem__`` / ``Dataset.iter_items()``.
            **kwargs: Model-specific keyword arguments.

        Returns:
            ``(predicted_moment, info)`` where *predicted_moment* is a
            ``np.datetime64`` and *info* is a dict of auxiliary outputs
            (e.g. intermediate states, uncertainty estimates).

        Raises:
            ModelException: If prediction fails.
        """

    def predict_all(
        self,
        samples: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Tuple[np.datetime64, Dict[str, Any]]]:
        """Predict for a list of samples by calling :meth:`predict` on each.

        Args:
            samples: List of sample dicts.
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            List of ``(predicted_moment, info)`` tuples.
        """
        return [self.predict(s, **kwargs) for s in samples]

    @classmethod
    @abstractmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['BaseModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple['BaseModel', Dict[str, Any]]:
        """Fit a model to *dataset*.

        Args:
            target_fn:    Callable that, given a sample dict, returns the
                          target value(s) to fit against.  For example::

                              target_fn = lambda s: s['observations']['BBCH_60']

            dataset:      Dataset to fit on.
            model_name:   Optional name; defaults to the class name.
            model:        Optional existing instance to warm-start from.
            model_kwargs: Keyword arguments forwarded to the model constructor
                          when creating a new instance.
            **kwargs:     Additional fitting arguments (e.g. optimiser settings).

        Returns:
            ``(fitted_model, info)`` where *info* carries training metrics,
            hyperparameters, or anything else useful for inspection.

        Raises:
            TypeError:      If *dataset* is not a :class:`~pysephone.dataset.dataset.Dataset`.
            ModelException: If fitting fails.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset instance, got {type(dataset)}")

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: ModelArgs,
        model: Optional['BaseModel'] = None,
        **kwargs,
    ) -> Tuple['BaseModel', Dict[str, Any]]:
        """Fit a model using a :class:`ModelArgs` instance.

        Subclasses that have fitting-procedure-specific arguments should
        override this method, extract those fields from *model_args*, and
        forward them as ``**kwargs`` to :meth:`fit`.

        Args:
            target_fn:  Callable extracting the target from a sample dict.
            dataset:    Dataset to fit on.
            model_args: Fitting arguments.  ``model_args.model_name`` is used
                        when set; falls back to the class name.
                        ``model_args.model_kwargs`` is forwarded to the
                        model constructor.
            model:      Optional existing instance to warm-start from.
            **kwargs:   Forwarded to :meth:`fit`.

        Returns:
            ``(fitted_model, info)``
        """
        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_args.model_kwargs,
            **kwargs,
        )

    def save(self, model_name: str, root=None) -> None:
        """Pickle the model to ``<data_root>/models/<model_name>/<model_name>.pickle``.

        Args:
            model_name: Identifier used for both the sub-directory and filename.
            root:       Data root path.  Defaults to :func:`~pysephone.paths.get_data_root`.

        Raises:
            ModelException: If the model cannot be written to disk.
        """
        model_dir = get_model_dir(root or get_data_root(), model_name)
        path = model_dir / f'{model_name}.pickle'
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except OSError as exc:
            raise ModelException(f"Failed to save model to {path}: {exc}") from exc

    @classmethod
    def load(cls, model_name: str, root=None) -> Tuple['BaseModel', Dict[str, Any]]:
        """Load a pickled model from disk.

        Args:
            model_name: Identifier used to locate
                        ``<data_root>/models/<model_name>/<model_name>.pickle``.
            root:       Data root path.  Defaults to :func:`~pysephone.paths.get_data_root`.

        Returns:
            ``(model, {})``

        Raises:
            ModelException: If the file is missing or cannot be unpickled.
        """
        path = get_model_dir(root or get_data_root(), model_name) / f'{model_name}.pickle'
        if not path.exists():
            raise ModelException(f"Model file not found: {path}")
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            if not isinstance(model, BaseModel):
                raise ModelException(
                    f"Loaded object is not a BaseModel instance, got {type(model)}"
                )
            return model, {}
        except pickle.UnpicklingError as exc:
            raise ModelException(f"Failed to unpickle model from {path}: {exc}") from exc


class NullModel(BaseModel):
    """Null model for debugging: always predicts ``1970-01-01``."""

    def predict(self, sample: Dict[str, Any], **kwargs) -> Tuple[np.datetime64, Dict[str, Any]]:
        return np.datetime64('1970-01-01'), {}

    @classmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['NullModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple['NullModel', Dict[str, Any]]:
        return NullModel(), {}
