"""
Abstract base class for feature providers.

A feature provider maps a sample index to a dict of named arrays::

    provider.get_data(index) -> dict[str, np.ndarray]

where *index* is a 5-tuple ``(src, loc_id, year, species_id, subgroup_id)``.

Subclass :class:`FeatureProvider` and implement :meth:`get_data` to integrate
any feature source with :class:`~pysephone.dataset.dataset.Dataset`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class FeatureProvider(ABC):
    """Abstract base for objects that pair a dataset sample with feature arrays.

    A concrete provider must implement :meth:`get_data`.  Optionally it may
    override :meth:`close` to release resources (file handles, sockets, etc.).

    Providers are context managers by default; the ``with`` statement calls
    :meth:`close` on exit.
    """

    @abstractmethod
    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return feature arrays for one dataset sample.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)`` identifying
                   a single entry in the dataset.

        Returns:
            A dict mapping feature names to 1-D numpy arrays.  All arrays for
            a given provider should cover the same time window so they can be
            stacked or concatenated by the caller.
        """

    def close(self) -> None:
        """Release any resources held by this provider.  No-op by default."""

    def __enter__(self) -> 'FeatureProvider':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
