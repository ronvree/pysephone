"""
Elevation feature provider for use with Dataset.

Wraps :mod:`pysephone.data.elevation` (OpenMeteo Elevation API + JSON cache)
to serve a single elevation value per dataset sample::

    features.get_data(index) -> {'elevation': np.ndarray([meters])}  # shape (1,)

Usage mirrors :class:`OpenMeteoFeatures`:

- ``Dataset.download_features()`` triggers :meth:`download` which resolves any
  uncached ``(lat, lon)`` via the API and persists them to disk.
- ``Dataset.preload_features()`` triggers :meth:`preload` which builds the
  in-memory ``(src, loc_id) -> elevation`` map used by :meth:`get_data`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pysephone.data.elevation.download import (
    DEFAULT_BATCH_SIZE,
    ElevationCache,
    fetch_elevations,
)
from pysephone.dataset.util.provider import FeatureProvider


class ElevationFeatures(FeatureProvider):
    """Per-location elevation (meters above sea level) as a feature.

    Args:
        cache_path: JSON cache file.  Defaults to
                    ``<data_root>/data/products/elevation/elevations.json``.
        precision:  Decimal places used for the ``(lat, lon)`` cache key.
                    4 ≈ 11 m at the equator — finer than the underlying DEM.
        key:        Feature name in the dict returned by :meth:`get_data`.
        timeout:    HTTP timeout, seconds.
        batch_size: Points per API request (capped at 100 by OpenMeteo).
    """

    DEFAULT_KEY = 'elevation'

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        precision: int = 4,
        key: str = DEFAULT_KEY,
        timeout: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._cache = ElevationCache(path=cache_path, precision=precision)
        self._key = str(key)
        self._timeout = float(timeout)
        self._batch_size = int(batch_size)

        # (src, loc_id) -> elevation_m
        self._loc_elev: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        src, loc_id, _year, _species_id, _subgroup_id = index
        key = (src, loc_id)
        if key not in self._loc_elev:
            raise RuntimeError(
                f'ElevationFeatures: no elevation loaded for ({src!r}, {loc_id!r}). '
                'Call Dataset.download_features() / preload_features(), or '
                'preload(observations) directly.'
            )
        return {self._key: np.asarray([self._loc_elev[key]], dtype=np.float32)}

    def download(
        self,
        observations,
        verbose: bool = True,
        download_mode: Optional[str] = None,
    ) -> None:
        """Resolve uncached elevations for every ``(src, loc_id)`` in
        *observations* via the OpenMeteo API.

        Args:
            download_mode:
                - ``None``   — fetch only missing values (default).
                - ``'forced'`` — re-fetch every point.
                - ``'skip'``  — skip the network entirely.
        """
        if download_mode == 'skip':
            return
        coords = [(lat, lon) for _src, _loc, lat, lon in self._collect_locations(observations)]
        fetch_elevations(
            coords,
            cache=self._cache,
            batch_size=self._batch_size,
            timeout=self._timeout,
            verbose=verbose,
            force=(download_mode == 'forced'),
        )

    def preload(self, observations, verbose: bool = True) -> None:
        """Populate the in-memory ``(src, loc_id) -> elevation`` map from the
        disk cache.  Samples whose coordinate is still missing will raise
        :class:`RuntimeError` at :meth:`get_data` time."""
        needed = self._collect_locations(observations)
        it = tqdm(needed, desc='Preloading elevation') if verbose else needed
        for src, loc_id, lat, lon in it:
            v = self._cache.get(lat, lon)
            if v is not None:
                self._loc_elev[(src, loc_id)] = v

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._loc_elev.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache_path(self) -> Path:
        return self._cache.path

    @property
    def key(self) -> str:
        return self._key

    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_locations(
        self, observations
    ) -> List[Tuple[str, str, float, float]]:
        seen: set = set()
        out: List[Tuple[str, str, float, float]] = []
        for ix in observations.iter_index():
            src, loc_id = ix[0], ix[1]
            if (src, loc_id) in seen:
                continue
            seen.add((src, loc_id))
            coords = observations.get_location_coordinates((src, loc_id))
            out.append((src, loc_id, float(coords['lat']), float(coords['lon'])))
        return out
