"""
AlphaEarth feature provider for use with Dataset.

Wraps an :class:`AlphaEarthEmbeddingStore` (HDF5-backed) to serve the 64-D
annual satellite embedding for each dataset sample::

    features.get_data(index) -> {'alphaearth_embedding': np.ndarray}  # shape (64,)

Embeddings are looked up by a stable ``location_id`` derived from
``(lat, lon)`` rounded to ``precision`` decimal places — matching the
convention used by the Earth Engine sampling script in
:mod:`pysephone.data.alphaearth.obtain_embeddings`.

Downloading new embeddings is *not* handled here — it requires Earth Engine
authentication and is done separately via that module's batch fetcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pysephone.data.alphaearth.obtain_embeddings import (
    EMBED_DIM,
    AlphaEarthEmbeddingStore,
    _stable_location_id,
)
from pysephone.dataset.util.provider import FeatureProvider


class AlphaEarthFeatures(FeatureProvider):
    """AlphaEarth satellite embedding feature provider.

    Returns a single 64-D vector per sample, looked up by
    ``(lat, lon, year + year_offset)``.  Because AlphaEarth embeddings are
    annual (not per-day), the output is a 1-D array of length ``EMBED_DIM``
    regardless of season length.

    The provider must be preloaded (directly via :meth:`preload` or
    indirectly via :meth:`Dataset.preload_features`) to build the
    ``(src, loc_id) -> alpha_location_id`` lookup before :meth:`get_data`
    can be called.

    Args:
        h5_path:      Path to the HDF5 embedding store.  Defaults to
                      ``<data_root>/data/products/alphaearth/alphaearth_embeddings.h5``.
        year_offset:  Year used for lookup is ``sample.year + year_offset``.
                      Set to ``-1`` to fetch the embedding from the calendar
                      year *before* the observation year (useful for seasons
                      spanning two calendar years).
        precision:    Coordinate rounding precision for ``location_id``.  Must
                      match the precision used when the store was populated.
        key:          Name of the returned feature in the dict from
                      :meth:`get_data`.
        missing:      How to handle missing embeddings:

                      - ``'error'``  (default) — raise :class:`KeyError`.
                      - ``'zeros'``  — return a zero vector.
                      - ``'nan'``    — return a vector of NaNs.
    """

    DEFAULT_KEY: str = 'alphaearth_embedding'

    def __init__(
        self,
        h5_path: Optional[Path] = None,
        year_offset: int = 0,
        precision: int = 6,
        key: str = DEFAULT_KEY,
        missing: str = 'error',
    ) -> None:
        if missing not in ('error', 'zeros', 'nan'):
            raise ValueError(
                f"missing must be 'error', 'zeros' or 'nan', got {missing!r}"
            )

        self._store = AlphaEarthEmbeddingStore(
            h5_path=str(h5_path) if h5_path is not None else None
        )
        self._year_offset = int(year_offset)
        self._precision = int(precision)
        self._key = str(key)
        self._missing = missing

        # (src, loc_id) -> alpha location_id, filled in by preload()
        self._loc_id_map: Dict[Tuple[str, str], str] = {}
        # (src, loc_id, year) -> np.ndarray, in-memory cache filled by preload()
        self._cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public interface expected by Dataset
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return the embedding for *index* as a single-entry dict.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)``

        Returns:
            ``{key: np.ndarray}`` with a 1-D array of length
            :data:`EMBED_DIM`.

        Raises:
            RuntimeError: If :meth:`preload` has not been called yet.
            KeyError:     If no embedding is stored for the sample and
                          ``missing='error'``.
        """
        src, loc_id, year, _species_id, _subgroup_id = index
        target_year = int(year) + self._year_offset

        cache_key = (src, loc_id, target_year)
        if cache_key in self._cache:
            return {self._key: self._cache[cache_key].copy()}

        vec = self._load(src, loc_id, target_year)
        self._cache[cache_key] = vec.copy()
        return {self._key: vec}

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def preload(self, observations, verbose: bool = True) -> None:
        """Build the ``(src, loc_id) -> location_id`` map and cache all
        embeddings for the samples in *observations*.

        Args:
            observations: :class:`~pysephone.dataset.observations.Observations`
                          or :class:`~pysephone.dataset.dataset.Dataset`
                          instance — anything exposing ``iter_index()`` and
                          ``get_location_coordinates((src, loc_id))``.
            verbose:      Show a tqdm progress bar.
        """
        # Build lat/lon lookup for every unique (src, loc_id) in the dataset
        for ix in observations.iter_index():
            src, loc_id = ix[0], ix[1]
            if (src, loc_id) in self._loc_id_map:
                continue
            coords = observations.get_location_coordinates((src, loc_id))
            self._loc_id_map[(src, loc_id)] = _stable_location_id(
                coords['lat'], coords['lon'], precision=self._precision
            )

        # Preload all embeddings into memory
        indices = list(observations.iter_index())
        it = tqdm(indices, desc='Preloading AlphaEarth embeddings') if verbose else indices
        for ix in it:
            src, loc_id, year, _sp, _sg = ix
            target_year = int(year) + self._year_offset
            cache_key = (src, loc_id, target_year)
            if cache_key in self._cache:
                continue
            try:
                self._cache[cache_key] = self._load(src, loc_id, target_year).copy()
            except KeyError:
                # Missing embeddings are resolved at get_data() time according to
                # the `missing` policy; do not fail preload.
                pass

    # ------------------------------------------------------------------
    # Cache / lifecycle
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self.clear_cache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def h5_path(self) -> str:
        return self._store.h5_path

    @property
    def key(self) -> str:
        return self._key

    @property
    def year_offset(self) -> int:
        return self._year_offset

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, src: str, loc_id: str, year: int) -> np.ndarray:
        """Load one embedding from the HDF5 store.  Applies the ``missing``
        policy when the key is absent."""
        if not self._loc_id_map:
            raise RuntimeError(
                'AlphaEarthFeatures has not been preloaded. Call preload(observations) '
                'first, or use Dataset.preload_features() / Dataset.download_features().'
            )
        if (src, loc_id) not in self._loc_id_map:
            raise KeyError(
                f'No AlphaEarth location_id built for ({src!r}, {loc_id!r}). '
                'Call preload(observations) with an Observations containing this sample.'
            )

        alpha_loc_id = self._loc_id_map[(src, loc_id)]
        with self._store._open('r') as f:
            path = f'v1/locations/{alpha_loc_id}/embeddings/{int(year)}'
            if path in f:
                return np.array(f[path][...], dtype=np.float32)

        if self._missing == 'zeros':
            return np.zeros(EMBED_DIM, dtype=np.float32)
        if self._missing == 'nan':
            return np.full(EMBED_DIM, np.nan, dtype=np.float32)
        raise KeyError(
            f'AlphaEarth embedding missing for ({src!r}, {loc_id!r}, year={year}) '
            f'[alpha_loc_id={alpha_loc_id}]'
        )
