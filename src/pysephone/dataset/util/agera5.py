"""
AgERA5 feature provider for use with Dataset.

AgEra5Features wraps the AgEra5Stores HDF5 backend together with a Calendar
to produce per-season daily meteorological time-series arrays.  It acts as
the feature provider expected by Dataset:

    features.get_data(index) -> dict[str, np.ndarray]

where *index* is a 5-tuple (src, loc_id, year, species_id, subgroup_id).

AgERA5 is the daily, agriculture-oriented derivative of ERA5 from the
Copernicus Climate Data Store.  Variables use AgERA5's native NetCDF names
(e.g. 'Temperature_Air_2m_Mean_24h', 'Solar_Radiation_Flux').
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pysephone.data.agera5.download import (
    AgEra5Stores,
    DEFAULT_TILE_DEG,
    build_entries,
    get_agera5_data,
)
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.provider import FeatureProvider


class AgEra5Features(FeatureProvider):
    """
    Meteorological feature provider backed by AgERA5 HDF5 stores.

    Retrieves daily time-series arrays for one or more AgERA5 variables for
    each dataset entry.  Season boundaries are determined by a
    :class:`Calendar`.

    Args:
        calendar:    Calendar used to determine season start/end for each entry.
        data_keys:   AgERA5 variable names to expose
                     (e.g. ``['Temperature_Air_2m_Mean_24h', 'Solar_Radiation_Flux']``).
        debug_mode:  When ``True`` returns zero-filled arrays of the correct shape
                     without touching any HDF5 store.  Useful for unit tests.
        root:        Data root directory.  Defaults to ``get_data_root()``.
    """

    DEFAULT_DATA_KEYS: Tuple[str, ...] = (
        'Temperature_Air_2m_Mean_24h',
        'Solar_Radiation_Flux',
    )
    STEP: str = 'daily'

    def __init__(
        self,
        calendar: Calendar,
        data_keys: Optional[List[str]] = None,
        debug_mode: bool = False,
        root: Optional[Path] = None,
    ) -> None:
        self._calendar = calendar
        self._data_keys: List[str] = list(data_keys or self.DEFAULT_DATA_KEYS)
        self._debug_mode = debug_mode
        self._root = root

        # In-memory cache: (key, index) -> np.ndarray
        self._cache: Dict[Tuple, np.ndarray] = {}

        if not debug_mode:
            self._store: Optional[AgEra5Stores] = AgEra5Stores(self._data_keys, root=root)
        else:
            self._store = None

    # ------------------------------------------------------------------
    # Public interface expected by Dataset
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return all AgERA5 variables for *index* as a dict of arrays."""
        return {key: self._get_variable(key, index) for key in self._data_keys}

    def get_variable(self, key: str, index: Tuple) -> np.ndarray:
        """Return a single AgERA5 variable array for *index*."""
        return self._get_variable(key, index)

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download(
        self,
        observations,  # Observations instance — avoid circular import at module level
        verbose: bool = True,
        download_mode: str = None,
        tile_deg: float = DEFAULT_TILE_DEG,
    ) -> None:
        """Download AgERA5 data for every entry in *observations*.

        Args:
            observations:  :class:`~pysephone.dataset.observations.Observations`
                           instance.  Its location coordinates and season
                           boundaries are used to build CDS download requests.
            verbose:       Show progress bars.
            download_mode: Forwarded to :func:`get_agera5_data`.
                           ``None`` downloads missing entries only,
                           ``'forced'`` re-downloads everything,
                           ``'skip'`` does nothing.
            tile_deg:      Geographic tile size (degrees) for chunking CDS
                           requests.  Shrink if a region's per-request cost
                           still exceeds CDS's 400-unit limit.
        """
        entries = build_entries(observations, self._calendar, self._data_keys)
        get_agera5_data(
            entries=entries,
            verbose=verbose,
            download_mode=download_mode,
            root=self._root,
            tile_deg=tile_deg,
        )

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def preload(self, observations_or_dataset, verbose: bool = True) -> None:
        """Load all feature data into memory and close the backing store."""
        if self._debug_mode:
            return

        indices = list(observations_or_dataset.iter_index())
        it = tqdm(indices, desc='Preloading AgERA5 features') if verbose else indices

        for index in it:
            for key in self._data_keys:
                self._get_variable(key, index)

        if self._store is not None:
            self._store.close()
            self._store = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> str:
        return self.STEP

    @property
    def data_keys(self) -> List[str]:
        return list(self._data_keys)

    @property
    def calendar(self) -> Calendar:
        return self._calendar

    def close(self) -> None:
        """Close the HDF5 store and release memory."""
        if self._store is not None:
            self._store.close()
            self._store = None
        self.clear_cache()

    # ------------------------------------------------------------------
    # Internal retrieval
    # ------------------------------------------------------------------

    def _get_variable(self, key: str, index: Tuple) -> np.ndarray:
        cache_key = (key, index)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if self._debug_mode:
            result = self._debug_array(index)
        else:
            result = self._load_from_store(key, index)

        self._cache[cache_key] = result.copy()
        return result

    def _season_bounds(self, index: Tuple) -> Tuple[np.datetime64, np.datetime64]:
        """Compute (season_start, season_end_inclusive) for the daily step."""
        src, loc_id, year, species_id, subgroup_id = index
        season = self._calendar.get_season_info(
            year=year, src=src, species_id=species_id,
            subgroup_id=subgroup_id, loc_id=loc_id,
        )
        season_start = season['season_start']
        # season_end marks the first moment *after* the season;
        # subtract one day to get the last element in the time series
        season_end = season['season_end'] - np.timedelta64(1, 'D')
        return season_start, season_end

    def _debug_array(self, index: Tuple) -> np.ndarray:
        season_start, season_end = self._season_bounds(index)
        n = int((season_end - season_start) // np.timedelta64(1, 'D'))
        return np.zeros(n)

    def _load_from_store(self, key: str, index: Tuple) -> np.ndarray:
        if self._store is None:
            raise RuntimeError(
                f"Cannot load '{key}' — the AgERA5 store is closed. "
                "Call preload() before closing the store, or do not call close() "
                "until you are done reading data."
            )
        src, loc_id, year, species_id, subgroup_id = index

        season_info = self._calendar.get_season_info(
            year=year, src=src, species_id=species_id,
            subgroup_id=subgroup_id, loc_id=loc_id,
        )
        season_start_raw = season_info['season_start']

        season_start, season_end = self._season_bounds(index)

        EPOCH_YEAR = 1970
        year_min = int(season_start_raw.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR
        year_max = int(season_end.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR

        ts_start = pd.Timestamp(season_start, tz='UTC')
        ts_end = pd.Timestamp(season_end, tz='UTC')

        try:
            frames = []
            for y in range(year_min, year_max + 1):
                df_year = self._store[key, src, loc_id, y]
                frames.append(df_year.bfill().ffill())

            df = pd.concat(frames, axis=0)
            df.set_index('date', inplace=True)
            df = df[ts_start:ts_end]

            if key not in df.columns:
                raise KeyError(
                    f"Variable '{key}' not found in store. "
                    f"Available: {df.columns.tolist()}"
                )
            return df[key].values

        except KeyError as exc:
            raise KeyError(
                f"Failed to load AgERA5 data: key={key!r}, "
                f"src={src!r}, loc_id={loc_id!r}, years={year_min}–{year_max}"
            ) from exc
