"""
OpenMeteo feature provider for use with Dataset.

OpenMeteoFeatures wraps the OpenMeteoStores HDF5 backend together with a
Calendar to produce per-season meteorological time-series arrays.  It acts
as the feature provider expected by Dataset:

    features.get_data(index) -> dict[str, np.ndarray]

where *index* is a 5-tuple (src, loc_id, year, species_id, subgroup_id).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pysephone.data.openmeteo.download import OpenMeteoEntry, OpenMeteoStores, get_openmeteo_data
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.provider import FeatureProvider


class OpenMeteoFeatures(FeatureProvider):
    """
    Meteorological feature provider backed by OpenMeteo HDF5 stores.

    Retrieves time-series arrays for one or more weather variables for each
    dataset entry.  Season boundaries are determined by a :class:`Calendar`.

    Args:
        calendar:    Calendar used to determine season start/end for each entry.
        data_keys:   Weather variable names to expose
                     (e.g. ``['temperature_2m_mean', 'daylight_duration']``).
        step:        Temporal resolution: ``'daily'`` (default) or ``'hourly'``.
        debug_mode:  When ``True`` returns zero-filled arrays of the correct shape
                     without touching any HDF5 store.  Useful for unit tests.
        root:        Data root directory.  Defaults to ``get_data_root()``.
    """

    DEFAULT_DATA_KEYS: Tuple[str, ...] = ('temperature_2m_mean', 'daylight_duration')
    DEFAULT_STEP: str = 'daily'

    def __init__(
        self,
        calendar: Calendar,
        data_keys: Optional[List[str]] = None,
        step: Optional[str] = None,
        debug_mode: bool = False,
        root: Optional[Path] = None,
    ) -> None:
        if step is not None and step not in ('hourly', 'daily'):
            raise ValueError(f"step must be 'hourly' or 'daily', got {step!r}")

        self._calendar = calendar
        self._data_keys: List[str] = list(data_keys or self.DEFAULT_DATA_KEYS)
        self._step: str = step or self.DEFAULT_STEP
        self._debug_mode = debug_mode
        self._root = root

        # In-memory cache: (step, key, index) -> np.ndarray
        self._cache: Dict[Tuple, np.ndarray] = {}

        if not debug_mode:
            step_key_pairs = [(self._step, k) for k in self._data_keys]
            self._store: Optional[OpenMeteoStores] = OpenMeteoStores(step_key_pairs, root=root)
        else:
            self._store = None

    # ------------------------------------------------------------------
    # Public interface expected by Dataset
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return all weather variables for *index* as a dict of arrays.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)``

        Returns:
            ``{data_key: np.ndarray}`` — one array per registered data key,
            each covering the season window defined by the Calendar.
        """
        return {key: self._get_variable(self._step, key, index) for key in self._data_keys}

    def get_variable(self, key: str, index: Tuple) -> np.ndarray:
        """Return a single weather variable array for *index*.

        Args:
            key:   Variable name (must be in ``data_keys``).
            index: ``(src, loc_id, year, species_id, subgroup_id)``
        """
        return self._get_variable(self._step, key, index)

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download(
        self,
        observations,  # Observations instance — avoid circular import at module level
        verbose: bool = True,
        download_mode: str = None,
    ) -> None:
        """Download OpenMeteo data for every entry in *observations*.

        Args:
            observations:  :class:`~pysephone.dataset.observations.Observations`
                           instance.  Its location coordinates and season
                           boundaries are used to build download requests.
            verbose:       Show progress bars.
            download_mode: Forwarded to :func:`get_openmeteo_data`.
                           ``None`` downloads missing entries only,
                           ``'forced'`` re-downloads everything,
                           ``'skip'`` does nothing.
        """
        entries = set()

        for key in self._data_keys:
            for index in observations.iter_index():
                src, loc_id, year, species_id, subgroup_id = index

                season = self._calendar.get_season_info(
                    year=year,
                    species_id=species_id,
                    subgroup_id=subgroup_id,
                    src=src,
                    loc_id=loc_id,
                )
                season_start = season['season_start']
                season_end = season['season_end']

                coords = observations.get_location_coordinates((src, loc_id))

                EPOCH_YEAR = 1970
                year_min = int(season_start.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR
                year_max = int(season_end.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR

                for year_ in range(year_min, year_max + 1):
                    entries.add(OpenMeteoEntry(
                        step=self._step,
                        data_key=key,
                        src_key=src,
                        loc_id=loc_id,
                        loc_name=observations.get_location_name((src, loc_id)),
                        lat=coords['lat'],
                        lon=coords['lon'],
                        year=year_,
                    ))

        get_openmeteo_data(
            entries=entries,
            verbose=verbose,
            download_mode=download_mode,
            root=self._root,
        )

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def preload(self, observations_or_dataset, verbose: bool = True) -> None:
        """Load all feature data into memory and close the backing store.

        After this call every variable for every entry is held in
        ``self._cache`` and the HDF5 store is closed, so multiple
        DataLoader workers can call :meth:`get_data` concurrently without
        hitting store-lock contention.

        Args:
            observations_or_dataset: An :class:`~pysephone.dataset.observations.Observations`
                                     or :class:`~pysephone.dataset.dataset.Dataset` instance —
                                     anything that exposes ``iter_index()``.
            verbose: Show a tqdm progress bar.
        """
        if self._debug_mode:
            return

        indices = list(observations_or_dataset.iter_index())
        it = tqdm(indices, desc='Preloading features') if verbose else indices

        for index in it:
            for key in self._data_keys:
                self._get_variable(self._step, key, index)

        if self._store is not None:
            self._store.close()
            self._store = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Discard all cached arrays."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> str:
        return self._step

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

    def _get_variable(self, step: str, key: str, index: Tuple) -> np.ndarray:
        cache_key = (step, key, index)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if self._debug_mode:
            result = self._debug_array(step, index)
        else:
            result = self._load_from_store(step, key, index)

        self._cache[cache_key] = result.copy()
        return result

    def _season_bounds(
        self, step: str, index: Tuple
    ) -> Tuple[np.datetime64, np.datetime64, str]:
        """Compute (season_start, season_end_inclusive, step_fmt)."""
        src, loc_id, year, species_id, subgroup_id = index
        season = self._calendar.get_season_info(
            year=year, src=src, species_id=species_id,
            subgroup_id=subgroup_id, loc_id=loc_id,
        )
        step_fmt = 'h' if step == 'hourly' else 'D'
        season_start = season['season_start']
        # season_end marks the first moment *after* the season;
        # subtract one time unit to get the last element in the time series
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)
        return season_start, season_end, step_fmt

    def _debug_array(self, step: str, index: Tuple) -> np.ndarray:
        season_start, season_end, step_fmt = self._season_bounds(step, index)
        n = int((season_end - season_start) // np.timedelta64(1, step_fmt))
        return np.zeros(n)

    def _load_from_store(self, step: str, key: str, index: Tuple) -> np.ndarray:
        if self._store is None:
            raise RuntimeError(
                f"Cannot load '{key}' — the OpenMeteo store is closed. "
                "Call preload() before closing the store, or do not call close() "
                "until you are done reading data."
            )
        src, loc_id, year, species_id, subgroup_id = index

        # Need the un-adjusted season_start to determine which calendar years to load
        season_info = self._calendar.get_season_info(
            year=year, src=src, species_id=species_id,
            subgroup_id=subgroup_id, loc_id=loc_id,
        )
        season_start_raw = season_info['season_start']

        season_start, season_end, step_fmt = self._season_bounds(step, index)

        EPOCH_YEAR = 1970
        year_min = int(season_start_raw.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR
        year_max = int(season_end.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR

        # Convert to pandas Timestamps for DataFrame slicing
        ts_start = pd.Timestamp(season_start, tz='UTC')
        ts_end = pd.Timestamp(season_end, tz='UTC')

        try:
            frames = []
            for y in range(year_min, year_max + 1):
                df_year = self._store[step, key, src, loc_id, y]
                frames.append(df_year.bfill().ffill())  # fill gaps at year boundaries

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
                f"Failed to load OpenMeteo data: step={step!r}, key={key!r}, "
                f"src={src!r}, loc_id={loc_id!r}, years={year_min}–{year_max}"
            ) from exc
