from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_FEATURES,
    KEY_LAT,
    KEY_LOC_ID,
    KEY_LON,
    KEY_OBSERVATIONS,
    KEY_OBSERVATIONS_INDEX,
    KEY_SPECIES_ID,
    KEY_SUBGROUP_ID,
    KEY_YEAR,
)
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.func import DatasetException
from pysephone.dataset.util.provider import FeatureProvider

# Keys added to items by Dataset.__getitem__ when a calendar is attached
_KEY_SEASON_START = 'season_start'
_KEY_SEASON_END = 'season_end'


class Dataset:
    """
    A dataset of phenological observations, optionally paired with one or more
    feature providers and a crop calendar.

    **Minimal usage** (observations only)::

        obs = Observations(df_y, df_y_loc)
        dataset = Dataset(obs)

    **With OpenMeteo features and a calendar**::

        from pysephone.dataset.util.calendar import Calendar
        from pysephone.dataset.util.openmeteo import OpenMeteoFeatures

        cal = Calendar()
        cal.set_season('pep725', species_id=333, subgroup_id=300,
                       start_date='10-01', length=365)

        features = OpenMeteoFeatures(
            calendar=cal,
            data_keys=['temperature_2m_mean', 'daylight_duration'],
            step='daily',
        )
        dataset = Dataset(obs, calendar=cal, feature_providers=[features])

    **Named dataset loading**::

        dataset = Dataset.load('CPF_PEP725_winter_wheat',
                               calendar=cal, feature_providers=[features])

    When *calendar* is attached :meth:`__getitem__` adds ``season_start``,
    ``season_end``, and ``observations_index`` to each item dict.  When
    feature providers are attached their outputs are merged into a single
    ``'features'`` dict; keys from later providers overwrite keys from earlier
    ones if they clash.

    All selection and splitting methods return new ``Dataset`` instances that
    share the same calendar and feature providers — only the observation view
    changes.
    """

    def __init__(
        self,
        observations: Observations,
        calendar=None,
        feature_providers: Optional[Sequence[FeatureProvider]] = None,
    ) -> None:
        """
        Args:
            observations:       An :class:`Observations` instance.
            calendar:           Optional :class:`~pysephone.dataset.util.calendar.Calendar`
                                used to compute season windows and convert observation
                                datetimes to within-season indices.
            feature_providers:  Optional list of :class:`~pysephone.dataset.util.provider.FeatureProvider`
                                instances.  Each is called with the sample index
                                and their output dicts are merged into ``item['features']``.
        """
        if not isinstance(observations, Observations):
            raise TypeError(f"Expected Observations, got {type(observations)}")

        self._obs = observations
        self._calendar = calendar
        self._providers: List[FeatureProvider] = list(feature_providers or [])

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'Dataset':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release resources held by all attached feature providers."""
        for provider in self._providers:
            provider.close()

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._obs)

    def __contains__(self, index: Union[int, Tuple]) -> bool:
        return index in self._obs

    def __getitem__(self, index: Union[int, Tuple]) -> Dict[str, Any]:
        """
        Return a dict for one entry.

        The returned dict always contains all fields from
        :meth:`Observations.__getitem__`.

        If a *calendar* is attached the following keys are also present:

        * ``'season_start'``          – ``np.datetime64``
        * ``'season_end'``            – ``np.datetime64`` (last element of season)
        * ``'observations_index'``    – ``dict[obs_type, int]`` — each observation
          expressed as a day index within the season window

        If feature providers are attached:

        * ``'features'``  – merged ``dict[str, np.ndarray]`` from all providers
        """
        item = self._obs[index]

        if self._calendar is not None:
            src = item[KEY_DATA_SOURCE]
            loc_id = item[KEY_LOC_ID]
            year = item[KEY_YEAR]
            species_id = item[KEY_SPECIES_ID]
            subgroup_id = item[KEY_SUBGROUP_ID]
            observations = item[KEY_OBSERVATIONS]

            season = self._calendar.get_season_info(
                year=year,
                src=src,
                species_id=species_id,
                subgroup_id=subgroup_id,
                loc_id=loc_id,
            )

            season_start = season['season_start']
            season_end = season['season_end'] - np.timedelta64(1, 'D')

            obs_ixs = {
                key: int((o - season_start) // np.timedelta64(1, 'D'))
                for key, o in observations.items()
            }

            item[_KEY_SEASON_START] = season_start
            item[_KEY_SEASON_END] = season_end
            item[KEY_OBSERVATIONS_INDEX] = obs_ixs

        if self._providers:
            ix = (
                item[KEY_DATA_SOURCE],
                item[KEY_LOC_ID],
                item[KEY_YEAR],
                item[KEY_SPECIES_ID],
                item[KEY_SUBGROUP_ID],
            )
            merged: Dict[str, np.ndarray] = {}
            for provider in self._providers:
                merged.update(provider.get_data(ix))
            item[KEY_FEATURES] = merged

        return item

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_index(self) -> Iterator[Tuple]:
        return self._obs.iter_index()

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        for ix in self.iter_index():
            yield self[ix]

    # ------------------------------------------------------------------
    # Forwarded observation properties
    # ------------------------------------------------------------------

    @property
    def observations(self) -> Observations:
        return self._obs

    @property
    def calendar(self):
        return self._calendar

    @property
    def feature_providers(self) -> List[FeatureProvider]:
        return list(self._providers)

    @property
    def locations(self):
        return self._obs.locations

    @property
    def species(self):
        return self._obs.species

    @property
    def species_subgroups(self):
        return self._obs.species_subgroups

    @property
    def locations_complete(self):
        return self._obs.locations_complete

    @property
    def species_complete(self):
        return self._obs.species_complete

    @property
    def species_subgroups_complete(self):
        return self._obs.species_subgroups_complete

    @property
    def years(self):
        return self._obs.years

    @property
    def observation_types(self):
        return self._obs.observation_types

    @property
    def num_observation_types(self):
        return self._obs.num_observation_types

    @property
    def bounding_box(self):
        return self._obs.bounding_box

    def get_location_coordinates(self, loc, from_index: bool = False):
        return self._obs.get_location_coordinates(loc, from_index=from_index)

    def get_location_name(self, loc, from_index: bool = False):
        return self._obs.get_location_name(loc, from_index=from_index)

    def observation_counts(self):
        return self._obs.observation_counts()

    # ------------------------------------------------------------------
    # Selection — delegate and re-wrap, preserving calendar + providers
    # ------------------------------------------------------------------

    def _wrap(self, obs: Observations) -> 'Dataset':
        return Dataset(
            obs,
            calendar=self._calendar,
            feature_providers=self._providers,
        )

    def select_locations(self, locations) -> 'Dataset':
        return self._wrap(self._obs.select_locations(locations))

    def select_years(self, years) -> 'Dataset':
        return self._wrap(self._obs.select_years(years))

    def select_species(self, species) -> 'Dataset':
        return self._wrap(self._obs.select_species(species))

    def select_by_observation_requirement(self, obs_key) -> 'Dataset':
        return self._wrap(self._obs.select_by_observation_requirement(obs_key))

    def select_by_local_num_observations(
        self, num_observations: int, obs_key: str
    ) -> 'Dataset':
        return self._wrap(
            self._obs.select_by_local_num_observations(num_observations, obs_key)
        )

    def select_by_ixs(self, ixs: List[Tuple]) -> 'Dataset':
        return self._wrap(self._obs.select_by_ixs(ixs))

    def aggregate_in_grid(self, method: str = 'median', grid_size=None) -> 'Dataset':
        return self._wrap(self._obs.aggregate_in_grid(method=method, grid_size=grid_size))

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_by_grid(
        self,
        grid_size: Tuple[float, float],
        split_size: float,
        shuffle: bool = True,
        random_state=None,
    ) -> Tuple['Dataset', 'Dataset', Dict[str, Any]]:
        obs_1, obs_2, info = self._obs.split_by_grid(
            grid_size, split_size, shuffle=shuffle, random_state=random_state
        )
        return self._wrap(obs_1), self._wrap(obs_2), info

    def split_by_lon_border(
        self, lon_border: float
    ) -> Tuple['Dataset', 'Dataset', Dict]:
        obs_1, obs_2, info = self._obs.split_by_lon_border(lon_border)
        return self._wrap(obs_1), self._wrap(obs_2), info

    def split_by_lat_border(
        self, lat_border: float
    ) -> Tuple['Dataset', 'Dataset', Dict]:
        obs_1, obs_2, info = self._obs.split_by_lat_border(lat_border)
        return self._wrap(obs_1), self._wrap(obs_2), info

    # ------------------------------------------------------------------
    # Feature downloading and statistics
    # ------------------------------------------------------------------

    def download_features(
        self, download_mode: str = None, verbose: bool = True
    ) -> None:
        """Download data for all attached providers and preload into memory.

        Calls ``provider.download(observations, ...)`` on each provider that
        implements a ``download`` method, then immediately preloads all feature
        data into RAM and closes the HDF5 store.  After this call the dataset
        is fully self-contained in memory, so multiple notebooks or processes
        can run concurrently without file-lock contention.

        Args:
            download_mode: Forwarded to each provider's ``download()`` call.
            verbose:       Show progress bars.
        """
        if not self._providers:
            raise DatasetException("No feature providers attached to this Dataset.")
        downloadable = [p for p in self._providers if hasattr(p, 'download')]
        if not downloadable:
            raise DatasetException(
                "None of the attached feature providers implement download()."
            )
        for provider in downloadable:
            provider.download(
                self._obs, download_mode=download_mode, verbose=verbose
            )
        self.preload_features(verbose=verbose)

    def preload_features(self, verbose: bool = True) -> None:
        """Preload all feature data into memory and close any backing stores.

        Calls ``provider.preload(observations, ...)`` on each attached provider
        that implements a ``preload`` method.  After this call, data is served
        entirely from in-memory caches, so multiple DataLoader workers can run
        concurrently without store-lock contention.

        Typical usage::

            ds.download_features()
            ds.preload_features()   # close store, keep data in RAM
            # safe to use with num_workers > 0

        Args:
            verbose: Show a tqdm progress bar per provider.
        """
        for provider in self._providers:
            if hasattr(provider, 'preload'):
                provider.preload(self._obs, verbose=verbose)

    def cache_data(self, verbose: bool = True, desc: Optional[str] = None) -> None:
        """Pre-warm the feature cache by iterating over all entries.

        Args:
            verbose: Show a progress bar.
            desc:    Custom label for the progress bar.
        """
        if verbose:
            iterator = tqdm(
                self.iter_items(),
                total=len(self),
                desc=desc or 'Caching data',
            )
        else:
            iterator = self.iter_items()
        for _ in iterator:
            pass

    def compute_feature_stats(
        self, verbose: bool = True
    ) -> Dict[str, Tuple[float, float]]:
        """Compute mean and standard deviation for each feature variable.

        Returns:
            ``{variable_name: (mean, std)}``
        """
        if not self._providers:
            raise DatasetException("No feature providers attached to this Dataset.")

        stats: Dict[str, list] = defaultdict(list)
        iterator = (
            tqdm(self.iter_items(), total=len(self), desc='Computing feature stats')
            if verbose
            else self.iter_items()
        )
        for sample in iterator:
            if KEY_FEATURES in sample:
                for key, arr in sample[KEY_FEATURES].items():
                    stats[key].append(arr)

        return {
            key: (float(np.mean(np.concatenate(arrs))), float(np.std(np.concatenate(arrs))))
            for key, arrs in stats.items()
        }

    # ------------------------------------------------------------------
    # Named-dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        key: str,
        calendar=None,
        feature_providers: Optional[Sequence[FeatureProvider]] = None,
        **kwargs,
    ) -> 'Dataset':
        """Load a pre-configured dataset by name.

        Dataset definitions live in :mod:`pysephone.dataset.registry`.
        Each entry is a callable that accepts ``**kwargs`` and returns an
        :class:`Observations` object.

        When a *calendar* is provided the registry's
        :data:`~pysephone.dataset.registry.CALENDAR_CONFIGS` entry (if any)
        is called to set species-specific season windows on the calendar
        before the Dataset is constructed.  You can override individual
        species afterwards with :meth:`Calendar.set_season`.

        Args:
            key:               Name of the dataset (e.g. ``'CPF_PEP725_winter_wheat'``).
            calendar:          Optional :class:`~pysephone.dataset.util.calendar.Calendar`
                               to attach.
            feature_providers: Optional list of feature providers to attach.
            **kwargs:          Forwarded to the dataset builder.

        Raises:
            DatasetException: If *key* is not found in the registry.
        """
        from pysephone.dataset.registry import REGISTRY, CALENDAR_CONFIGS

        if key not in REGISTRY:
            available = sorted(REGISTRY)
            raise DatasetException(
                f"Unknown dataset '{key}'. Available: {available}"
            )

        obs = REGISTRY[key](**kwargs)

        if calendar is not None and key in CALENDAR_CONFIGS:
            CALENDAR_CONFIGS[key](calendar)

        return Dataset(obs, calendar=calendar, feature_providers=feature_providers)

    @staticmethod
    def list_datasets() -> List[str]:
        """Return the names of all registered datasets."""
        from pysephone.dataset.registry import REGISTRY
        return sorted(REGISTRY)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge(a: 'Dataset', b: 'Dataset') -> 'Dataset':
        """Merge two datasets.  The calendar and feature providers of *a* are kept."""
        merged_obs = Observations.merge(a._obs, b._obs)
        return Dataset(merged_obs, calendar=a._calendar, feature_providers=a._providers)
