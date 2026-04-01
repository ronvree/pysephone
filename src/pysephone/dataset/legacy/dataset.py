from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from phenology import config
from phenology.data.openmeteo.download import OpenMeteoEntry, get_openmeteo_data
from phenology.dataset.base import BaseDataset
from phenology.dataset.util.calendar import Calendar
from phenology.dataset.util.openmeteo import OpenMeteoDataset


class Dataset(BaseDataset):

    DEFAULT_DATA_KEYS = ('temperature_2m_mean', 'daylight_duration')
    DEFAULT_STEP = 'daily'

    def __init__(self,
                 df_y: pd.DataFrame,
                 df_y_loc: pd.DataFrame,
                 species: list = None,
                 species_subgroups: list = None,
                 locations: list = None,
                 data_openmeteo: Optional[OpenMeteoDataset] = None,
                 calendar: Optional[Calendar] = None,
                 mode_download_meteo: Optional[str] = None,
                 data_keys: Optional[Tuple[str, ...]] = None,
                 step: Optional[str] = None,
                 match_observation_ixs_to_step: bool = False,
                 debug_mode: bool = False,
                 ) -> None:
        """
        Initialize a Dataset object that extends BaseDataset with meteorological features.
        
        This class integrates OpenMeteo weather data with phenology observations, providing
        time-series features for each observation entry. The preferred way to create a
        Dataset is through Dataset.load().
        
        Args:
            df_y: DataFrame containing phenology observations (inherited from BaseDataset).
            df_y_loc: DataFrame containing location data (inherited from BaseDataset).
            data_openmeteo: Pre-initialized OpenMeteoDataset. If None, will be created automatically.
            calendar: Calendar instance for season calculations. If None, uses Calendar.from_config().
            mode_download_meteo: Download mode for OpenMeteo data ('skip', 'download', 'force', etc.).
                                Only used if data_openmeteo is None.
            data_keys: Tuple of feature keys to load (e.g., ('temperature_2m_mean', 'daylight_duration')).
                      If None, uses DEFAULT_DATA_KEYS.
            step: Time step for features ('hourly' or 'daily'). If None, uses DEFAULT_STEP.
            match_observation_ixs_to_step: If True, observation indices match feature time step.
                                          If False, indices are always in days.
            debug_mode: If True, uses dummy data instead of downloading real weather data.
        
        Raises:
            ValueError: If step is not None, 'hourly', or 'daily'.
            DatasetException: If dataframes don't meet required structure (from BaseDataset).
        """
        if step is not None and step not in ('hourly', 'daily'):
            raise ValueError(f"step must be 'hourly' or 'daily', got '{step}'")
        super().__init__(df_y, df_y_loc, species=species, species_subgroups=species_subgroups, locations=locations)
        self._debug_mode = debug_mode
        self._match_obs_ixs_to_step = match_observation_ixs_to_step

        if data_keys is None:
            self._data_keys = self.DEFAULT_DATA_KEYS
        else:
            self._data_keys = data_keys

        if step is None:
            self._step = self.DEFAULT_STEP
        else:
            self._step = step

        self._calendar = calendar or Calendar.from_config()

        if data_openmeteo is None:
            self._obtain_openmeteo_data(mode_download_meteo)  # TODO -- move to preprocessing
            self._data_om = OpenMeteoDataset(
                calendar=self._calendar,
                data_keys=list(self._data_keys),
                step=self._step,
                debug_mode=debug_mode,
            )
        else:
            self._data_om = data_openmeteo


    def _obtain_openmeteo_data(self, download_mode: Optional[str]) -> None:
        """
        Download OpenMeteo data for all required entries.
        
        This method collects all required OpenMeteo entries based on the dataset's
        observations and downloads the corresponding weather data. It determines
        the year range for each entry based on season boundaries.
        
        Args:
            download_mode: Download mode ('skip', 'download', 'force', etc.).
                          If None, no download is performed.
        
        Note:
            This method should ideally be moved to preprocessing as indicated by the TODO.
        """
        
        entries = set()

        for data_key in self._data_keys:

            for index in self.iter_index():
                src, loc_id, year, species_code, subgroup_code = index

                season = self._calendar.get_season_info(year=year,
                                                        species_code=species_code,
                                                        subgroup_code=subgroup_code,
                                                        src=src,
                                                        loc_id=loc_id,
                                                        )

                season_start = season['season_start']
                season_end = season['season_end']

                coords = self.get_location_coordinates(i=(src, loc_id))
                # cc, _ = self.get_location_country(i=(src, loc_id))

                EPOCH_YEAR = 1970
                year_min = season_start.astype('datetime64[Y]').astype(int) + EPOCH_YEAR
                year_max = season_end.astype('datetime64[Y]').astype(int) + EPOCH_YEAR

                for year_ in range(year_min, year_max + 1):

                    entries.add(
                        OpenMeteoEntry(
                            step=self._step,
                            data_key=data_key,
                            src_key=src,
                            loc_id=loc_id,
                            year=year_,
                            loc_name=self.get_location_name(i=(src, loc_id)),
                            lat=coords['lat'],
                            lon=coords['lon'],
                            # country_code=cc,
                        )
                    )

        get_openmeteo_data(entries=entries,
                           verbose=True,
                           download_mode=download_mode,
                           )

    def __enter__(self) -> 'Dataset':
        """
        Context manager entry.
        
        Allows using the dataset as a context manager:
        with Dataset.load(...) as dataset:
            # use dataset
        
        Returns:
            Self instance.
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], 
                 exc_tb: Optional[Any]) -> None:
        """
        Context manager exit.
        
        Ensures resources are properly closed when exiting the context.
        
        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_val: Exception value if an exception occurred, None otherwise.
            exc_tb: Exception traceback if an exception occurred, None otherwise.
        """
        self.close()

    def close(self) -> None:
        """
        Close the dataset and release resources.
        
        This method closes the underlying OpenMeteoDataset connection.
        It should be called when the dataset is no longer needed, or use
        the context manager syntax for automatic cleanup.
        
        After calling close(), the dataset should not be used further.
        
        Example:
            >>> dataset = Dataset.load(...)
            >>> # ... use dataset ...
            >>> dataset.close()
            
            # Or use context manager:
            >>> with Dataset.load(...) as dataset:
            ...     # ... use dataset ...
            ...     pass  # Automatically closed on exit
        """
        if self._data_om is not None:
            self._data_om.close()

    def __getitem__(self, index: Union[int, Tuple]) -> Dict[str, Any]:
        """
        Get a dataset entry with observations and meteorological features.
        
        Args:
            index: Either an integer position or a tuple index.
        
        Returns:
            Dictionary containing:
            - All fields from BaseDataset.__getitem__()
            - 'season_start': Start datetime of the season
            - 'season_end': End datetime of the season (inclusive)
            - config.KEY_OBSERVATIONS_INDEX: Dictionary mapping observation types to indices
            - config.KEY_FEATURES: Dictionary mapping feature keys to numpy arrays
        
        Raises:
            DatasetException: If observations or features cannot be retrieved.
        """
        # Obtain the observations as defined by the super class
        item = super().__getitem__(index)

        # Obtain info from the data sample index
        src = item[config.KEY_DATA_SOURCE]
        loc_id = item[config.KEY_LOC_ID]
        year = item[config.KEY_YEAR]
        species_code = item[config.KEY_SPECIES_CODE]
        subgroup_code = item[config.KEY_SUBGROUP_CODE]

        index = src, loc_id, year, species_code, subgroup_code

        # Obtain the observations
        observations = item[config.KEY_OBSERVATIONS]

        # Obtain season info
        season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)

        # Compute the start and end moment of the season
        # Start and end moment correspond to the first and last element in the time series features
        # (i.e. season end is included)
        step_fmt = 'h' if self._step == 'hourly' else 'D'
        # Get season start
        season_start = season['season_start']
        # Get season end (calendar entry marks the day at which the season has ended)
        # Subtract one time unit to make the datetime object match the last entry in the time series data
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)

        # Provide observations as indices within the season, next to datetime objects
        # Depending on the configuration of this object, these indices are based on nr of days or the same time step as
        # the feature data
        step_fmt_obs = step_fmt if self._match_obs_ixs_to_step else 'D'
        obs_ixs = {
            key: (o - season_start) // np.timedelta64(1, step_fmt_obs) for key, o in observations.items()
        }

        # Return all relevant data
        return {
            **item,
            'season_start': season_start,
            'season_end': season_end,
            config.KEY_OBSERVATIONS_INDEX: obs_ixs,
            config.KEY_FEATURES: {
                key: self._get_feature(self._step, key, index) for key in self._data_keys
            },
        }

    def _get_feature(self, step: str, data_key: str, index: Tuple[str, str, int, int, int]) -> np.ndarray:
        """
        Retrieve a meteorological feature for a specific entry.
        
        Args:
            step: Time step ('hourly' or 'daily').
            data_key: Data key identifying the meteorological variable.
            index: Tuple containing (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            NumPy array containing the time-series feature data.
        """
        return self._data_om.get_data(step, data_key, index)
        #
        # src, loc_id, year, species_code, subgroup_code = index
        #
        # season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)
        #
        # step_fmt = 'h' if self._step == 'hourly' else 'D'
        #
        # season_start = season['season_start']
        # season_end = season['season_end'] - np.timedelta64(1, step_fmt)
        #
        # if self._debug_mode:
        #     return np.zeros((season_end - season_start) // np.timedelta64(1, step_fmt))
        #
        # year_min = season_start.astype('datetime64[Y]').astype(int) + 1970
        # year_max = season_end.astype('datetime64[Y]').astype(int) + 1970
        #
        # season_start = pd.Timestamp(season_start, tz='UTC')
        # season_end = pd.Timestamp(season_end, tz='UTC')
        #
        # df_x = pd.concat([self._store[step, data_key, src, loc_id, year_] for year_ in range(year_min, year_max + 1)],
        #                  axis=0,
        #                  )
        # df_x.set_index('date', inplace=True)
        #
        # df_x = df_x[season_start:season_end]
        #
        # return df_x[data_key].values

    def _get_season_info(self, src: str, loc_id: str, year: int, 
                         species_code: int, subgroup_code: int) -> Dict[str, Any]:
        """
        Get season information for a specific entry.
        
        This is a wrapper around Calendar.get_season_info() that provides season
        boundaries needed for data retrieval. The Calendar class already implements
        caching, so this method is primarily for convenience.
        
        Args:
            src: Data source identifier.
            loc_id: Location identifier.
            year: Year of observation.
            species_code: Species code.
            subgroup_code: Subgroup code.
        
        Returns:
            Dictionary containing season information with keys:
            - 'season_start': Start datetime of the season
            - 'season_end': End datetime of the season
            - 'season_length': Duration of the season
            - 'year': Year of observation
        
        Raises:
            KeyError: If season information cannot be found for the given parameters.
        """
        season = self._calendar.get_season_info(
            year=year,
            species_code=species_code,
            subgroup_code=subgroup_code,
            src=src,
            loc_id=loc_id,
        )
        return season

    def cache_data(self, verbose: bool = True, desc: Optional[str] = None) -> None:
        """
        Load all to-be-cached data in memory by iterating over the dataset.
        
        This method iterates through all entries in the dataset, which causes
        the underlying OpenMeteoDataset to cache the retrieved data. This is
        useful for pre-warming the cache before intensive operations.
        
        Args:
            verbose: If True, displays a progress bar during caching.
            desc: Custom description for the progress bar. If None, uses 'Caching data'.
        
        Example:
            >>> dataset.cache_data(verbose=True, desc='Pre-warming cache')
        """
        if verbose:
            if desc is None:
                desc = 'Caching data'
            for _ in tqdm(self.iter_items(),
                          total=len(self),
                          desc=desc,
                          ):
                pass
        else:
            for _ in self.iter_items():
                pass

    @property
    def step(self) -> str:
        return self._step

    @staticmethod
    def _from_base(dataset: BaseDataset, **kwargs) -> 'Dataset':
        """
        Create a Dataset from a BaseDataset with additional keyword arguments.
        
        Args:
            dataset: BaseDataset instance to convert.
            **kwargs: Additional arguments to pass to Dataset.__init__.
        
        Returns:
            A new Dataset instance.
        """
        return Dataset(
            dataset._df_y,
            dataset._df_y_loc,
            species=dataset.species_complete,
            locations=dataset.locations_complete,
            **kwargs,
        )

    @staticmethod
    def load(key: str, **kwargs) -> 'Dataset':
        """
        Load a pre-configured dataset based on its key/name.
        
        This is the preferred method for creating Dataset instances, as it handles
        all the necessary preprocessing, filtering, and aggregation steps.
        
        Args:
            key: The key/name of the dataset to load. Available keys are the same
                 as those for BaseDataset.load().
            **kwargs: Additional arguments to pass to Dataset.__init__, such as:
                     - step: Time step ('hourly' or 'daily')
                     - data_keys: Tuple of feature keys
                     - mode_download_meteo: Download mode for OpenMeteo data
                     - match_observation_ixs_to_step: Whether to match observation indices to step
                     - debug_mode: Whether to use debug mode
        
        Returns:
            A Dataset instance configured according to the specified key.
        
        Raises:
            DatasetException: If the dataset key is not recognized.
        
        Example:
            >>> dataset = Dataset.load('PEP725_Apple',
            ...                        step='hourly',
            ...                        data_keys=('temperature_2m', 'precipitation'))
        """
        return Dataset._from_base(
            BaseDataset.load(key),
            debug_mode=key == 'debug',
            **kwargs,
        )

    def _copy_from_base(self, dataset: BaseDataset) -> 'Dataset':
        """
        Create a new Dataset from a BaseDataset, preserving Dataset-specific configuration.
        
        Args:
            dataset: BaseDataset instance to convert.
        
        Returns:
            New Dataset instance with same configuration as self.
        
        Raises:
            TypeError: If dataset is not a BaseDataset instance.
        """
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"Expected BaseDataset, got {type(dataset)}")
        
        return Dataset._from_base(
            dataset,
            data_openmeteo=self._data_om,
            calendar=self._calendar,
            debug_mode=self._debug_mode,
            step=self._step,
            data_keys=self._data_keys,
            match_observation_ixs_to_step=self._match_obs_ixs_to_step,
        )

    def select_locations(self, locations: Union[Tuple[str, str], List[Tuple[str, str]]]) -> 'Dataset':
        """
        Create a new Dataset with only the specified locations.
        
        Args:
            locations: The locations to select. Can be specified using either:
                - A two-tuple containing (data source key, loc_id)
                - A list of two-tuples (for selecting multiple locations)
        
        Returns:
            A new Dataset object containing only the specified locations.
        """
        base = super().select_locations(locations)
        return self._copy_from_base(base)

    def select_years(self, years: Union[int, List[int]]) -> 'Dataset':
        """
        Create a new Dataset with only the specified years.
        
        Args:
            years: The years to select. Can be specified using either:
                - A single year (as integer)
                - A list of years
        
        Returns:
            A new Dataset object containing only the specified years.
        """
        base = super().select_years(years)
        return self._copy_from_base(base)

    def select_species(self, species: Union[Tuple, List[Tuple]]) -> 'Dataset':
        base = super().select_species(species)
        return self._copy_from_base(base)

    def select_by_observation_requirement(self, obs_key: Union[str, List[str]]) -> 'Dataset':
        """
        Create a new Dataset containing only entries that have all specified observation types.
        
        Args:
            obs_key: Single observation type key (str) or list of observation type keys.
                    Only entries that have ALL specified observation types will be included.
        
        Returns:
            A new Dataset object containing only entries with all required observation types.
        """
        base = super().select_by_observation_requirement(obs_key)
        return self._copy_from_base(base)

    def select_by_local_num_observations(self, num_observations: int, obs_key: str) -> 'Dataset':
        """
        Create a new Dataset containing only entries with at least the specified number
        of observations of a given type.
        
        Args:
            num_observations: Minimum number of observations required (must be >= 0).
            obs_key: Observation type key to filter on.
        
        Returns:
            A new Dataset object containing only entries meeting the requirement.
        """
        base = super().select_by_local_num_observations(num_observations, obs_key)
        return self._copy_from_base(base)

    def select_by_ixs(self, ixs: List[Tuple]) -> 'Dataset':
        """
        Create a new Dataset containing only the specified base indices.
        
        Args:
            ixs: List of base index tuples (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            A new Dataset object containing only the specified indices.
        """
        base = super().select_by_ixs(ixs)
        return self._copy_from_base(base)

    # def select_by_position_ixs(self, ixs: list) -> 'Dataset':
    #     base = super().select_by_position_ixs(ixs)
    #     return self._copy_from_base(base)

    def aggregate_in_grid(self, method: str = 'mean', 
                          grid_size: Optional[Tuple[float, float]] = None) -> 'Dataset':
        """
        Divide the dataset into grid cells and aggregate observations within each cell.
        
        Args:
            method: Method of aggregation/selection. Options: 'median' (default), 'mean', 'first'.
            grid_size: Tuple of (grid latitude cell size, grid longitude cell size) in degrees.
                      If None, uses config.MIN_GRID_SIZE.
        
        Returns:
            A new Dataset object with aggregated observations.
        """
        base = super().aggregate_in_grid(method=method, grid_size=grid_size)
        return self._copy_from_base(base)

    def compute_feature_stats(self, verbose: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean and standard deviation statistics for each feature across the dataset.
        
        Iterates through all samples in the dataset and computes aggregate statistics
        for each feature key. In debug mode, returns dummy statistics (0, 1) for all features.
        
        Args:
            verbose: If True, displays a progress bar during computation.
        
        Returns:
            Dictionary mapping each feature key to a tuple of (mean, std_dev).
        
        Example:
            >>> stats = dataset.compute_feature_stats()
            >>> print(stats['temperature_2m_mean'])
            (15.3, 5.2)  # (mean, std)
        """

        if self._debug_mode:  # If debug_mode -> skip the costly statistic computation
            return {k: (0, 1) for k in self._data_keys}

        stats = defaultdict(list)
        # Collect all values of all features
        if verbose:
            iter_samples = tqdm(self.iter_items(), total=len(self), desc='Computing feature statistics')
        else:
            iter_samples = self.iter_items()

        for sample in iter_samples:
            fs = sample[config.KEY_FEATURES]
            for key, f in fs.items():
                stats[key].append(f)
        # Concatenate them to one array
        stats = {
            key: np.concatenate(f) for key, f in stats.items()
        }
        # Compute statistics on the arrays
        stats = {
            key: (np.mean(f), np.std(f)) for key, f in stats.items()
        }
        # print(stats)
        return stats


if __name__ == '__main__':

    from tqdm import tqdm
    import time


    _time_start = time.time()

    # _dataset = Dataset.load('all_fruit_trees',
    #                         mode_download_meteo='skip',
    #                         step='daily',
    #                         )

    _dataset = Dataset.load(
        key='PEP725_fruit_trees',
        mode_download_meteo = 'skip',
        step = 'daily',
    )

    _ys = set()
    _ss = set()
    for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
        _fs = _x[config.KEY_FEATURES]
        for _f in _fs.values():
            if np.isnan(_f).sum() > 0:
                print(_x[config.KEY_YEAR])
                print(np.isnan(_f).sum())
                print(_x[config.KEY_SPECIES_CODE])
                _ys.add(_x[config.KEY_YEAR])
                _ss.add(_x[config.KEY_SPECIES_CODE])

    print(_ys)
    print(_ss)

    print('seconds: ', time.time() - _time_start)

    # print('Iter 1')
    # for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
    #     pass
    # print('Iter 2')
    # for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
    #     pass

    _dataset.close()

