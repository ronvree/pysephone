from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd

from phenology.data.openmeteo.download import OpenMeteoStores
from phenology.dataset.util.calendar import Calendar


class OpenMeteoDataset:
    """
    Dataset class for accessing OpenMeteo meteorological data.
    
    This class provides access to weather data stored in HDF5 files through OpenMeteoStores.
    It integrates with the Calendar system to determine season boundaries and retrieves
    time-series meteorological features for specific locations and time periods.
    
    The class implements caching to avoid redundant data retrieval and supports both
    normal operation and debug mode (which returns dummy data).
    
    Attributes:
        _debug_mode: If True, returns dummy data instead of real weather data.
        _calendar: Calendar instance for determining season boundaries.
        _cache: Dictionary cache for storing retrieved data.
        _store: OpenMeteoStores instance for accessing HDF5 data stores.
    
    Example:
        >>> calendar = Calendar.from_config()
        >>> dataset = OpenMeteoDataset(
        ...     calendar=calendar,
        ...     step='daily',
        ...     data_keys=['temperature_2m_mean', 'daylight_duration'],
        ...     debug_mode=False
        ... )
        >>> # Get data for a specific entry
        >>> data = dataset.get_data('daily', 'temperature_2m_mean', 
        ...                         ('PEP725', '12345', 2020, 1, 0))
        >>> # Or use dictionary-style access
        >>> data = dataset[('daily', 'temperature_2m_mean', 
        ...                 ('PEP725', '12345', 2020, 1, 0))]
        >>> dataset.close()
    """

    def __init__(self,
                 calendar: Calendar,
                 step: Optional[str] = None,
                 data_keys: Optional[List[str]] = None,
                 debug_mode: bool = False,
                 ) -> None:
        """
        Initialize an OpenMeteoDataset instance.
        
        Args:
            calendar: Calendar instance used to determine season boundaries for data retrieval.
            step: Time step for the data ('hourly' or 'daily'). Required if debug_mode is False.
            data_keys: List of data keys to access (e.g., ['temperature_2m_mean', 'daylight_duration']).
                      Required if debug_mode is False.
            debug_mode: If True, operates in debug mode and returns dummy data instead of
                        accessing real weather data. Defaults to False.
        
        Raises:
            ValueError: If step or data_keys is None when debug_mode is False.
            ValueError: If data_keys is empty when debug_mode is False.
        """

        self._debug_mode = debug_mode
        self._calendar = calendar

        # Initialize dictionary cache for storing retrieved data
        self._cache: Dict[Tuple[str, str, Tuple], np.ndarray] = {}

        if not self._debug_mode:
            # Validate required parameters
            if step is None:
                raise ValueError("step cannot be None when debug_mode is False")
            
            # Create step-key pairs for OpenMeteoStores
            step_key_pairs = list(zip([step] * len(data_keys), data_keys))
            self._store = OpenMeteoStores(step_key_pairs)
        else:
            self._store = None

    def get_data(self, step: str, key: str, index: Tuple) -> np.ndarray:
        """
        Retrieve meteorological data for a specific entry.
        
        This method implements caching to avoid redundant data retrieval. If the data
        has been requested before, it returns the cached result. Otherwise, it retrieves
        the data from the store (or generates dummy data in debug mode) and caches it.
        
        Args:
            step: Time step for the data ('hourly' or 'daily').
            key: Data key identifying the meteorological variable (e.g., 'temperature_2m_mean').
            index: Tuple containing (src, loc_id, year, species_code, subgroup_code) identifying
                   the specific entry for which to retrieve data.
        
        Returns:
            NumPy array containing the time-series data for the specified entry.
            The array length corresponds to the season length in the appropriate time units.
        
        Raises:
            KeyError: If the requested data is not found in the store (when not in debug mode).
            ValueError: If step is not 'hourly' or 'daily'.
        
        Example:
            >>> data = dataset.get_data('daily', 'temperature_2m_mean', 
            ...                        ('PEP725', '12345', 2020, 1, 0))
            >>> print(data.shape)  # Shape depends on season length
        """
        # Create cache key from arguments
        cache_key = (step, key, index)
        
        # Check if result is in cache
        if cache_key in self._cache:
            return self._cache[cache_key].copy()  # Return copy to prevent mutation

        # Retrieve data (from store or generate dummy data)
        if self._debug_mode:
            result = self._get_debug_data(step, key, index)
        else:
            result = self._get_from_store(step, key, index)
        
        # Store copy in cache for future use
        self._cache[cache_key] = result.copy()
        
        return result

    def _calculate_season_bounds(self, step: str, index: Tuple[str, str, int, int, int]) -> Tuple[np.datetime64, np.datetime64, str]:
        """
        Calculate season start, end, and step format for a given entry.
        
        This helper method extracts the common season calculation logic used by
        both _get_debug_data and _get_from_store to avoid code duplication.
        
        Args:
            step: Time step ('hourly' or 'daily').
            index: Tuple containing (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            Tuple of (season_start, season_end, step_fmt) where:
            - season_start: Start datetime of the season
            - season_end: End datetime of the season (adjusted by one time unit)
            - step_fmt: Format string for timedelta ('h' for hourly, 'D' for daily)
        
        Raises:
            KeyError: If season information cannot be found for the given parameters.
        """
        src, loc_id, year, species_code, subgroup_code = index
        season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)
        
        step_fmt = 'h' if step == 'hourly' else 'D'
        season_start = season['season_start']
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)
        
        return season_start, season_end, step_fmt

    def _get_debug_data(self, step: str, data_key: str, index: Tuple[str, str, int, int, int]) -> np.ndarray:
        """
        Generate dummy data for debug mode.
        
        Returns an array of zeros with the same shape as real data would have,
        based on the season length for the given entry. This is useful for testing
        without requiring actual weather data.
        
        Args:
            step: Time step ('hourly' or 'daily').
            data_key: Data key (unused in debug mode, kept for interface consistency).
            index: Tuple containing (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            NumPy array of zeros with length matching the season duration in the
            appropriate time units.
        """
        season_start, season_end, step_fmt = self._calculate_season_bounds(step, index)
        
        # Calculate array size and ensure it's an integer
        duration = (season_end - season_start) // np.timedelta64(1, step_fmt)
        return np.zeros(int(duration))

    def _get_from_store(self, step: str, data_key: str, index: Tuple[str, str, int, int, int]) -> np.ndarray:
        """
        Retrieve meteorological data from the HDF5 store.
        
        This method retrieves data from the OpenMeteoStores for the specified entry.
        It handles cases where the season spans multiple calendar years by concatenating
        data from all relevant years. The data is then filtered to the exact season
        boundaries and returned as a NumPy array.
        
        Args:
            step: Time step ('hourly' or 'daily').
            data_key: Data key identifying the meteorological variable.
            index: Tuple containing (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            NumPy array containing the time-series data for the season period.
            Missing values are forward-filled and backward-filled before returning.
        
        Raises:
            KeyError: If the requested data is not found in the store.
            KeyError: If data_key column is not present in the retrieved DataFrame.
            ValueError: If step is not 'hourly' or 'daily'.
        
        Note:
            The bfill().ffill() operation fills missing values and should ideally
            be moved to postprocessing (see TODO comment).
        """
        src, loc_id, year, species_code, subgroup_code = index
        
        # Get season info for year range calculation (need original season_start)
        season_info = self._get_season_info(src, loc_id, year, species_code, subgroup_code)
        season_start_orig = season_info['season_start']
        
        # Use extracted season calculation logic for adjusted bounds
        season_start, season_end, step_fmt = self._calculate_season_bounds(step, index)

        # Calculate year range (season may span multiple calendar years)
        EPOCH_YEAR = 1970
        year_min = season_start_orig.astype('datetime64[Y]').astype(int) + EPOCH_YEAR
        year_max = season_end.astype('datetime64[Y]').astype(int) + EPOCH_YEAR

        # Convert to pandas Timestamps for DataFrame indexing
        season_start = pd.Timestamp(season_start, tz='UTC')
        season_end = pd.Timestamp(season_end, tz='UTC')

        # Retrieve and concatenate data for all years in the season
        # TODO: bfill().ffill() should be moved to postprocessing
        try:
            dfs = []
            for year_ in range(year_min, year_max + 1):
                df_year = self._store[step, data_key, src, loc_id, year_]
                dfs.append(df_year.bfill().ffill())
            
            df_x = pd.concat(dfs, axis=0)
            df_x.set_index('date', inplace=True)

            # Filter to exact season boundaries
            df_x = df_x[season_start:season_end]

            # Extract the requested data column
            if data_key not in df_x.columns:
                raise KeyError(
                    f"Data key '{data_key}' not found in DataFrame. "
                    f"Available columns: {df_x.columns.tolist()}"
                )
            
            return df_x[data_key].values
        except KeyError as e:
            raise KeyError(
                f"Failed to retrieve data from store for "
                f"step={step}, key={data_key}, src={src}, loc_id={loc_id}, "
                f"year range={year_min}-{year_max}: {e}"
            )

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

    def __getitem__(self, i: Tuple) -> np.ndarray:
        """
        Dictionary-style access to get_data method.
        
        Allows accessing data using dictionary-style syntax:
        dataset[('step', 'key', index)] instead of dataset.get_data('step', 'key', index)
        
        Args:
            i: Tuple of (step, key, index) where:
               - step: Time step ('hourly' or 'daily')
               - key: Data key
               - index: Tuple (src, loc_id, year, species_code, subgroup_code)
        
        Returns:
            NumPy array containing the time-series data.
        
        Raises:
            ValueError: If the input tuple does not have exactly 3 elements.
            KeyError: If the requested data is not found (when not in debug mode).
        
        Example:
            >>> data = dataset[('daily', 'temperature_2m_mean', 
            ...                ('PEP725', '12345', 2020, 1, 0))]
        """
        if not isinstance(i, tuple) or len(i) != 3:
            raise ValueError(
                f"Expected tuple of length 3 (step, key, index), got {i}"
            )
        step, key, index = i
        return self.get_data(step, key, index)

    def __enter__(self) -> 'OpenMeteoDataset':
        """
        Context manager entry.
        
        Allows using the dataset as a context manager:
        with OpenMeteoDataset(...) as dataset:
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

    def clear_cache(self) -> None:
        """
        Clear the entire cache.
        
        This method removes all cached data entries, freeing memory.
        Useful when you want to force fresh data retrieval or reduce memory usage.
        
        Example:
            >>> dataset.clear_cache()
            >>> print(dataset.get_cache_size())  # 0
        """
        self._cache.clear()

    def get_cache_size(self) -> int:
        """
        Get the current number of cached entries.
        
        Returns:
            Number of entries currently stored in the cache.
        
        Example:
            >>> size = dataset.get_cache_size()
            >>> print(f"Cache contains {size} entries")
        """
        return len(self._cache)

    def cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary containing cache information with keys:
            - 'size': Number of cached entries
            - 'keys': List of cache keys (for debugging)
        
        Example:
            >>> info = dataset.cache_info()
            >>> print(f"Cache size: {info['size']}")
        """
        return {
            'size': len(self._cache),
            'keys': list(self._cache.keys()),
        }

    def close(self) -> None:
        """
        Close the dataset and release resources.
        
        This method closes the underlying OpenMeteoStores connection and clears
        the cache to free memory. It should be called when the dataset is no longer
        needed, or use the context manager syntax for automatic cleanup.
        
        After calling close(), the dataset should not be used further.
        
        Example:
            >>> dataset = OpenMeteoDataset(...)
            >>> # ... use dataset ...
            >>> dataset.close()
            
            # Or use context manager:
            >>> with OpenMeteoDataset(...) as dataset:
            ...     # ... use dataset ...
            ...     pass  # Automatically closed on exit
        """
        if self._store is not None:
            self._store.close()
        self._store = None
        # Clear cache to free memory
        self.clear_cache()
