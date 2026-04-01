import os
from datetime import datetime
from functools import reduce, lru_cache

from collections import defaultdict
from typing import Iterator, Union, List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from rasterio.rio.helpers import coords

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from phenology import config
from phenology.data.resources.load_admin_boundaries import load_admin_boundaries
from phenology.dataset.preprocessing.pep725 import get_pep725_dataframes
from phenology.dataset.preprocessing.gmu_cherry import get_gmu_cherry_dataset_japan, get_gmu_cherry_dataset_switzerland, get_gmu_cherry_dataset_south_korea
from phenology.dataset.util import DatasetException
from phenology.dataset.util.func_pandas import select_location, select_year, empty_df_like, select_species

from phenology.util.func import round_partial


class BaseDataset:

    def __init__(self,
                 df_y: pd.DataFrame,
                 df_y_loc: pd.DataFrame,
                 species: list = None,
                 species_subgroups: list = None,
                 locations: list = None,
                 ) -> None:
        """
        Create a dataset for conveniently accessing phenology observations.
        
        Args:
            df_y: A dataframe containing phenology observations.
                  The dataframe should be indexed by (data src, loc_id, year, species_code, subgroup_code, obs_type).
                  The dataframe should have a column containing observations (config.KEY_OBSERVATIONS).
            df_y_loc: A dataframe containing location data.
                      The dataframe should be indexed by (data src, loc_id).
                      The dataframe should contain columns: lat, lon, and optionally loc_name, alt, country_code.
        
        Raises:
            DatasetException: If the dataframes don't meet the required structure.
        """

        self._df_y = df_y
        self._df_y_loc = df_y_loc
        self._validate_dfs()

        # Sort data for faster lookups
        self._df_y.sort_index(inplace=True)
        self._df_y_loc.sort_index(inplace=True)

        # Create a shared index between all dataframes
        self._index = self._reset_index()

        if species is None:
            self._species = self.species
        else:
            self._species = list(species)

        if species_subgroups is None:
            self._species_subgroups = self.species_subgroups
        else:
            self._species_subgroups = list(species_subgroups)

        if locations is None:
            self._locations = self.locations
        else:
            self._locations = list(locations)

    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        return len(self._index)

    def __contains__(self, index: Union[int, Tuple]) -> bool:
        """
        Check if an index exists in the dataset.
        
        Args:
            index: Either an integer position or a tuple index.
        
        Returns:
            True if the index exists, False otherwise.
        """
        return index in self._index

    @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def __getitem__(self, index: Union[int, Tuple]) -> Dict[str, Any]:
        """
        Get a dataset entry by index.
        
        Args:
            index: Either an integer position in the dataset or a tuple index
                   (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            A dictionary containing:
            - config.KEY_DATA_SOURCE: Data source identifier
            - config.KEY_LOC_ID: Location identifier
            - config.KEY_YEAR: Year of observation
            - config.KEY_SPECIES_CODE: Species code
            - config.KEY_SUBGROUP_CODE: Subgroup code
            - config.KEY_OBSERVATIONS: Dictionary of observation types to values
        
        Raises:
            IndexError: If integer index is out of range.
            ValueError: If tuple index has wrong length.
            DatasetException: If index type is unsupported or observations not found.
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self._index):
                raise IndexError(f'Index {index} out of range for dataset of size {len(self._index)}')
            index = self._index[index]
            src, loc_id, year, species_code, subgroup_code = index

        elif isinstance(index, tuple):
            if len(index) != len(config.KEYS_INDEX):
                raise ValueError(f'Index tuple must have length {len(config.KEYS_INDEX)}, got {len(index)}')
            src, loc_id, year, species_code, subgroup_code = index

        else:
            raise DatasetException(f'Unsupported index type {type(index)}')

        try:
            df_obs = self._df_y.loc[src, loc_id, year, species_code, subgroup_code]
            observations = df_obs.to_dict()[config.KEY_OBSERVATIONS]
        except KeyError as e:
            raise DatasetException(f'No observations found for index {(src, loc_id, year, species_code, subgroup_code)}: {e}')

        coords = self.get_location_coordinates(index, from_index=True)

        return {
            config.KEY_DATA_SOURCE: src,
            config.KEY_LOC_ID: loc_id,
            config.KEY_YEAR: year,
            config.KEY_SPECIES_CODE: species_code,
            config.KEY_SUBGROUP_CODE: subgroup_code,
            config.KEY_OBSERVATIONS: observations,
            config.KEY_LAT: coords[config.KEY_LAT],
            config.KEY_LON: coords[config.KEY_LON],
        }

    def _validate_dfs(self) -> None:
        """
        Validate that dataframes have the expected structure.
        
        Raises:
            DatasetException: If dataframes don't meet the required structure.
        """
        if self._df_y.empty:
            raise DatasetException("df_y cannot be empty")
        if self._df_y_loc.empty:
            raise DatasetException("df_y_loc cannot be empty")
        
        # Check required index levels for df_y
        required_levels = list(config.KEYS_INDEX) + [config.KEY_OBS_TYPE]
        if not all(level in self._df_y.index.names for level in required_levels):
            missing = [l for l in required_levels if l not in self._df_y.index.names]
            raise DatasetException(f"df_y missing required index levels: {missing}")
        
        # Check required columns for df_y
        if config.KEY_OBSERVATIONS not in self._df_y.columns:
            raise DatasetException(f"df_y missing required column: {config.KEY_OBSERVATIONS}")
        
        # Check location dataframe index
        required_loc_levels = [config.KEY_DATA_SOURCE, config.KEY_LOC_ID]
        if not all(level in self._df_y_loc.index.names for level in required_loc_levels):
            missing = [l for l in required_loc_levels if l not in self._df_y_loc.index.names]
            raise DatasetException(f"df_y_loc missing required index levels: {missing}")
        
        # Check location dataframe columns
        required_loc_cols = [config.KEY_LAT, config.KEY_LON]
        if not all(col in self._df_y_loc.columns for col in required_loc_cols):
            missing = [c for c in required_loc_cols if c not in self._df_y_loc.columns]
            raise DatasetException(f"df_y_loc missing required columns: {missing}")

    def _reset_index(self) -> pd.MultiIndex:
        """
        Create a shared index between all observation types.
        
        Groups observations by observation type and takes the union of their indices,
        creating a base index that represents all unique combinations of
        (data source, location, year, species, subgroup) present in the dataset.
        
        Returns:
            A sorted MultiIndex containing all unique base indices.
        """
        # Group observations by observation type (i.e. which phenological stage was observed)
        groups = self._df_y.groupby(level=config.KEY_OBS_TYPE)

        # Create an empty index before calling reduce (in case the dataset is empty)
        eix = pd.MultiIndex.from_tuples([], names=(
            config.KEY_DATA_SOURCE,
            config.KEY_LOC_ID,
            config.KEY_YEAR,
            config.KEY_SPECIES_CODE,
            config.KEY_SUBGROUP_CODE,
        ))

        # Take the union of the indices of all groups
        self._index = reduce(pd.MultiIndex.union,
                             [group.xs(key, level=config.KEY_OBS_TYPE).index for key, group in groups],
                             eix,
                             ).drop_duplicates(keep='first')

        return self._index.sortlevel(level=0, sort_remaining=True)[0]

    def iter_index(self) -> Iterator[Tuple]:
        """
        Iterate over all base indices in the dataset.
        
        Yields:
            Tuples of (src, loc_id, year, species_code, subgroup_code).
        """
        for i in self._index:
            yield i

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all dataset entries.
        
        Yields:
            Dictionaries containing observation data for each entry.
        """
        for i in self.iter_index():
            yield self[i]

    @property
    def locations(self) -> List[Tuple[str, str]]:
        """
        Get all unique location identifiers in the dataset.

        Returns:
            List of tuples (data_source, loc_id) for all locations in the dataset.
        """
        locations = set(zip(self._index.get_level_values(config.KEY_DATA_SOURCE),
                            self._index.get_level_values(config.KEY_LOC_ID)))
        return list(locations)

    @property
    def locations_complete(self) -> List[Tuple[str, str]]:
        return list(self._locations)

    @property
    def species(self) -> List[Tuple[str, int, int]]:
        """
        Get all unique species identifiers in the dataset.

        Returns:
            List of tuples (data_source, species_code) for all species in the dataset.
        """
        species = set(zip(self._index.get_level_values(config.KEY_DATA_SOURCE),
                          self._index.get_level_values(config.KEY_SPECIES_CODE),
                          # self._index.get_level_values(config.KEY_SUBGROUP_CODE),
                          ))
        return list(species)

    @property
    def species_complete(self) -> list:
        return list(self._species)

    @property
    def species_subgroups(self):
        """
        Get all unique species identifiers in the dataset.

        Returns:
            List of tuples (data_source, species_code, subgroup_code) for all species in the dataset.
        """
        species = set(zip(self._index.get_level_values(config.KEY_DATA_SOURCE),
                          self._index.get_level_values(config.KEY_SPECIES_CODE),
                          self._index.get_level_values(config.KEY_SUBGROUP_CODE),
                          ))
        return list(species)

    @property
    def species_subgroups_complete(self):
        return list(self._species_subgroups)

    @property
    def years(self) -> List[int]:
        """
        Get all years present in the dataset, sorted.
        
        Returns:
            Sorted list of unique years.
        """
        return list(sorted(set(self._index.get_level_values(config.KEY_YEAR))))

    @property
    def num_observation_types(self) -> int:
        """
        Get the number of different observation types in the dataset.
        
        Returns:
            Number of unique observation types.
        """
        return self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).nunique()

    @property
    def observation_types(self) -> List[str]:
        """
        Get all observation types present in the dataset.
        
        Returns:
            List of unique observation type identifiers.
        """
        return list(self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).unique())

    @property
    def bounding_box(self) -> Dict[str, float]:
        """
        Calculate the spatial bounding box of all locations in the dataset.
        
        Returns:
            Dictionary with keys:
            - 'min_lat': Minimum latitude
            - 'max_lat': Maximum latitude
            - 'min_lon': Minimum longitude
            - 'max_lon': Maximum longitude
            
            Returns zeros for all values if the dataset is empty.
        """
        if len(self._index) == 0:
            return {
                'min_lat': 0.0,
                'max_lat': 0.0,
                'min_lon': 0.0,
                'max_lon': 0.0,
            }

        min_lat = np.inf
        max_lat = -np.inf
        min_lon = np.inf
        max_lon = -np.inf

        for i in self.iter_index():
            coords = self.get_location_coordinates(i, from_index=True)
            lon = coords[config.KEY_LON]
            lat = coords[config.KEY_LAT]

            if min_lat > lat:
                min_lat = lat
            if max_lat < lat:
                max_lat = lat
            if min_lon > lon:
                min_lon = lon
            if max_lon < lon:
                max_lon = lon

        return {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon,
        }

    def get_location_coordinates(self, i: Tuple, from_index: bool = False) -> Dict[str, float]:
        """
        Get the coordinates (latitude, longitude) for a location.
        
        Args:
            i: Location identifier. If from_index=True, expects a full index tuple
               (src, loc_id, year, species_code, subgroup_code).
               If from_index=False, expects (src, loc_id).
            from_index: Whether the input is a full index tuple or just location tuple.
        
        Returns:
            Dictionary with 'lat' and 'lon' keys.
        
        Raises:
            KeyError: If the location is not found in df_y_loc.
        """
        if from_index:
            src, loc_id, _, _, _ = i
        else:
            src, loc_id = i
        lat = self._df_y_loc.loc[src, loc_id][config.KEY_LAT]
        lon = self._df_y_loc.loc[src, loc_id][config.KEY_LON]
        return {
            'lat': lat,
            'lon': lon,
        }

    def get_location_name(self, i: Tuple, from_index: bool = False) -> str:
        """
        Get the name of a location.
        
        Args:
            i: Location identifier. If from_index=True, expects a full index tuple
               (src, loc_id, year, species_code, subgroup_code).
               If from_index=False, expects (src, loc_id).
            from_index: Whether the input is a full index tuple or just location tuple.
        
        Returns:
            Location name as a string.
        
        Raises:
            KeyError: If the location is not found in df_y_loc.
        """
        if from_index:
            src, loc_id, _, _, _ = i
        else:
            src, loc_id = i
        return self._df_y_loc.loc[src, loc_id][config.KEY_LOC_NAME]

    # def get_location_country(self, i: tuple, from_index: bool = False) -> tuple:
    #     if from_index:
    #         src, loc_id, _, _, _ = i
    #     else:
    #         src, loc_id = i
    #
    #     cc = self._df_y_loc.loc[src, loc_id][config.KEY_COUNTRY_CODE]
    #
    #     return cc, config.COUNTRY_CODE_NAME[cc]

    def select_locations(self, locations: Union[Tuple[str, str], List[Tuple[str, str]]]) -> 'BaseDataset':
        """
        Create a new BaseDataset with only the specified locations.
        
        Args:
            locations: The locations to select. Can be specified using either:
                - A two-tuple containing (data source key, loc_id)
                - A list of two-tuples (for selecting multiple locations)
        
        Returns:
            A new BaseDataset object containing only the specified locations.
        """
        return BaseDataset(
            select_location(self._df_y, locations),
            self._df_y_loc,
            species=self.species_complete,
            species_subgroups=self.species_subgroups_complete,
            locations=self.locations_complete,
        )

    def select_years(self, years: Union[int, List[int]]) -> 'BaseDataset':
        """
        Create a new BaseDataset with only the specified years.
        
        Args:
            years: The years to select. Can be specified using either:
                - A single year (as integer)
                - A list of years
        
        Returns:
            A new BaseDataset object containing only the specified years.
        """
        return BaseDataset(
            select_year(self._df_y, years),
            self._df_y_loc,
            species=self.species_complete,
            species_subgroups=self.species_subgroups_complete,
            locations=self.locations_complete,
        )

    def select_species(self, species: Union[Tuple, List[Tuple]]) -> 'BaseDataset':
        """
        Create a new BaseDataset with only the specified species.
        
        Args:
            species: The species to select. Can be specified using either:
                - A species tuple (can be either (src, species_id, subgroup_id) or (src, species_id))
                - A list of species tuples
        
        Returns:
            A new BaseDataset object containing only the specified species.
        """
        return BaseDataset(
            select_species(self._df_y, species),
            self._df_y_loc,
            species=self.species_complete,
            species_subgroups=self.species_subgroups_complete,
            locations=self.locations_complete,
        )

    def select_by_local_num_observations(self, num_observations: int, obs_key: str) -> 'BaseDataset':
        """
        Create a new BaseDataset containing only entries with at least the specified number
        of observations of a given type.
        
        Args:
            num_observations: Minimum number of observations required (must be >= 0).
            obs_key: Observation type key to filter on.
        
        Returns:
            A new BaseDataset object containing only entries meeting the requirement.
        
        Raises:
            AssertionError: If num_observations is negative.
        """
        assert num_observations >= 0
        if num_observations == 0:
            return BaseDataset(
                self._df_y,
                self._df_y_loc,
                species=self.species_complete,
                species_subgroups=self.species_subgroups_complete,
                locations=self.locations_complete,
            )

        # df_y is indexed by
        # (data src, loc_id, year, species_code, subgroup_code, obs_type)

        df_y = self._df_y.groupby(
            level=[0, 1, 3, 4]
        ).filter(
            lambda df: len(df.xs(obs_key, level=config.KEY_OBS_TYPE)) >= num_observations
        )

        return BaseDataset(
            df_y,
            self._df_y_loc,
        )

    def select_by_observation_requirement(self, obs_key: Union[str, List[str]]) -> 'BaseDataset':
        """
        Create a new BaseDataset containing only entries that have all specified observation types.
        
        Only entries that have ALL specified observation types will be included in the result.
        
        Args:
            obs_key: Single observation type key (str) or list of observation type keys.
                    Only entries that have ALL specified observation types will be included.
        
        Returns:
            A new BaseDataset object containing only entries with all required observation types.
        
        Example:
            >>> dataset = dataset.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])
            >>> # Returns dataset with only entries that have both BBCH_0 and BBCH_51 observations
        """
        if not isinstance(obs_key, list):
            obs_key = [obs_key]

        ixs_selection = []
        for ix in self.iter_index():

            if all([(ix + (key,)) in self._df_y.index for key in obs_key]):
                ixs_selection.extend([(ix + (key,)) for key in obs_key])

        if not ixs_selection:
            return BaseDataset(
                empty_df_like(self._df_y),
                self._df_y_loc,
                species=self.species_complete,
                species_subgroups=self.species_subgroups_complete,
                locations=self.locations_complete,
            )

        df_y = self._df_y.loc[ixs_selection]

        return BaseDataset(
            df_y,
            self._df_y_loc,
            species=self.species_complete,
            species_subgroups=self.species_subgroups_complete,
            locations=self.locations_complete,
        )

    def select_by_ixs(self, ixs: List[Tuple]) -> 'BaseDataset':
        """
        Create a new BaseDataset containing only the specified base indices.
        
        Args:
            ixs: List of base index tuples (src, loc_id, year, species_code, subgroup_code).
        
        Returns:
            A new BaseDataset object containing only the specified indices.
        
        Raises:
            TypeError: If ixs is not a list.
        """
        if not isinstance(ixs, list):
            raise TypeError(f"ixs must be a list, got {type(ixs)}")

        # df_y is indexed by
        # (data src, loc_id, year, species_code, subgroup_code, obs_type)

        if len(ixs) == 0:
            return BaseDataset(
                empty_df_like(self._df_y),
                self._df_y_loc,
                species=self.species_complete,
                species_subgroups=self.species_subgroups_complete,
                locations=self.locations_complete,
            )

        df_y = pd.concat([
            self._df_y.xs((src, loc_id, year, spc, sub),
                       level=[
                           config.KEY_DATA_SOURCE,
                           config.KEY_LOC_ID,
                           config.KEY_YEAR,
                           config.KEY_SPECIES_CODE,
                           config.KEY_SUBGROUP_CODE,
                       ],
                       drop_level=False,
                       )
            for src, loc_id, year, spc, sub in ixs
        ])

        return BaseDataset(
            df_y,
            self._df_y_loc,
            species=self.species_complete,
            species_subgroups=self.species_subgroups_complete,
            locations=self.locations_complete,
        )

    # def select_by_position_ixs(self, ixs: list) -> 'BaseDataset':
    #     assert isinstance(ixs, list)
    #     # ixs do not match index of self._df_y and should thus be converted
    #     ixs = list(self._index[ixs].values)
    #
    #     # df_y is indexed by
    #     # (data src, loc_id, year, species_code, subgroup_code, obs_type)
    #
    #     df_y = pd.concat([
    #         self._df_y.xs((src, loc_id, year, spc, sub),
    #                    level=[
    #                        config.KEY_DATA_SOURCE,
    #                        config.KEY_LOC_ID,
    #                        config.KEY_YEAR,
    #                        config.KEY_SPECIES_CODE,
    #                        config.KEY_SUBGROUP_CODE,
    #                    ],
    #                    drop_level=False,
    #                    )
    #         for src, loc_id, year, spc, sub in ixs
    #     ])
    #
    #     return BaseDataset(
    #         df_y,
    #         self._df_y_loc,
    #     )

    def aggregate_in_grid(self,
                          method: str = 'median',
                          grid_size: Optional[Tuple[float, float]] = None,
                          ) -> 'BaseDataset':
        """
        Divide the dataset into grid cells and aggregate observations within each cell.
        
        Observations are grouped by spatial grid cells, and one aggregated observation
        is selected per grid cell using the specified method.
        
        Args:
            method: Method of aggregation/selection. Options: 'median' (default), 'mean', 'first'.
                    Note: 'mean' and 'first' are currently deprecated/not implemented.
            grid_size: Tuple of (grid latitude cell size, grid longitude cell size) in degrees.
                      If None, uses config.MIN_GRID_SIZE.
        
        Returns:
            A new BaseDataset object with aggregated observations.
        
        Raises:
            DatasetException: If an unknown aggregation method is specified.
            NotImplementedError: If 'first' method is requested (not yet implemented).
        """

        # Select minimal grid size if none was provided
        grid_size = grid_size or config.MIN_GRID_SIZE

        res_lat, res_lon = grid_size

        # Create a copy of the observations dataframe
        # Add a latitude and longitude column
        df_selection = self._df_y.join(self._df_y_loc[[config.KEY_LAT, config.KEY_LON]])

        # Round location coordinates to corresponding grid cell
        df_selection[config.KEY_LAT] = df_selection[config.KEY_LAT].apply(lambda x: round_partial(x, res_lat))
        df_selection[config.KEY_LON] = df_selection[config.KEY_LON].apply(lambda x: round_partial(x, res_lon))

        # Subsequent steps depend on aggregation method
        match method:
            case 'mean':

                raise Exception('Deprecated: First mean method of aggregation should be fixed')

            case 'median':

                # df_selection[config.KEY_OBSERVATIONS] = df_selection[config.KEY_OBSERVATIONS].astype(np.int64)

                df_selection.reset_index(inplace=True)

                # Aggregate the phenology observations
                df = df_selection.groupby(
                    [config.KEY_LAT, config.KEY_LON, config.KEY_OBS_TYPE, config.KEY_YEAR],
                    as_index=False,
                ).agg(
                    {
                        config.KEY_LAT: 'first',
                        config.KEY_LON: 'first',
                        config.KEY_OBS_TYPE: 'first',
                        config.KEY_YEAR: 'first',
                        config.KEY_OBSERVATIONS: 'median',
                        config.KEY_DATA_SOURCE: 'first',
                        config.KEY_LOC_ID: 'first',
                        config.KEY_SPECIES_CODE: 'first',
                        config.KEY_SUBGROUP_CODE: 'first',
                    }
                )

                df.reset_index(inplace=True)

                # df[config.KEY_OBSERVATIONS] = pd.to_datetime(df[config.KEY_OBSERVATIONS]).apply(
                #     lambda x: np.datetime64(x.date()))

                df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
                df.drop([config.KEY_LAT, config.KEY_LON, 'index'], axis=1, inplace=True)

                return BaseDataset(
                    df_y=df,
                    df_y_loc=self._df_y_loc,
                    species=self.species_complete,
                    species_subgroups=self.species_subgroups_complete,
                    locations=self.locations_complete,
                )

            case 'first':
                raise NotImplementedError  # TODO
            case _:
                raise DatasetException(f'Unknown aggregation method "{method}" for phenological observations')

    def split_by_grid(self,
                      grid_size: Tuple[float, float],
                      split_size: float,
                      shuffle: bool = True,
                      random_state: Optional[int] = None,
                      ) -> Tuple['BaseDataset', 'BaseDataset', Dict[str, Any]]:
        """
        Split the dataset spatially by assigning locations to grid cells.
        
        Locations are assigned to spatial grid cells, and cells are randomly split
        into two groups. All locations within a cell are assigned to the same split.
        This ensures spatial separation between train/test sets.
        
        Args:
            grid_size: Size of the grid cells (in degrees) as (lat_size, lon_size).
            split_size: Proportion of cells assigned to the first dataset (0.0 to 1.0).
                       Remainder is assigned to the second dataset.
            shuffle: Whether to shuffle the cells before splitting.
            random_state: Random seed for shuffling the cells (for reproducibility).
        
        Returns:
            A three-tuple containing:
            - First dataset (BaseDataset)
            - Second dataset (BaseDataset)
            - Info dictionary with keys: 'cells_1', 'cells_2', 'cells_to_locations', 'cell_size'
        
        Raises:
            AssertionError: If split_size is not between 0 and 1, or if there's overlap between splits.
        """
        assert 0 <= split_size <= 1
        lat_size, lon_size = grid_size

        cell_to_locations = defaultdict(set)

        # Assign locations to cells by discretizing their coordinates
        for loc in self.locations:
            coords = self.get_location_coordinates(loc, from_index=False)
            lat = coords[config.KEY_LAT]
            lon = coords[config.KEY_LON]
            cell = (round_partial(lat, lat_size), round_partial(lon, lon_size))
            cell_to_locations[cell].add(loc)

        # Get all cells containing data
        cells = list(cell_to_locations.keys())
        # Sort cells to ensure determinism in split
        cells = sorted(cells)

        # Split the cells
        cells_1, cells_2 = train_test_split(cells,
                                            train_size=split_size,
                                            random_state=random_state,
                                            shuffle=shuffle,
                                            )

        # Merge locations corresponding to the cell split
        locs_1 = set.union(*[cell_to_locations[cell] for cell in cells_1])
        locs_2 = set.union(*[cell_to_locations[cell] for cell in cells_2])
        # Ensure there's no overlap -- as sanity check
        assert len(set.intersection(locs_1, locs_2)) == 0

        # Return two datasets with only the selected locations
        return (self.select_locations(list(locs_1)),
                self.select_locations(list(locs_2)),
                {
                    'cells_1': cells_1,
                    'cells_2': cells_2,
                    'cells_to_locations': cell_to_locations,
                    'cell_size': grid_size,
                },
                )

    def split_by_lon_border(self, lon_border: float) -> tuple:

        info = {}

        ixs_below = []
        ixs_above = []
        for ix in self.iter_index():
            coords = self.get_location_coordinates(ix, from_index=True)
            lon = coords[config.KEY_LON]
            if lon <= lon_border:
                ixs_below.append(ix)
            else:
                ixs_above.append(ix)

        return (self.select_by_ixs(ixs_below),
                self.select_by_ixs(ixs_above),
                info,
                )

    def split_by_lat_border(self, lat_border: float) -> tuple:

        info = {}

        ixs_below = []
        ixs_above = []
        for ix in self.iter_index():
            coords = self.get_location_coordinates(ix, from_index=True)
            lat = coords[config.KEY_LAT]
            if lat <= lat_border:
                ixs_below.append(ix)
            else:
                ixs_above.append(ix)

        return (self.select_by_ixs(ixs_below),
                self.select_by_ixs(ixs_above),
                info,
                )

    def observation_counts(self) -> Dict[str, int]:
        """
        Count the number of observations for each observation type.
        
        Returns:
            Dictionary mapping observation type keys to their counts.
        """
        counts = self._df_y.value_counts(subset=config.KEY_OBS_TYPE)
        return counts.to_dict()

    """
    ####################################################################################################################
     View
    ####################################################################################################################
    """

    def savefig_observation_hists(self, path: str, n_bins: int = 50) -> None:
        """
        Save histograms of observation day-of-year distributions for each observation type.
        
        Creates separate histograms for each observation type showing the distribution
        of day-of-year values.
        
        Args:
            path: Directory path where the histogram will be saved.
            n_bins: Number of bins for the histogram (default: 50).
        """

        fig, ax = plt.subplots(nrows=self.num_observation_types, sharex=True, sharey=True)
        if self.num_observation_types == 1:
            ax = (ax,)
        for i, obs_type in enumerate(self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).unique()):
            hist = self._df_y.xs(obs_type,
                                 level=config.KEY_OBS_TYPE,
                                 ).apply(lambda x: x.dt.dayofyear).hist(bins=n_bins, sharex=True, sharey=True, ax=ax[i])
            # ax[i].set_title(f'Observation type: {obs_type}')
            ax[i].set_title('')
            ax[i].set_xlim(0, 365)
            ax[i].set_xlabel(f'Day of year')
            # ax[i].set_ylabel(f'Count')
            ax[i].set_ylabel(f'# {obs_type}')

        fn = 'histograms.png'

        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, fn), bbox_inches='tight')
        plt.close()

    def savefig_observation_over_time(self, path: str) -> None:
        """
        Save scatter plots of observations over time for each observation type.
        
        Creates scatter plots showing how observation day-of-year values change
        over the years for each observation type.
        
        Args:
            path: File path where the plot will be saved (without extension).
        """

        fig, axs = plt.subplots(nrows=self.num_observation_types, sharex=True, sharey=False)

        x_obs = defaultdict(list)
        y_obs = defaultdict(list)

        for x in self.iter_items():

            for obs_type, obs in x[config.KEY_OBSERVATIONS].items():
                x_obs[obs_type].append(x[config.KEY_YEAR])
                y_obs[obs_type].append(obs.dayofyear)

        for ax, obs_type in zip(axs, self.observation_types):

            y_min = min(y_obs[obs_type])
            y_max = max(y_obs[obs_type])

            ax.scatter(x_obs[obs_type],
                       y_obs[obs_type],
                       s=1,
                       alpha=0.5,
                       )
            ax.set_title(f'Observation type: {obs_type}')
            ax.set_ylim(y_min, y_max)

        os.makedirs(path, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def savefig_observation_mean_over_time(self,
                                           path: str,
                                           obs_type: str,
                                           dpi: int = 500,
                                           ) -> None:
        """
        Save a plot showing mean observation day-of-year over time with linear trend.
        
        Creates a scatter plot of mean observation values per year and fits a linear
        trend line to visualize temporal changes.
        
        Args:
            path: Directory path where plots will be saved (as both SVG and PNG).
            obs_type: Observation type to plot.
            dpi: DPI for the PNG file (default: 500).
        """

        xs = []  # years
        ys = []  # mean observation
        counts = self.observation_counts()
        if obs_type in counts.keys() and counts[obs_type] > 0:
            df_y = self._df_y.xs(obs_type, level=config.KEY_OBS_TYPE)
            df_y = df_y.groupby(config.KEY_YEAR).mean()
            xys = df_y.to_dict()[config.KEY_OBSERVATIONS]

            for x, y in xys.items():
                xs.append(x)
                ys.append(y.dayofyear)

        # Fit linear trend
        reg = linear_model.LinearRegression()
        reg.fit(np.array(xs).reshape(-1, 1), ys)
        xs_lin = list(range(min(xs), max(xs) + 1))
        ys_lin = reg.predict(np.array(xs_lin).reshape(-1, 1))

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        ax.scatter(xs, ys, color='tomato')
        a = reg.coef_[0]
        b = reg.intercept_
        ax.plot(xs_lin, ys_lin, '--', color='black', label=f'$f(x)=${a:.2f}$x${"+" if b >= 0 else ""}{b:.2f}')

        ymin = 0
        ymax = 365
        if len(ys) > 0:
            margin = 1.
            dy = round((max(ys) - min(ys)) * margin)
            ymin = max(ymin, min(ys) - dy)
            ymax = min(ymax, max(ys) + dy)

        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Day of year ({obs_type})')

        ax.legend()

        # Save figure
        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)

        plt.savefig(os.path.join(path_svg, f'observation_mean_over_time.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'observation_mean_over_time.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    def savefig_observation_map(self,
                                path: str,
                                dpi: int = 500,
                                ) -> None:
        """
        Save a map showing all observation locations in the dataset.
        
        Creates a map with administrative boundaries and scatter plots of all
        observation locations. Map is saved as both PNG and SVG files.
        
        Args:
            path: Directory path where the map will be saved.
            dpi: DPI for the PNG file (default: 500).
        """

        # Get all observation types present
        obs_types = self.observation_types

        # Load administrative boundaries for plotting
        gdf_admin = load_admin_boundaries()
        # Get dataset spatial bounding box to filter administrative region
        bb = self.bounding_box

        minx = bb['min_lon']
        miny = bb['min_lat']
        maxx = bb['max_lon']
        maxy = bb['max_lat']

        gdf_admin = gdf_admin.cx[minx:maxx, miny:maxy]

        # Group all observations of the specific year by observation type
        obs_per_type = {
            ot: list() for ot in obs_types
        }

        for ot in obs_types:  # TODO -- option to color by obs type -- or color by species subgroup!
            # Get all observations for the specific year and observation type
            # Drop observation type from index
            # df index: (data src, loc_id, year, species_code, subgroup_code)
            df_year_ot = self._df_y.xs(ot, level=config.KEY_OBS_TYPE, drop_level=True)

            # Iterate over samples in the dataframe. Store their coordinates
            for i in df_year_ot.index:
                coords = self.get_location_coordinates(i, from_index=True)
                coords = (coords[config.KEY_LON], coords[config.KEY_LAT])
                obs_per_type[ot].append(coords)

        # transform coordinate lists in geopandas dataframes
        obs_per_type = {
            ot: gpd.GeoDataFrame(
                obs,
                geometry=gpd.points_from_xy(
                    [x for x, y in obs],
                    [y for x, y in obs],
                ),
                crs=gdf_admin.crs,
            ) for ot, obs in obs_per_type.items()
        }

        # Plot observations per observation type
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # First plot country borders
        gdf_admin.plot(ax=ax, color='lightgrey', edgecolor='black')

        for ot, gdf_obs in obs_per_type.items():
            gdf_obs.plot(ax=ax,
                         label=ot,
                         marker='o',
                         markersize=0.6,
                         alpha=0.5,
                         color='tomato',
                         )

        # Set bounds to map
        margin = 0.1  # Keep a small margin bordering the plot as a percentage of total distance
        x_margin = (maxx - minx) * margin
        y_margin = (maxy - miny) * margin
        ax.set_xlim(minx - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + y_margin)

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')

        # Save figure

        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)
        plt.savefig(os.path.join(path_svg, f'observations_map.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'observations_map.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    def savefigs_observation_maps_over_time(self,
                                            path: str,
                                            verbose: bool = True,
                                            dpi: int = 500,
                                            ) -> None:
        """
        Generate maps showing observation locations for each year in the dataset.
        
        Creates separate maps for each year, showing observation locations as scatter plots.
        Points are colored by observation type. Plots are saved as both SVG and PNG files.
        
        Args:
            path: Directory path where the maps will be saved.
            verbose: Whether to display a progress bar (default: True).
            dpi: DPI for the PNG files (default: 500).
        """

        # Figures will be created per year
        iter_years = self.years
        if verbose:
            iter_years = tqdm(iter_years, total=len(iter_years), desc='Creating observations maps...')

        # Get all observation types present
        obs_types = self.observation_types

        # Load administrative boundaries for plotting
        gdf_admin = load_admin_boundaries()
        # Get dataset spatial bounding box to filter administrative region
        bb = self.bounding_box

        minx = bb['min_lon']
        miny = bb['min_lat']
        maxx = bb['max_lon']
        maxy = bb['max_lat']

        gdf_admin = gdf_admin.cx[minx:maxx, miny:maxy]

        for year in iter_years:

            # Get all observations for the specific year
            # Keep year in index
            # df index: (data src, loc_id, year, species_code, subgroup_code, obs_type)
            df_year = self._df_y.xs(year, level=config.KEY_YEAR, drop_level=False)

            # Group all observations of the specific year by observation type
            obs_per_type = {
                ot: list() for ot in obs_types
            }

            for ot in obs_types:
                # Get all observations for the specific year and observation type
                # Drop observation type from index
                # df index: (data src, loc_id, year, species_code, subgroup_code)

                if ot not in df_year.index.get_level_values(config.KEY_OBS_TYPE):
                    continue

                df_year_ot = df_year.xs(ot, level=config.KEY_OBS_TYPE, drop_level=True)

                # Iterate over samples in the dataframe. Store their coordinates
                for i in df_year_ot.index:
                    coords = self.get_location_coordinates(i, from_index=True)
                    coords = (coords[config.KEY_LON], coords[config.KEY_LAT])
                    obs_per_type[ot].append(coords)

            # transform coordinate lists in geopandas dataframes
            obs_per_type = {
                ot: gpd.GeoDataFrame(
                    obs,
                    geometry=gpd.points_from_xy(
                        [x for x, y in obs],
                        [y for x, y in obs],
                    ),
                    crs=gdf_admin.crs,
                ) for ot, obs in obs_per_type.items()
            }

            # Plot observations per observation type
            fig, ax = plt.subplots(1, 1)
            # First plot country borders
            gdf_admin.plot(ax=ax, color='lightgrey', edgecolor='black')

            for ot, gdf_obs in obs_per_type.items():
                if len(gdf_obs) == 0:
                    continue
                gdf_obs.plot(ax=ax,
                             label=ot,
                             marker='o',
                             markersize=1,
                             alpha=0.1,
                             )

            # Set bounds to map
            margin = 0.1  # Keep a small margin bordering the plot as a percentage of total distance
            x_margin = (maxx - minx) * margin
            y_margin = (maxy - miny) * margin
            ax.set_xlim(minx - x_margin, maxx + x_margin)
            ax.set_ylim(miny - y_margin, maxy + y_margin)

            # Save figure

            path_svg = os.path.join(path, 'svg')
            path_png = os.path.join(path, 'png')

            os.makedirs(path_svg, exist_ok=True)
            os.makedirs(path_png, exist_ok=True)
            plt.savefig(os.path.join(path_svg, f'observations_map_{year}.svg'),
                        bbox_inches='tight',
                        )
            plt.savefig(os.path.join(path_png, f'observations_map_{year}.png'),
                        bbox_inches='tight',
                        dpi=dpi,
                        )
            plt.close()

        # TODO -- generate video

    def savefig_species_subgroup_occurrence_temporal(self, path: str, dpi: int = 500) -> None:
        """
        Save a timeline plot showing when each species-subgroup combination occurs in the dataset.
        
        Creates a horizontal bar chart (broken_barh) showing the years in which each
        species-subgroup combination has observations.
        
        Args:
            path: Directory path where the plot will be saved (as both SVG and PNG).
            dpi: DPI for the PNG file (default: 500).
        """

        sss_year_occurrence = defaultdict(set)

        # df index: (data src, loc_id, year, species_code, subgroup_code)
        for src, _, year, species, subgroup in self.iter_index():
            sss_year_occurrence[src, species, subgroup].add(year)

        fig, ax = plt.subplots(1, 1,
                               # figsize=(10, 3 + len(sss_year_occurrence.keys())),
                               )

        bar_height = 0.4
        y = -bar_height / 2
        labels = []
        bars = []
        for (src, species, subgroup), years in sss_year_occurrence.items():

            # broken_barh(xranges, (ymin, height))
            # xranges is a sequence of (start, duration) tuples
            bar = ax.broken_barh([(year, 1) for year in years], (y, bar_height),
                                 color='grey',
                                 )

            bars.append(bar)
            y += 1
            labels.append((species, subgroup))

        ax.set_yticks(range(len(labels)),
                      labels=labels)
        ax.set_xlabel('Year')
        ax.set_ylabel('(Species ID, subgroup ID) occurrence')

        # # Access each bar's segments and adjust their heights if necessary
        # for bar in bars:
        #     for patch in bar.get_children():
        #         y, h = patch.get_y(), patch.get_height()
        #         if h > bar_height:
        #             # Cap the height at max_height and reset y to 0 (assuming it starts from bottom)
        #             patch.set_height(bar_height)
        #             patch.set_y(0)
        #         else:
        #             # Keep original values
        #             pass

        # Save figure
        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)
        plt.savefig(os.path.join(path_svg, f'species_subgroup_occurrence.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'species_subgroup_occurrence.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    """
    ####################################################################################################################
     Pre-defined datasets
    ####################################################################################################################
    """

    @staticmethod
    def load(key: str) -> 'BaseDataset':
        """
        Load a pre-configured dataset based on its key/name.
        
        This is the preferred method for creating BaseDataset instances, as it handles
        all the necessary preprocessing, filtering, and aggregation steps.
        
        Args:
            key: The key/name of the dataset to load.
        
        Returns:
            A BaseDataset instance configured according to the specified key.
        
        Raises:
            DatasetException: If the dataset key is not recognized.
        """

        """
        CPF Config
        """
        # cpf_year_min = 1980
        cpf_year_min = 1986
        cpf_year_max = 2015
        cpf_year_range = list(range(cpf_year_min, cpf_year_max + 1))
        cpf_remove_outliers = True
        # cpf_remove_outliers = False
        cpf_do_agg = True
        # cpf_do_agg = False
        # cpf_agg_method = 'mean'
        cpf_agg_method = 'median'

        """
        Benchmark config
        """
        # Set year range
        bm_year_min = 1986  # Start year  -- lower sometimes gives NaN features
        bm_year_max = datetime.now().year - 1  # Set previous year as start year
        bm_years = list(range(bm_year_min, bm_year_max + 1))
        bm_do_agg = True
        bm_agg_method = 'median'
        bm_remove_outliers = True
        bm_assert_target = True

        """
        Test Earth Embedding config
        """
        tbe_year_min = 2017
        tbe_year_max = datetime.now().year - 1  # Set previous year as start year
        tbe_years = list(range(tbe_year_min, tbe_year_max + 1))
        tbe_do_agg = False
        tbe_agg_method = 'median'
        tbe_remove_outliers = True
        tbe_assert_target = True

        match key:

            case 'test_dataset':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 333, 300),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=cpf_remove_outliers,
                    filter_on_observation_types=['BBCH_0', 'BBCH_51'],
                    filter_on_years=cpf_year_range,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                # print(df_y)
                # print(df_y_loc)

                df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])

                if cpf_do_agg:
                    base = base.aggregate_in_grid(method=cpf_agg_method)

                return base

            # PEP725 winter wheat
            case 'CPF_PEP725_winter_wheat':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 333, 300),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=cpf_remove_outliers,
                    filter_on_observation_types=['BBCH_0', 'BBCH_51'],
                    filter_on_years=cpf_year_range,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # base = base.select_locations_by_country_codes('DE')

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])

                if cpf_do_agg:
                    base = base.aggregate_in_grid(method=cpf_agg_method)

                return base

            # PEP725 winter barley
            case 'CPF_PEP725_winter_barley':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 330, 300),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=cpf_remove_outliers,
                    # remove_outliers=False,
                    filter_on_observation_types=['BBCH_0', 'BBCH_51'],
                    filter_on_years=cpf_year_range,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])

                if cpf_do_agg:
                    base = base.aggregate_in_grid(method=cpf_agg_method)

                return base

            # PEP725 winter rye
            case 'CPF_PEP725_winter_rye':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 332, 300),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=cpf_remove_outliers,
                    # remove_outliers=False,
                    filter_on_observation_types=['BBCH_0', 'BBCH_61'],
                    filter_on_years=cpf_year_range,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_61'])

                if cpf_do_agg:
                    base = base.aggregate_in_grid(method=cpf_agg_method)

                return base

            case 'GMU_Cherry_Japan':

                dfs = get_gmu_cherry_dataset_japan(
                    # remove_outliers=bm_remove_outliers,
                    remove_outliers=False,  # Set to false since multiple species occur in this dataset!
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_0')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_Switzerland':

                dfs = get_gmu_cherry_dataset_switzerland(
                    remove_outliers=bm_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_1')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_South_Korea':

                dfs = get_gmu_cherry_dataset_south_korea(
                    remove_outliers=bm_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_2')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_Japan_Y':
                from phenology.data.gmu_cherry.regions_data import LOCATIONS_JAPAN_YEDOENSIS

                dfs = get_gmu_cherry_dataset_japan(
                    # remove_outliers=bm_remove_outliers,
                    remove_outliers=False,  # Set to false since multiple species occur in this dataset!
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                locations_y = [(config.KEY_GMU_CHERRY, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_YEDOENSIS.keys()]

                base = base.select_locations(locations_y)

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_0')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_Japan_YS':
                from phenology.data.gmu_cherry.regions_data import LOCATIONS_JAPAN_YEDOENSIS, LOCATIONS_JAPAN_SARGENTII
                from phenology.config import KEY_GMU_CHERRY
                dfs = get_gmu_cherry_dataset_japan(
                    # remove_outliers=bm_remove_outliers,
                    remove_outliers=False,  # Set to false since multiple species occur in this dataset!
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                locations_y = [(KEY_GMU_CHERRY, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_YEDOENSIS.keys()]
                locations_s = [(KEY_GMU_CHERRY, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_SARGENTII.keys()]

                base = base.select_locations(locations_y + locations_s)

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_0')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Apple':  # Malus x Domestica

                species_subgroups = [
                    (config.KEY_PEP725, 220, 100),  # Early cultivar
                    (config.KEY_PEP725, 220, 130),  # Late cultivar
                    (config.KEY_PEP725, 220, 115),  # Middle cultivar
                    (config.KEY_PEP725, 220, 433),  # Cox Orange Renette
                    (config.KEY_PEP725, 220, 508),  # Elstar
                    (config.KEY_PEP725, 220, 437),  # Golden Delicious
                    (config.KEY_PEP725, 220, 430),  # Goldparm
                    (config.KEY_PEP725, 220, 438),  # Gravensteiner
                    (config.KEY_PEP725, 220, 509),  # Idared
                    (config.KEY_PEP725, 220, 500),  # James Grieve
                    (config.KEY_PEP725, 220, 501),  # Jonagold
                    (config.KEY_PEP725, 220, 510),  # Jonathan
                    (config.KEY_PEP725, 220, 503),  # Roter Boskoop
                    (config.KEY_PEP725, 220, 506),  # Wei
                    (config.KEY_PEP725, 220, 615),  # Granny Smith
                    (config.KEY_PEP725, 220, 617),  # Bobovec
                ]
                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Pear':  # Pyrus Communis

                species_subgroups = [
                    (config.KEY_PEP725, 227, 100),  # Early cultivar
                    (config.KEY_PEP725, 227, 130),  # Late cultivar
                    (config.KEY_PEP725, 227, 590),  # Williams
                    (config.KEY_PEP725, 227, 586),  # Bunte Julibirne
                    (config.KEY_PEP725, 227, 585),  # Jakob
                    (config.KEY_PEP725, 227, 587),  # Junsko Zlato
                    (config.KEY_PEP725, 227, 589),  # Karamanka
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Peach':  # Prunus Persica

                species_subgroups = [
                    (config.KEY_PEP725, 202, 0),  # No group
                    (config.KEY_PEP725, 202, 579),  # Alberta
                    (config.KEY_PEP725, 202, 580),  # Dixired
                    (config.KEY_PEP725, 202, 581),  # Hale
                    (config.KEY_PEP725, 202, 578),  # Red Haven
                    (config.KEY_PEP725, 202, 582),  # Springtime
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Almond':  # Prunus Amygdalis

                species_subgroups = [
                    (config.KEY_PEP725, 782, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Hazel':  # Corylus Avellana

                species_subgroups = [
                    (config.KEY_PEP725, 107, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_86',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Cherry':  # Prunus Avium

                species_subgroups = [
                    (config.KEY_PEP725, 222, 0),    # No group
                    (config.KEY_PEP725, 222, 100),  # Early cultivar
                    (config.KEY_PEP725, 222, 130),  # Late cultivar
                    (config.KEY_PEP725, 222, 494),  # Regina
                    (config.KEY_PEP725, 222, 495),  # Schwarze Knorpelkirsch
                    (config.KEY_PEP725, 222, 618),  # Majska rana
                    (config.KEY_PEP725, 222, 602),  # Germersdorfer
                    (config.KEY_PEP725, 222, 603),  # Hedelfinger
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Apricot':  # Prunus Armeniaca

                species_subgroups = [
                    (config.KEY_PEP725, 205, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_87', ],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Plum':  # Prunus Domestica

                species_subgroups = [
                    (config.KEY_PEP725, 225, 0),    # No group
                    (config.KEY_PEP725, 225, 100),  # Early cultivar
                    (config.KEY_PEP725, 225, 130),  # Late cultivar
                    (config.KEY_PEP725, 225, 621),  # Besztercei
                    (config.KEY_PEP725, 225, 595),  # Bosankska
                    (config.KEY_PEP725, 225, 596),  # Dzanarika
                    (config.KEY_PEP725, 225, 597),  # Pozegaca
                    (config.KEY_PEP725, 225, 612),  # Renkloda
                    (config.KEY_PEP725, 225, 614),  # Stanlay
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Blackthorn':  # Prunus Spinosa

                species_subgroups = [
                    (config.KEY_PEP725, 123, 0),
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Oak':  # Prunus Persica

                species_subgroups = [
                    (config.KEY_PEP725, 111, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=[
                        'BBCH_11',
                        'BBCH_94',
                        'BBCH_86',
                        'BBCH_95',
                    ],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            # PEP725 winter wheat
            case 'CFM_zea_mays':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 440, 0),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=True,
                    filter_on_observation_types=['BBCH_0', 'BBCH_51'],
                    # filter_on_years=cpf_year_range,
                    filter_on_years=list(range(1980, 2025)),
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])

                # base = base.aggregate_in_grid(method='median')

                return base

            case 'PEP725_fruit_trees':
                keys = [
                    'PEP725_Apple',
                    'PEP725_Pear',
                    'PEP725_Peach',
                    'PEP725_Almond',
                    'PEP725_Hazel',
                    'PEP725_Cherry',
                    'PEP725_Apricot',
                    'PEP725_Plum',
                    'PEP725_Blackthorn',
                ]

                return reduce(
                    merge_datasets,
                    [BaseDataset.load(key) for key in keys],
                )

            case 'PEP725_fruit_trees_2':  # Datasets represented in germany
                keys = [
                    'PEP725_Apple',
                    'PEP725_Pear',
                    'PEP725_Peach',
                    'PEP725_Almond',
                    'PEP725_Hazel',
                    'PEP725_Cherry',
                    # 'PEP725_Apricot',
                    'PEP725_Plum',
                    'PEP725_Blackthorn',
                ]

                return reduce(
                    merge_datasets,
                    [BaseDataset.load(key) for key in keys],
                )

            case 'PEP725_fruit_trees_3':  # Datasets with cultivar information
                keys = [
                    'PEP725_Apple',
                    'PEP725_Pear',
                    # 'PEP725_Peach',
                    'PEP725_Plum',
                    'PEP725_Cherry',
                ]

                return reduce(
                    merge_datasets,
                    [BaseDataset.load(key) for key in keys],
                )

            case 'PEP725_fruit_trees_4':  # Datasets wo hazel
                keys = [
                    'PEP725_Apple',
                    'PEP725_Pear',
                    'PEP725_Peach',
                    'PEP725_Almond',
                    # 'PEP725_Hazel',
                    'PEP725_Cherry',
                    'PEP725_Apricot',
                    'PEP725_Plum',
                    'PEP725_Blackthorn',
                ]

                return reduce(
                    merge_datasets,
                    [BaseDataset.load(key) for key in keys],
                )

            case 'PEP725_fruit_trees_5':  # Datasets wo hazel
                keys = [
                    'PEP725_Apple',
                    'PEP725_Pear',
                    'PEP725_Peach',
                    # 'PEP725_Almond',
                    # 'PEP725_Hazel',
                    'PEP725_Cherry',
                    # 'PEP725_Apricot',
                    'PEP725_Plum',
                    'PEP725_Blackthorn',
                ]

                return reduce(
                    merge_datasets,
                    [BaseDataset.load(key) for key in keys],
                )

            case 'all_fruit_trees':
                keys = {
                    'GMU_Cherry_Japan': 'gmu_0',
                    'GMU_Cherry_Switzerland': 'gmu_1',
                    'GMU_Cherry_South_Korea': 'gmu_2',
                    'PEP725_fruit_trees': 'BBCH_60',
                }

                # Key to denote mixed observation types
                mixed_key = 'mixed'
                # Load all datasets
                datasets = {key: BaseDataset.load(key) for key in keys.keys()}
                # Modify the datasets to use the 'mixed' observation type key
                for key, dataset in datasets.items():
                    obs_key = keys[key]

                    obs_type_level_key = config.KEY_OBS_TYPE

                    dataset._df_y.reset_index(inplace=True)
                    dataset._df_y[obs_type_level_key] = dataset._df_y[obs_type_level_key].replace({obs_key: mixed_key})
                    dataset._df_y.set_index([
                        config.KEY_DATA_SOURCE,
                        config.KEY_LOC_ID,
                        config.KEY_YEAR,
                        config.KEY_SPECIES_CODE,
                        config.KEY_SUBGROUP_CODE,
                        config.KEY_OBS_TYPE,
                    ], inplace=True
                    )

                # Merge all datasets
                return reduce(
                    merge_datasets,
                    list(datasets.values()),
                )

            case 'TBE_GMU_Cherry_Japan':

                dfs = get_gmu_cherry_dataset_japan(
                    # remove_outliers=bm_remove_outliers,
                    remove_outliers=False,  # Set to false since multiple species occur in this dataset!
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=tbe_years,
                )

                # Only keep data samples where the required observations are present
                if tbe_assert_target:
                    base = base.select_by_observation_requirement('gmu_0')
                # Aggregate observations in a grid
                if tbe_do_agg:
                    base = base.aggregate_in_grid(method=tbe_agg_method)

                return base

            case 'TBE_GMU_Cherry_Switzerland':

                dfs = get_gmu_cherry_dataset_switzerland(
                    # remove_outliers=tbe_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=tbe_years,
                )

                # Only keep data samples where the required observations are present
                if tbe_assert_target:
                    base = base.select_by_observation_requirement('gmu_1')
                # Aggregate observations in a grid
                if tbe_do_agg:
                    base = base.aggregate_in_grid(method=tbe_agg_method)

                return base

            case 'TBE_GMU_Cherry_South_Korea':

                dfs = get_gmu_cherry_dataset_south_korea(
                    # remove_outliers=tbe_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=tbe_years,
                )

                # Only keep data samples where the required observations are present
                if tbe_assert_target:
                    base = base.select_by_observation_requirement('gmu_2')
                # Aggregate observations in a grid
                if tbe_do_agg:
                    base = base.aggregate_in_grid(method=tbe_agg_method)

                return base

            case _:
                raise DatasetException(f'Undefined dataset key "{key}"')

    @staticmethod
    def _modify_year(df: pd.DataFrame, obs_type: str, offset: int) -> pd.DataFrame:
        """
        Modify the year for a specific observation type by adding an offset.
        
        This is a helper method used in dataset loading to adjust years for certain
        observation types (e.g., when observations span multiple calendar years).
        
        Args:
            df: DataFrame with observations.
            obs_type: Observation type to modify.
            offset: Year offset to add (can be negative).
        
        Returns:
            DataFrame with modified years for the specified observation type.
        """
        groups = df.groupby(level=config.KEY_OBS_TYPE)
        dfs = [group if key != obs_type else BaseDataset._modify_year_helper(group, offset) for key, group in groups]
        return pd.concat(dfs)

    @staticmethod
    def _modify_year_helper(df: pd.DataFrame, offset: int) -> pd.DataFrame:
        """
        Helper method to add an offset to year values in a DataFrame.
        
        Args:
            df: DataFrame to modify.
            offset: Year offset to add.
        
        Returns:
            DataFrame with modified year values.
        """
        df = df.copy()  # Avoid modifying original
        df.reset_index(inplace=True)
        df[config.KEY_YEAR] = df[config.KEY_YEAR] + offset
        df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
        return df

    @staticmethod
    def get_dataset_target_key(key: str) -> str:
       """
       Get the dataset target key based on the dataset name key
       :param key: dataset name key
       :return: dataset target key
       """
       match key:

           case 'test_dataset':
               return 'BBCH_51'
           case 'CPF_PEP725_winter_wheat':
               return 'BBCH_51'
           case 'CPF_PEP725_winter_barley':
               return 'BBCH_51'
           case 'CPF_PEP725_winter_rye':
               return 'BBCH_61'
           case 'GMU_Cherry_Japan':
               return 'gmu_0'
           case 'GMU_Cherry_Switzerland':
               return 'gmu_1'
           case 'GMU_Cherry_South_Korea':
               return 'gmu_2'
           case 'GMU_Cherry_Japan_Y':
               return 'gmu_0'
           case 'GMU_Cherry_Japan_YS':
               return 'gmu_0'
           case 'PEP725_Apple':
               return 'BBCH_60'
           case 'PEP725_Pear':
               return 'BBCH_60'
           case 'PEP725_Peach':
               return 'BBCH_60'
           case 'PEP725_Almond':
               return 'BBCH_60'
           case 'PEP725_Hazel':
               return 'BBCH_60'
           case 'PEP725_Cherry':
               return 'BBCH_60'
           case 'PEP725_Apricot':
               return 'BBCH_60'
           case 'PEP725_Plum':
               return 'BBCH_60'
           case 'PEP725_Blackthorn':
               return 'BBCH_60'
           case 'PEP725_Oak':
               return 'BBCH_60'
           case 'CFM_zea_mays':
               return 'BBCH_51'
           case 'PEP725_fruit_trees':
               return 'BBCH_60'
           case 'PEP725_fruit_trees_2':
               return 'BBCH_60'
           case 'PEP725_fruit_trees_3':
               return 'BBCH_60'
           case 'PEP725_fruit_trees_4':
               return 'BBCH_60'
           case 'all_fruit_trees':
               return 'mixed'

           case 'TBE_GMU_Cherry_Japan':
               return 'gmu_0'
           case 'TBE_GMU_Cherry_Switzerland':
               return 'gmu_1'
           case 'TBE_GMU_Cherry_South_Korea':
               return 'gmu_2'

           case _:
               raise DatasetException(f'Undefined dataset key "{key}"')


def merge_datasets(d1: BaseDataset, d2: BaseDataset) -> BaseDataset:
    """
    Merge two BaseDataset instances into a single dataset.
    
    Combines observations and locations from both datasets, removing duplicates.
    Duplicates are determined by the full index (including observation type).
    
    Args:
        d1: First dataset to merge.
        d2: Second dataset to merge.
    
    Returns:
        A new BaseDataset containing merged data from both input datasets.
    """
    df_y = pd.concat([d1._df_y, d2._df_y])
    df_y_loc = pd.concat([d1._df_y_loc, d2._df_y_loc])

    df_y = df_y[~df_y.index.duplicated(keep='first')]
    df_y_loc = df_y_loc[~df_y_loc.index.duplicated(keep='first')]

    s1 = d1.species_complete
    s2 = d2.species_complete

    sset = set.union(
        set(s1),
        set(s2),
    )

    ss1 = d1.species_subgroups_complete
    ss2 = d2.species_subgroups_complete

    ssset = set.union(
        set(ss1),
        set(ss2),
    )

    l1 = d1.locations_complete
    l2 = d2.locations_complete

    lset = set.union(
        set(l1),
        set(l2),
    )

    return BaseDataset(df_y,
                       df_y_loc,
                       species=list(sset),
                       species_subgroups=list(ssset),
                       locations=list(lset),
                       )


if __name__ == '__main__':
    from tqdm import tqdm

    # _dataset_name = 'CPF_PEP725_winter_wheat'
    # _dataset_name = 'GMU_Cherry_Japan'
    # _dataset_name = 'GMU_Cherry_Switzerland'
    # _dataset_name = 'GMU_Cherry_South_Korea'
    # _dataset_name = 'PEP725_Oak'
    # _dataset_name = 'PEP725_Apple'
    # _dataset_name = 'PEP725_Pear'
    # _dataset_name = 'PEP725_Peach'
    # _dataset_name = 'PEP725_Almond'
    # _dataset_name = 'PEP725_Hazel'
    # _dataset_name = 'PEP725_Cherry'
    # _dataset_name = 'PEP725_Apricot'
    # _dataset_name = 'PEP725_Blackthorn'
    # _dataset_name = 'PEP725_Plum'
    # _dataset_name = 'CFM_zea_mays'
    # _dataset_name = 'all_fruit_trees'
    _dataset_name = 'PEP725_fruit_trees_3'

    # _dataset_name = 'TBE_GMU_Cherry_Japan'
    # _dataset_name = 'TBE_GMU_Cherry_Switzerland'
    # _dataset_name = 'TBE_GMU_Cherry_South_Korea'

    _dataset = BaseDataset.load(_dataset_name)

    # print(_dataset._index[0])

    # _dataset_selection = _dataset.select_by_observation_requirement([0, 51])

    # _dataset_selection = _dataset.aggregate_in_grid()

    # _dataset_selection.show_observation_hists()
    # _dataset_selection.show_observation_over_time()

    # print(_dataset_selection.observation_counts())
    #
    # print(len(_dataset))
    # print(len(_dataset_selection))

    # _dataset.select_by_local_num_observations(2, 51)

    # for _x in _dataset_selection.iter_items():
    # for _x in tqdm(_dataset_selection.iter_items(), total=len(_dataset_selection)):
    #     # print(_x)
    #     pass
    # #     input()
    #
    # for _x in tqdm(_dataset_selection.iter_items(), total=len(_dataset_selection)):
    #     # print(_x)
    #     pass

    # _dataset.savefig_observation_map('temp_map')
    # _dataset.savefigs_observation_maps_over_time('temp_map_over_time')
    # _dataset.savefig_observation_hists('temp_hist')

    # _dataset.savefigs_complete(f'temp_dataset_figures/{_dataset_name}')
    
    # _dataset.savefig_observation_mean_over_time('temp', 'BBCH_60')

    # _dataset.savefig_species_subgroup_occurrence_temporal('temp')

    dpi = 500

    # Use same dir as script to store figures
    path_base = os.path.abspath(os.path.dirname(__file__))


    path = os.path.join(
        path_base,
        'figures',
        'datasets',
        _dataset_name,
    )

    # obs_type = 'BBCH_60'
    obs_type = BaseDataset.get_dataset_target_key(_dataset_name)

    _dataset.savefig_observation_mean_over_time(os.path.join(path, 'observation_trend'),
                                                obs_type=obs_type,
                                                dpi=dpi,
                                                )

    _dataset.savefig_observation_map(
        path=os.path.join(path, 'observation_map'),
    )

    # self.savefigs_observation_maps_over_time(
    #     path=os.path.join(path, 'observation_maps_over_time'),
    #     verbose=False,
    # )

    _dataset.savefig_observation_hists(
        path=os.path.join(path, 'observation_hists'),
    )

    _dataset.savefig_species_subgroup_occurrence_temporal(
        path=os.path.join(path, 'species_subgroup_occurrence_temporal'),
    )


    print(_dataset.observation_counts())

