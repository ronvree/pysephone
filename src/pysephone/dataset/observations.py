from collections import defaultdict
from functools import reduce
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_LAT,
    KEY_LOC_ID,
    KEY_LOC_NAME,
    KEY_LON,
    KEY_OBS_TYPE,
    KEY_OBSERVATIONS,
    KEY_SPECIES_ID,
    KEY_SUBGROUP_ID,
    KEY_YEAR,
    KEYS_INDEX,
)
from pysephone.dataset.util.func import (
    DatasetException,
    empty_df_like,
    select_location,
    select_species,
    select_year,
)
from pysephone.utils.func import round_partial


class Observations:
    """
    Phenological observation data.

    Holds observations indexed by (src, loc_id, year, species_id, subgroup_id, obs_type)
    and corresponding location metadata (lat, lon, etc.).  All filtering and splitting
    methods return new Observations instances — this object is effectively immutable.
    """

    def __init__(
        self,
        df_y: pd.DataFrame,
        df_y_loc: pd.DataFrame,
        locations_complete: Optional[List[Tuple[str, str]]] = None,
        species_complete: Optional[List[Tuple[str, int]]] = None,
        species_subgroups_complete: Optional[List[Tuple[str, int, int]]] = None,
        species_names: Optional[Dict[Tuple[str, int], str]] = None,
    ) -> None:
        """
        Args:
            df_y:     Observations DataFrame indexed by
                      (src, loc_id, year, species_id, subgroup_id, obs_type)
                      with a KEY_OBSERVATIONS column.
            df_y_loc: Locations DataFrame indexed by (src, loc_id)
                      with at least KEY_LAT and KEY_LON columns.
            locations_complete:         Full set of locations from the original
                      (pre-split) dataset.  Defaults to the current view.
            species_complete:           Full set of species from the original dataset.
            species_subgroups_complete: Full set of (species, subgroup) pairs from
                      the original dataset.
            species_names:              Optional ``{(src, species_id): scientific_name}``
                      mapping.  Used by :class:`~pysephone.dataset.util.phylogeny.PhylogenyFeatures`
                      and any other provider that needs taxonomic names.
        """
        self._df_y = df_y
        self._df_y_loc = df_y_loc
        self._validate()
        self._df_y.sort_index(inplace=True)
        self._df_y_loc.sort_index(inplace=True)
        self._index = self._build_index()

        # Complete sets: set once on the full dataset, propagated through splits
        self._locations_complete: List[Tuple[str, str]] = (
            list(locations_complete) if locations_complete is not None else self.locations
        )
        self._species_complete: List[Tuple[str, int]] = (
            list(species_complete) if species_complete is not None else self.species
        )
        self._species_subgroups_complete: List[Tuple[str, int, int]] = (
            list(species_subgroups_complete)
            if species_subgroups_complete is not None
            else self.species_subgroups
        )
        self._species_names: Dict[Tuple[str, int], str] = dict(species_names or {})

    # ------------------------------------------------------------------
    # Validation & index helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        required_obs_levels = list(KEYS_INDEX) + [KEY_OBS_TYPE]
        missing = [l for l in required_obs_levels if l not in self._df_y.index.names]
        if missing:
            raise DatasetException(f"df_y missing required index levels: {missing}")

        if KEY_OBSERVATIONS not in self._df_y.columns:
            raise DatasetException(f"df_y missing required column: '{KEY_OBSERVATIONS}'")

        required_loc_levels = [KEY_DATA_SOURCE, KEY_LOC_ID]
        missing = [l for l in required_loc_levels if l not in self._df_y_loc.index.names]
        if missing:
            raise DatasetException(f"df_y_loc missing required index levels: {missing}")

        for col in (KEY_LAT, KEY_LON):
            if col not in self._df_y_loc.columns:
                raise DatasetException(f"df_y_loc missing required column: '{col}'")

    def _build_index(self) -> pd.MultiIndex:
        """Union of base indices across all observation types."""
        empty = pd.MultiIndex.from_tuples([], names=list(KEYS_INDEX))
        groups = self._df_y.groupby(level=KEY_OBS_TYPE)
        index = reduce(
            pd.MultiIndex.union,
            [grp.xs(key, level=KEY_OBS_TYPE).index for key, grp in groups],
            empty,
        ).drop_duplicates(keep="first")
        return index.sortlevel(level=0, sort_remaining=True)[0]

    def _new(self, df_y: pd.DataFrame, df_y_loc: Optional[pd.DataFrame] = None) -> "Observations":
        return Observations(
            df_y,
            df_y_loc if df_y_loc is not None else self._df_y_loc,
            locations_complete=self._locations_complete,
            species_complete=self._species_complete,
            species_subgroups_complete=self._species_subgroups_complete,
            species_names=self._species_names,
        )

    # ------------------------------------------------------------------
    # Basic sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, index: Union[int, Tuple]) -> bool:
        return index in self._index

    def __getitem__(self, index: Union[int, Tuple]) -> Dict[str, Any]:
        """
        Return a dict for one entry keyed by the constants in pysephone.constants.

        Accepts either an integer position or a 5-tuple
        (src, loc_id, year, species_id, subgroup_id).
        """
        if isinstance(index, int):
            if not (0 <= index < len(self._index)):
                raise IndexError(f"Index {index} out of range (size {len(self._index)})")
            index = self._index[index]

        if not isinstance(index, tuple) or len(index) != len(KEYS_INDEX):
            raise DatasetException(
                f"index must be an int or a {len(KEYS_INDEX)}-tuple, got {type(index)}"
            )

        src, loc_id, year, species_id, subgroup_id = index

        try:
            df_obs = self._df_y.loc[src, loc_id, year, species_id, subgroup_id]
            observations = df_obs.to_dict()[KEY_OBSERVATIONS]
        except KeyError as exc:
            raise DatasetException(f"No observations found for {index}: {exc}") from exc

        coords = self.get_location_coordinates((src, loc_id))

        return {
            KEY_DATA_SOURCE: src,
            KEY_LOC_ID: loc_id,
            KEY_YEAR: year,
            KEY_SPECIES_ID: species_id,
            KEY_SUBGROUP_ID: subgroup_id,
            KEY_OBSERVATIONS: observations,
            KEY_LAT: coords[KEY_LAT],
            KEY_LON: coords[KEY_LON],
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_index(self) -> Iterator[Tuple]:
        yield from self._index

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        for ix in self.iter_index():
            yield self[ix]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> pd.MultiIndex:
        return self._index

    @property
    def df_y(self) -> pd.DataFrame:
        return self._df_y

    @property
    def df_y_loc(self) -> pd.DataFrame:
        return self._df_y_loc

    @property
    def locations(self) -> List[Tuple[str, str]]:
        return list(set(zip(
            self._index.get_level_values(KEY_DATA_SOURCE),
            self._index.get_level_values(KEY_LOC_ID),
        )))

    @property
    def species(self) -> List[Tuple[str, int]]:
        return list(set(zip(
            self._index.get_level_values(KEY_DATA_SOURCE),
            self._index.get_level_values(KEY_SPECIES_ID),
        )))

    @property
    def species_subgroups(self) -> List[Tuple[str, int, int]]:
        return list(set(zip(
            self._index.get_level_values(KEY_DATA_SOURCE),
            self._index.get_level_values(KEY_SPECIES_ID),
            self._index.get_level_values(KEY_SUBGROUP_ID),
        )))

    @property
    def locations_complete(self) -> List[Tuple[str, str]]:
        """Locations from the original dataset before any splits or selections."""
        return list(self._locations_complete)

    @property
    def species_complete(self) -> List[Tuple[str, int]]:
        """Species from the original dataset before any splits or selections."""
        return list(self._species_complete)

    @property
    def species_subgroups_complete(self) -> List[Tuple[str, int, int]]:
        """(species, subgroup) pairs from the original dataset before any splits or selections."""
        return list(self._species_subgroups_complete)

    @property
    def species_names(self) -> Dict[Tuple[str, int], str]:
        """Scientific name lookup: ``{(src, species_id): name}``.

        Empty dict when the data source did not provide names.
        """
        return dict(self._species_names)

    def get_species_name(self, src: str, species_id: int) -> Optional[str]:
        """Return the scientific name for ``(src, species_id)``, or ``None``."""
        return self._species_names.get((src, species_id))

    @property
    def years(self) -> List[int]:
        return sorted(set(self._index.get_level_values(KEY_YEAR)))

    @property
    def observation_types(self) -> List[str]:
        return list(self._df_y.index.get_level_values(KEY_OBS_TYPE).unique())

    @property
    def num_observation_types(self) -> int:
        return self._df_y.index.get_level_values(KEY_OBS_TYPE).nunique()

    @property
    def bounding_box(self) -> Dict[str, float]:
        if len(self._index) == 0:
            return {"min_lat": 0.0, "max_lat": 0.0, "min_lon": 0.0, "max_lon": 0.0}
        lats = self._df_y_loc[KEY_LAT]
        lons = self._df_y_loc[KEY_LON]
        return {
            "min_lat": lats.min(),
            "max_lat": lats.max(),
            "min_lon": lons.min(),
            "max_lon": lons.max(),
        }

    # ------------------------------------------------------------------
    # Location helpers
    # ------------------------------------------------------------------

    def get_location_coordinates(
        self, loc: Tuple[str, str], from_index: bool = False
    ) -> Dict[str, float]:
        """
        Args:
            loc:        (src, loc_id) or, if from_index=True, the full 5-tuple index.
            from_index: If True, extract (src, loc_id) from a 5-tuple.
        """
        if from_index:
            src, loc_id = loc[0], loc[1]
        else:
            src, loc_id = loc
        row = self._df_y_loc.loc[src, loc_id]
        return {KEY_LAT: row[KEY_LAT], KEY_LON: row[KEY_LON]}

    def get_location_name(
        self, loc: Tuple[str, str], from_index: bool = False
    ) -> str:
        if from_index:
            src, loc_id = loc[0], loc[1]
        else:
            src, loc_id = loc
        return self._df_y_loc.loc[src, loc_id][KEY_LOC_NAME]

    # ------------------------------------------------------------------
    # Selection — all return new Observations
    # ------------------------------------------------------------------

    def select_locations(
        self, locations: Union[Tuple[str, str], List[Tuple[str, str]]]
    ) -> "Observations":
        return self._new(select_location(self._df_y, locations))

    def select_years(self, years: Union[int, List[int]]) -> "Observations":
        return self._new(select_year(self._df_y, years))

    def select_species(
        self, species: Union[Tuple, List[Tuple]]
    ) -> "Observations":
        return self._new(select_species(self._df_y, species))

    def select_by_observation_requirement(
        self, obs_key: Union[str, List[str]]
    ) -> "Observations":
        """Keep only entries that have ALL of the specified observation types."""
        if not isinstance(obs_key, list):
            obs_key = [obs_key]

        selected_rows = []
        for ix in self.iter_index():
            if all((ix + (key,)) in self._df_y.index for key in obs_key):
                selected_rows.extend(ix + (key,) for key in obs_key)

        if not selected_rows:
            return self._new(empty_df_like(self._df_y))

        return self._new(self._df_y.loc[selected_rows])

    def select_by_local_num_observations(
        self, num_observations: int, obs_key: str
    ) -> "Observations":
        """Keep only (src, loc_id, species_id, subgroup_id) groups with at least
        num_observations entries of obs_key."""
        assert num_observations >= 0
        if num_observations == 0:
            return self._new(self._df_y)

        df_y = self._df_y.groupby(level=[0, 1, 3, 4]).filter(
            lambda df: len(df.xs(obs_key, level=KEY_OBS_TYPE)) >= num_observations
        )
        return self._new(df_y)

    def select_by_ixs(self, ixs: List[Tuple]) -> "Observations":
        """Keep only entries whose base index appears in ixs."""
        if not isinstance(ixs, list):
            raise TypeError(f"ixs must be a list, got {type(ixs)}")
        if not ixs:
            return self._new(empty_df_like(self._df_y))

        df_y = pd.concat([
            self._df_y.xs(
                (src, loc_id, year, spc, sub),
                level=[KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID],
                drop_level=False,
            )
            for src, loc_id, year, spc, sub in ixs
        ])
        return self._new(df_y)

    # ------------------------------------------------------------------
    # Aggregation & splitting
    # ------------------------------------------------------------------

    def aggregate_in_grid(
        self,
        method: str = "median",
        grid_size: Optional[Tuple[float, float]] = None,
    ) -> "Observations":
        """Aggregate observations per spatial grid cell.

        Args:
            method:    'median' (default).  'mean' and 'first' are not yet implemented.
            grid_size: (lat_cell_size, lon_cell_size) in degrees.
                       Defaults to (0.5, 0.5) when None.
        """
        if grid_size is None:
            grid_size = (0.5, 0.5)
        res_lat, res_lon = grid_size

        df_sel = self._df_y.join(self._df_y_loc[[KEY_LAT, KEY_LON]])
        df_sel[KEY_LAT] = df_sel[KEY_LAT].apply(lambda x: round_partial(x, res_lat))
        df_sel[KEY_LON] = df_sel[KEY_LON].apply(lambda x: round_partial(x, res_lon))

        match method:
            case "median":
                df_sel.reset_index(inplace=True)
                df = df_sel.groupby(
                    [KEY_LAT, KEY_LON, KEY_OBS_TYPE, KEY_YEAR], as_index=False
                ).agg({
                    KEY_LAT: "first",
                    KEY_LON: "first",
                    KEY_OBS_TYPE: "first",
                    KEY_YEAR: "first",
                    KEY_OBSERVATIONS: "median",
                    KEY_DATA_SOURCE: "first",
                    KEY_LOC_ID: "first",
                    KEY_SPECIES_ID: "first",
                    KEY_SUBGROUP_ID: "first",
                })
                df.reset_index(inplace=True)
                df.set_index(list(KEYS_INDEX) + [KEY_OBS_TYPE], inplace=True)
                df.drop([KEY_LAT, KEY_LON, "index"], axis=1, inplace=True)
                return self._new(df)

            case "mean" | "first":
                raise NotImplementedError(f"Aggregation method '{method}' is not yet implemented")

            case _:
                raise DatasetException(f'Unknown aggregation method "{method}"')

    def split_by_grid(
        self,
        grid_size: Tuple[float, float],
        split_size: float,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> Tuple["Observations", "Observations", Dict[str, Any]]:
        """Spatially split by assigning locations to grid cells.

        Returns (part_1, part_2, info_dict) where info_dict contains
        'cells_1', 'cells_2', 'cells_to_locations', and 'cell_size'.
        """
        assert 0 <= split_size <= 1
        lat_size, lon_size = grid_size

        cell_to_locations: Dict[Tuple, set] = defaultdict(set)
        for loc in self.locations:
            coords = self.get_location_coordinates(loc)
            cell = (
                round_partial(coords[KEY_LAT], lat_size),
                round_partial(coords[KEY_LON], lon_size),
            )
            cell_to_locations[cell].add(loc)

        cells = sorted(cell_to_locations)
        cells_1, cells_2 = train_test_split(
            cells, train_size=split_size, random_state=random_state, shuffle=shuffle
        )

        locs_1 = set.union(*[cell_to_locations[c] for c in cells_1])
        locs_2 = set.union(*[cell_to_locations[c] for c in cells_2])
        assert not locs_1.intersection(locs_2)

        info = {
            "cells_1": cells_1,
            "cells_2": cells_2,
            "cells_to_locations": cell_to_locations,
            "cell_size": grid_size,
        }
        return self.select_locations(list(locs_1)), self.select_locations(list(locs_2)), info

    def split_by_lon_border(
        self, lon_border: float
    ) -> Tuple["Observations", "Observations", Dict]:
        below, above = [], []
        for ix in self.iter_index():
            lon = self.get_location_coordinates(ix, from_index=True)[KEY_LON]
            (below if lon <= lon_border else above).append(ix)
        return self.select_by_ixs(below), self.select_by_ixs(above), {}

    def split_by_lat_border(
        self, lat_border: float
    ) -> Tuple["Observations", "Observations", Dict]:
        below, above = [], []
        for ix in self.iter_index():
            lat = self.get_location_coordinates(ix, from_index=True)[KEY_LAT]
            (below if lat <= lat_border else above).append(ix)
        return self.select_by_ixs(below), self.select_by_ixs(above), {}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def observation_counts(self) -> Dict[str, int]:
        return self._df_y.value_counts(subset=KEY_OBS_TYPE).to_dict()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def merge(a: "Observations", b: "Observations") -> "Observations":
        """Combine two Observations objects, deduplicating on index."""
        df_y = pd.concat([a._df_y, b._df_y])
        df_y = df_y[~df_y.index.duplicated(keep="first")]
        df_y_loc = pd.concat([a._df_y_loc, b._df_y_loc])
        df_y_loc = df_y_loc[~df_y_loc.index.duplicated(keep="first")]
        merged_names = {**a._species_names, **b._species_names}
        return Observations(df_y, df_y_loc, species_names=merged_names)

    @staticmethod
    def shift_year(
        df_y: pd.DataFrame, obs_type: str, offset: int
    ) -> pd.DataFrame:
        """Add *offset* to the year index level for rows of *obs_type*.

        Useful for winter-crop datasets where sowing (BBCH_0) happens in the
        previous calendar year relative to harvest.
        """
        groups = df_y.groupby(level=KEY_OBS_TYPE)
        parts = []
        for key, grp in groups:
            if key != obs_type:
                parts.append(grp)
            else:
                grp = grp.reset_index()
                grp[KEY_YEAR] = grp[KEY_YEAR] + offset
                grp.set_index(list(KEYS_INDEX) + [KEY_OBS_TYPE], inplace=True)
                parts.append(grp)
        return pd.concat(parts)
