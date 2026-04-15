"""
GMU Cherry Blossom dataset definitions.

Each public builder returns an Observations object for one pre-configured
dataset variant.  Shared configuration lives at module level.

Observation type keys:
    gmu_0  — Japan bloom DOY
    gmu_1  — Switzerland bloom DOY
    gmu_2  — South Korea bloom DOY
"""

from datetime import datetime
from functools import reduce
from typing import List

from pysephone.dataset.observations import Observations
from pysephone.dataset.preprocessing.gmu_cherry import (
    get_gmu_cherry_dataset_japan,
    get_gmu_cherry_dataset_switzerland,
    get_gmu_cherry_dataset_south_korea,
)

_GMU_SRC = 'GMU_cherry'

# ---------------------------------------------------------------------------
# Shared configuration blocks
# ---------------------------------------------------------------------------

# General benchmark (bm)
_BM_YEAR_MIN: int = 1986
_BM_YEAR_MAX: int = datetime.now().year - 1
_BM_YEARS: List[int] = list(range(_BM_YEAR_MIN, _BM_YEAR_MAX + 1))
_BM_DO_AGG: bool = True
_BM_AGG_METHOD: str = 'median'
_BM_REMOVE_OUTLIERS: bool = True
_BM_ASSERT_TARGET: bool = True

# Test Earth Embedding (tbe)
_TBE_YEAR_MIN: int = 2017
_TBE_YEAR_MAX: int = datetime.now().year - 1
_TBE_YEARS: List[int] = list(range(_TBE_YEAR_MIN, _TBE_YEAR_MAX + 1))
_TBE_DO_AGG: bool = False
_TBE_AGG_METHOD: str = 'median'
_TBE_ASSERT_TARGET: bool = True


# ---------------------------------------------------------------------------
# GMU Cherry — benchmark variants
# ---------------------------------------------------------------------------

def build_gmu_cherry_japan(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_japan(
        remove_outliers=False,  # Multiple species occur in this dataset — outlier removal skipped
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_BM_YEARS)
    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_0')
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


def build_gmu_cherry_switzerland(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_switzerland(
        remove_outliers=_BM_REMOVE_OUTLIERS,
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_BM_YEARS)
    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_1')
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


def build_gmu_cherry_south_korea(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_south_korea(
        remove_outliers=_BM_REMOVE_OUTLIERS,
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_BM_YEARS)
    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_2')
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


def build_gmu_cherry_japan_y(**kwargs) -> Observations:
    """Japan — Cerasus yedoensis (Somei-yoshino) locations only."""
    from pysephone.data.gmu_cherry.regions_data import LOCATIONS_JAPAN_YEDOENSIS

    dfs = get_gmu_cherry_dataset_japan(
        remove_outliers=False,  # Multiple species — outlier removal skipped
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_BM_YEARS)

    locations_y = [(_GMU_SRC, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_YEDOENSIS.keys()]
    obs = obs.select_locations(locations_y)

    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_0')
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


def build_gmu_cherry_japan_ys(**kwargs) -> Observations:
    """Japan — Cerasus yedoensis (Somei-yoshino) + C. sargentii locations."""
    from pysephone.data.gmu_cherry.regions_data import LOCATIONS_JAPAN_YEDOENSIS, LOCATIONS_JAPAN_SARGENTII

    dfs = get_gmu_cherry_dataset_japan(
        remove_outliers=False,  # Multiple species — outlier removal skipped
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_BM_YEARS)

    locations_y = [(_GMU_SRC, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_YEDOENSIS.keys()]
    locations_s = [(_GMU_SRC, loc.replace('/', '__')) for loc in LOCATIONS_JAPAN_SARGENTII.keys()]
    obs = obs.select_locations(locations_y + locations_s)

    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_0')
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


# ---------------------------------------------------------------------------
# GMU Cherry — TBE (Test Earth Embedding) variants
# ---------------------------------------------------------------------------

def build_tbe_gmu_cherry_japan(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_japan(
        remove_outliers=False,  # Multiple species — outlier removal skipped
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_TBE_YEARS)
    if _TBE_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_0')
    if _TBE_DO_AGG:
        obs = obs.aggregate_in_grid(method=_TBE_AGG_METHOD)
    return obs


def build_tbe_gmu_cherry_switzerland(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_switzerland(
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_TBE_YEARS)
    if _TBE_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_1')
    if _TBE_DO_AGG:
        obs = obs.aggregate_in_grid(method=_TBE_AGG_METHOD)
    return obs


def build_tbe_gmu_cherry_south_korea(**kwargs) -> Observations:
    dfs = get_gmu_cherry_dataset_south_korea(
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'])
    obs = obs.select_years(_TBE_YEARS)
    if _TBE_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement('gmu_2')
    if _TBE_DO_AGG:
        obs = obs.aggregate_in_grid(method=_TBE_AGG_METHOD)
    return obs


# ---------------------------------------------------------------------------
# Composite: all fruit trees (GMU + PEP725), harmonised to a single obs type
# ---------------------------------------------------------------------------

def build_all_fruit_trees(**kwargs) -> Observations:
    """Merge GMU Cherry regions with PEP725 fruit trees, unifying obs type to 'mixed'."""
    import pandas as pd
    from pysephone.constants import KEY_OBS_TYPE

    # Map each sub-dataset key to its primary observation type
    source_obs_map = {
        'GMU_Cherry_Japan':       'gmu_0',
        'GMU_Cherry_Switzerland': 'gmu_1',
        'GMU_Cherry_South_Korea': 'gmu_2',
        'PEP725_fruit_trees':     'BBCH_60',
    }

    mixed_key = 'mixed'
    parts: list[Observations] = []

    for ds_key, obs_key in source_obs_map.items():
        obs = DATASETS[ds_key]()
        # Rename the primary obs_type to 'mixed' so all sources share one key
        df_y = obs.df_y.reset_index()
        df_y[KEY_OBS_TYPE] = df_y[KEY_OBS_TYPE].replace({obs_key: mixed_key})
        from pysephone.constants import KEYS_INDEX
        df_y.set_index(list(KEYS_INDEX) + [KEY_OBS_TYPE], inplace=True)
        parts.append(Observations(df_y, obs.df_y_loc))

    return reduce(Observations.merge, parts)


# ---------------------------------------------------------------------------
# Calendar configuration: per-dataset season window defaults
# ---------------------------------------------------------------------------

CALENDAR_CONFIGS = {}


# ---------------------------------------------------------------------------
# Registry mapping for GMU Cherry datasets
# ---------------------------------------------------------------------------

DATASETS = {
    # Benchmark variants
    'GMU_Cherry_Japan':       build_gmu_cherry_japan,
    'GMU_Cherry_Switzerland': build_gmu_cherry_switzerland,
    'GMU_Cherry_South_Korea': build_gmu_cherry_south_korea,
    'GMU_Cherry_Japan_Y':     build_gmu_cherry_japan_y,
    'GMU_Cherry_Japan_YS':    build_gmu_cherry_japan_ys,
    # TBE variants
    'TBE_GMU_Cherry_Japan':       build_tbe_gmu_cherry_japan,
    'TBE_GMU_Cherry_Switzerland': build_tbe_gmu_cherry_switzerland,
    'TBE_GMU_Cherry_South_Korea': build_tbe_gmu_cherry_south_korea,
}
