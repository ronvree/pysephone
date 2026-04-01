import numpy as np
import pandas as pd

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_LOC_ID,
    KEY_YEAR,
    KEY_SPECIES_ID,
    KEY_SUBGROUP_ID,
    KEY_OBS_TYPE,
    KEY_OBSERVATIONS,
)


class DatasetException(Exception):
    pass


def empty_df_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns).reindex_like(df).dropna().astype(df.dtypes)


def filter_outliers(df: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Remove per-obs_type outliers via symmetric quantile trimming.

    Works for both datetime and integer day-of-year observation columns.
    """
    col = df[KEY_OBSERVATIONS]
    values = col.dt.dayofyear if pd.api.types.is_datetime64_any_dtype(col) else col
    temp = df.assign(_doy=values)
    groups = {
        k: grp[
            (grp['_doy'].quantile(q) < grp['_doy']) & (grp['_doy'] < grp['_doy'].quantile(1 - q))
        ]
        for k, grp in temp.groupby(KEY_OBS_TYPE)
    }
    if not groups:
        return df
    return pd.concat(groups.values()).drop(columns='_doy')


def select_species(df: pd.DataFrame, species) -> pd.DataFrame:
    """Filter observations by species (and optionally subgroup).

    Args:
        df:      Observations DataFrame with standard MultiIndex.
        species: A 2-tuple (src, species_id), a 3-tuple (src, species_id, subgroup_id),
                 or a list of such tuples.
    """
    assert isinstance(species, (tuple, list))

    if isinstance(species, list):
        if len(species) == 0:
            return empty_df_like(df)
        return pd.concat([select_species(df, s) for s in species], axis=0)

    assert len(species) in (2, 3)

    if len(species) == 3:
        src, species_id, subgroup_id = species
        try:
            return df.xs(
                (src, species_id, subgroup_id),
                level=[KEY_DATA_SOURCE, KEY_SPECIES_ID, KEY_SUBGROUP_ID],
                drop_level=False,
            )
        except KeyError:
            return empty_df_like(df)

    src, species_id = species
    try:
        return df.xs(
            (src, species_id),
            level=[KEY_DATA_SOURCE, KEY_SPECIES_ID],
            drop_level=False,
        )
    except KeyError:
        return empty_df_like(df)


def select_year(df: pd.DataFrame, year) -> pd.DataFrame:
    """Filter observations by year or list of years."""
    assert isinstance(year, (int, list))

    if isinstance(year, list):
        if len(year) == 0:
            return empty_df_like(df)
        return pd.concat([select_year(df, y) for y in year], axis=0)

    try:
        return df.xs(year, level=KEY_YEAR, drop_level=False)
    except KeyError:
        return empty_df_like(df)


def select_location(df: pd.DataFrame, location) -> pd.DataFrame:
    """Filter observations by location or list of locations.

    Args:
        location: A 2-tuple (src, loc_id) or a list of such tuples.
    """
    assert isinstance(location, (tuple, list))

    if isinstance(location, list):
        if len(location) == 0:
            return empty_df_like(df)
        return pd.concat([select_location(df, loc) for loc in location], axis=0)

    src, loc_id = location
    try:
        return df.xs(
            (src, loc_id),
            level=[KEY_DATA_SOURCE, KEY_LOC_ID],
            drop_level=False,
        )
    except KeyError:
        return empty_df_like(df)


def select_observation_type(df: pd.DataFrame, obs_type) -> pd.DataFrame:
    """Filter observations by observation type or list of types."""
    if isinstance(obs_type, list):
        if len(obs_type) == 0:
            return empty_df_like(df)
        return pd.concat([select_observation_type(df, t) for t in obs_type], axis=0)

    try:
        return df.xs(obs_type, level=KEY_OBS_TYPE, drop_level=False)
    except KeyError:
        return empty_df_like(df)


def doy_to_date_in_year(year: int, doy: int) -> np.datetime64:
    """Convert a day-of-year integer to a numpy datetime64 in the given year."""
    assert 0 < doy <= 365
    return np.datetime64(f'{year}-01-01') + np.timedelta64(doy - 1, 'D')
