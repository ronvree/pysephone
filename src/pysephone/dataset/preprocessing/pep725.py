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
    KEY_LAT,
    KEY_LON,
)
from pysephone.data.source import ObservationData
from pysephone.dataset.util.func import (
    DatasetException,
    filter_outliers,
    select_species,
    select_year,
    select_location,
    select_observation_type,
)
from pysephone.utils.func import round_partial


"""
    Preprocess PEP725 ObservationData into dataframes compatible with the dataset classes.
"""

# Spatial aggregation grid — only used in the deprecated _filter_in_grid
_AGG_GRID = None
_RNG_DATA_PREPROCESSING = None


def get_pep725_dataframes(data: ObservationData,
                          remove_outliers: bool = True,
                          filter_on_species=None,
                          filter_on_years=None,
                          filter_on_locations=None,
                          filter_on_observation_types=None,
                          datetime_observations: bool = True,
                          ) -> dict:
    """
    Process a PEP725 ObservationData into dataframes compatible with a dataset class.

    Args:
        data:                        ObservationData produced by PEP725Source.get_data().
        remove_outliers:             Remove per-obs_type outliers via symmetric quantile trimming.
        filter_on_species:           Optional iterable of species_id values to keep.
        filter_on_years:             Optional iterable of year values to keep.
        filter_on_locations:         Optional iterable of loc_id values to keep.
        filter_on_observation_types: Optional iterable of obs_type values to keep.
        datetime_observations:       If True, keep the KEY_OBSERVATIONS column as datetime.
                                     If False, replace with integer day-of-year.

    Returns:
        dict with keys:
            'data':      observations DataFrame indexed by the standard multi-index.
            'locations': locations DataFrame indexed by (src, loc_id).
    """

    df_y_loc = data.locations.copy()

    df_y = data.observations.copy()

    if filter_on_species is not None:
        df_y = select_species(df_y, filter_on_species)

    if filter_on_years is not None:
        df_y = select_year(df_y, filter_on_years)

    if filter_on_locations is not None:
        df_y = select_location(df_y, filter_on_locations)

    if filter_on_observation_types is not None:
        df_y = select_observation_type(df_y, filter_on_observation_types)

    if remove_outliers:
        df_y = filter_outliers(df_y)

    if not datetime_observations:
        df_y[KEY_OBSERVATIONS] = df_y[KEY_OBSERVATIONS].dt.dayofyear

    return {'data': df_y, 'locations': df_y_loc}


def _filter_in_grid(df: pd.DataFrame, df_loc: pd.DataFrame, method: str = None) -> pd.DataFrame:
    assert False, 'Method has been temporarily deprecated and should not be used since it throws away data unnecessarily. Filtering should happen after pairing observations based on type'

    df_selection = df.join(df_loc[[KEY_LAT, KEY_LON]])

    res_lat, res_lon = _AGG_GRID

    df_selection[KEY_LAT] = df_selection[KEY_LAT].apply(lambda x: round_partial(x, res_lat))
    df_selection[KEY_LON] = df_selection[KEY_LON].apply(lambda x: round_partial(x, res_lon))

    match method:
        case 'first':
            df_selection.sort_index(inplace=True)
            df_selection = df_selection.sample(frac=1, random_state=_RNG_DATA_PREPROCESSING)
            df_selection.reset_index(inplace=True)
            df = df_selection.groupby(
                [KEY_LAT, KEY_LON, KEY_OBS_TYPE, KEY_YEAR], as_index=False,
            ).first()
            df.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)
            df.drop([KEY_LAT, KEY_LON], axis=1, inplace=True)
            return df

        case 'mean':
            df_selection.sort_index(inplace=True)
            df_selection = df_selection.sample(frac=1, random_state=_RNG_DATA_PREPROCESSING)
            df_selection.reset_index(inplace=True)
            df = df_selection.groupby(
                [KEY_LAT, KEY_LON, KEY_OBS_TYPE, KEY_YEAR], as_index=False,
            ).agg({
                KEY_LAT: 'first', KEY_LON: 'first',
                KEY_OBS_TYPE: 'first', KEY_YEAR: 'first',
                KEY_OBSERVATIONS: 'mean',
                KEY_DATA_SOURCE: 'first', KEY_LOC_ID: 'first',
                KEY_SPECIES_ID: 'first', KEY_SUBGROUP_ID: 'first',
            })
            df.reset_index(inplace=True)
            df[KEY_OBSERVATIONS] = (df[KEY_OBSERVATIONS] + 0.5).astype(int)
            df.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)
            df.drop([KEY_LAT, KEY_LON], axis=1, inplace=True)
            return df

        case 'median':
            df_selection.sort_index(inplace=True)
            df_selection = df_selection.sample(frac=1, random_state=_RNG_DATA_PREPROCESSING)
            df_selection.reset_index(inplace=True)
            df = df_selection.groupby(
                [KEY_LAT, KEY_LON, KEY_OBS_TYPE, KEY_YEAR], as_index=False,
            ).agg({
                KEY_LAT: 'first', KEY_LON: 'first',
                KEY_OBS_TYPE: 'first', KEY_YEAR: 'first',
                KEY_OBSERVATIONS: 'median',
                KEY_DATA_SOURCE: 'first', KEY_LOC_ID: 'first',
                KEY_SPECIES_ID: 'first', KEY_SUBGROUP_ID: 'first',
            })
            df.reset_index(inplace=True)
            df.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)
            df.drop([KEY_LAT, KEY_LON], axis=1, inplace=True)
            return df

        case _:
            raise DatasetException(f'Unknown aggregation method "{method}" for phenological observations')


if __name__ == '__main__':
    from pysephone.data.pep725.source import PEP725Source
    _result = get_pep725_dataframes(PEP725Source().get_data({}))
