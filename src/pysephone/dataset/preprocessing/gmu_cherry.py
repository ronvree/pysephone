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
    KEY_ALT,
    KEY_COUNTRY_CODE,
    KEY_LOC_NAME,
)
from pysephone.data.gmu_cherry.bloom_doy import get_df_japan, get_df_switzerland, get_df_south_korea
from pysephone.data.gmu_cherry.regions_data import LOCATION_VARIETY_JAPAN
from pysephone.dataset.util.func import filter_outliers

_SRC = 'gmu_cherry'


def get_gmu_cherry_dataset_japan(remove_outliers: bool = True,
                                 datetime_observations: bool = True,
                                 ) -> dict:

    df_data = get_df_japan()
    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[KEY_DATA_SOURCE] = _SRC
    df_y_loc[KEY_COUNTRY_CODE] = 'JP'
    df_y_loc[KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(columns={'location': KEY_LOC_ID, 'lat': KEY_LAT, 'long': KEY_LON, 'alt': KEY_ALT}, inplace=True)
    df_y_loc.set_index([KEY_DATA_SOURCE, KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_0'  # TODO -- proper name

    loc_species_map = {k.replace('/', '__'): v for k, v in LOCATION_VARIETY_JAPAN.items()}

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[KEY_DATA_SOURCE] = _SRC
    df_y[KEY_OBS_TYPE] = bloom_obs_type
    # Filter to locations that have species information
    df_y = df_y[df_y['location'].isin(loc_species_map.keys())]
    df_y[KEY_SPECIES_ID] = df_y['location'].map(loc_species_map)
    df_y[KEY_SUBGROUP_ID] = 0  # Unknown
    df_y.rename(columns={'location': KEY_LOC_ID, 'year': KEY_YEAR, 'bloom_doy': KEY_OBSERVATIONS}, inplace=True)
    df_y.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)

    if remove_outliers:
        df_y = filter_outliers(df_y)

    # If set -> convert DOY observations to datetime objects
    if datetime_observations:
        years = df_y.index.get_level_values(KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        days = (df_y[KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        df_y[KEY_OBSERVATIONS] = years + days

    return {'data': df_y, 'locations': df_y_loc}


def get_gmu_cherry_dataset_switzerland(remove_outliers: bool = True,
                                       datetime_observations: bool = True,
                                       ) -> dict:

    df_data = get_df_switzerland()
    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[KEY_DATA_SOURCE] = _SRC
    df_y_loc[KEY_COUNTRY_CODE] = 'SW'
    df_y_loc[KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(columns={'location': KEY_LOC_ID, 'lat': KEY_LAT, 'long': KEY_LON, 'alt': KEY_ALT}, inplace=True)
    df_y_loc.set_index([KEY_DATA_SOURCE, KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_1'  # TODO

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[KEY_DATA_SOURCE] = _SRC
    df_y[KEY_OBS_TYPE] = bloom_obs_type
    df_y[KEY_SPECIES_ID] = 5
    df_y[KEY_SUBGROUP_ID] = 0
    df_y.rename(columns={'location': KEY_LOC_ID, 'year': KEY_YEAR, 'bloom_doy': KEY_OBSERVATIONS}, inplace=True)
    df_y.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)

    if remove_outliers:
        df_y = filter_outliers(df_y)

    if datetime_observations:
        years = df_y.index.get_level_values(KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        days = (df_y[KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        df_y[KEY_OBSERVATIONS] = years + days

    return {'data': df_y, 'locations': df_y_loc}


def get_gmu_cherry_dataset_south_korea(remove_outliers: bool = True,
                                       datetime_observations: bool = True,
                                       ) -> dict:

    df_data = get_df_south_korea()
    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[KEY_DATA_SOURCE] = _SRC
    df_y_loc[KEY_COUNTRY_CODE] = 'KR'
    df_y_loc[KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(columns={'location': KEY_LOC_ID, 'lat': KEY_LAT, 'long': KEY_LON, 'alt': KEY_ALT}, inplace=True)
    df_y_loc.set_index([KEY_DATA_SOURCE, KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_2'

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[KEY_DATA_SOURCE] = _SRC
    df_y[KEY_OBS_TYPE] = bloom_obs_type
    df_y[KEY_SPECIES_ID] = 0
    df_y[KEY_SUBGROUP_ID] = 0
    df_y.rename(columns={'location': KEY_LOC_ID, 'year': KEY_YEAR, 'bloom_doy': KEY_OBSERVATIONS}, inplace=True)
    df_y.set_index([KEY_DATA_SOURCE, KEY_LOC_ID, KEY_YEAR, KEY_SPECIES_ID, KEY_SUBGROUP_ID, KEY_OBS_TYPE], inplace=True)

    if remove_outliers:
        df_y = filter_outliers(df_y)

    if datetime_observations:
        years = df_y.index.get_level_values(KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        days = (df_y[KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        df_y[KEY_OBSERVATIONS] = years + days

    return {'data': df_y, 'locations': df_y_loc}


if __name__ == '__main__':
    dfs = get_gmu_cherry_dataset_japan(datetime_observations=True)
    print(dfs['data'])
    get_gmu_cherry_dataset_switzerland(datetime_observations=False)
    get_gmu_cherry_dataset_south_korea(datetime_observations=False)
