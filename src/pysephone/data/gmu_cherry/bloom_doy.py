import pandas as pd
from unidecode import unidecode

from pysephone.data.gmu_cherry.download import (
    get_data_japan,
    get_data_kyoto,
    get_data_liestal,
    get_data_meteoswiss,
    get_data_south_korea,
)


COUNTRIES = ('Japan', 'South Korea', 'Switzerland',)

# Spaces are removed; hyphens and other punctuation are replaced with underscores.
_COUNTRY_TRANS = str.maketrans('', '', ' -')
_LOCATION_TRANS = str.maketrans("-',.()","______", ' ')


def _sanitize_location_label(s: str) -> str:
    tokens = s.split('/')
    if len(tokens) < 2:
        raise ValueError(f'Could not parse location entry "{s}"')
    country = tokens[0].translate(_COUNTRY_TRANS)
    location = unidecode(','.join(tokens[1:]).translate(_LOCATION_TRANS))
    return f'{country}/{location}'


def get_df_japan() -> pd.DataFrame:
    df_japan = get_data_japan()
    df_japan['location'] = df_japan['location'].apply(_sanitize_location_label)
    df_japan = df_japan.drop_duplicates()

    # The following locations have data for two separate stations; disambiguate by altitude/lat.
    df_japan.loc[(df_japan['location'] == 'Japan/Kushiro') & (df_japan['alt'] == 4.5), 'location'] = 'Japan/Kushiro_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Kushiro') & (df_japan['alt'] == 14.05), 'location'] = 'Japan/Kushiro_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Muroran') & (df_japan['alt'] == 39.89), 'location'] = 'Japan/Muroran_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Muroran') & (df_japan['alt'] == 3.00), 'location'] = 'Japan/Muroran_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Sendai') & (df_japan['alt'] == 38.85), 'location'] = 'Japan/Sendai_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Sendai') & (df_japan['alt'] == 37.90), 'location'] = 'Japan/Sendai_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Nagoya') & (df_japan['lat'] < 35.168), 'location'] = 'Japan/Nagoya_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Nagoya') & (df_japan['lat'] > 35.168), 'location'] = 'Japan/Nagoya_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Tottori') & (df_japan['alt'] == 7.1), 'location'] = 'Japan/Tottori_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Tottori') & (df_japan['alt'] == 6.0), 'location'] = 'Japan/Tottori_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Izuhara') & (df_japan['alt'] == 3.65), 'location'] = 'Japan/Izuhara_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Izuhara') & (df_japan['alt'] == 130.00), 'location'] = 'Japan/Izuhara_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Yakushima') & (df_japan['alt'] == 37.3), 'location'] = 'Japan/Yakushima_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Yakushima') & (df_japan['alt'] == 36.0), 'location'] = 'Japan/Yakushima_2'
    df_japan.loc[(df_japan['location'] == 'Japan/Kochi') & (df_japan['alt'] == 0.5), 'location'] = 'Japan/Kochi_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Kochi') & (df_japan['alt'] == 3.0), 'location'] = 'Japan/Kochi_2'

    df_kyoto = get_data_kyoto()
    df_kyoto['location'] = 'Japan/Kyoto_2'
    df_japan['location'] = df_japan['location'].replace('Japan/Kyoto', 'Japan/Kyoto_1')

    return pd.concat([df_japan, df_kyoto])


def get_df_switzerland() -> pd.DataFrame:
    df_swiss = get_data_meteoswiss()
    df_swiss['location'] = df_swiss['location'].apply(_sanitize_location_label)

    df_liestal = get_data_liestal()
    df_liestal['location'] = 'Switzerland/Liestal_2'
    df_swiss['location'] = df_swiss['location'].replace('Switzerland/Liestal', 'Switzerland/Liestal_1')

    return pd.concat([df_swiss, df_liestal])


def get_df_south_korea() -> pd.DataFrame:
    df = get_data_south_korea()
    df['location'] = df['location'].apply(_sanitize_location_label)
    return df


def get_locations_japan() -> list:
    return list(set(get_df_japan()['location']))


def get_locations_coordinates_japan() -> pd.DataFrame:
    df = get_data_japan()
    df.set_index('location', inplace=True)
    return df[['lat', 'long', 'alt']].drop_duplicates()


def get_locations_switzerland() -> list:
    return list(set(get_df_switzerland()['location']))


def get_locations_coordinates_switzerland() -> pd.DataFrame:
    df = get_df_switzerland()
    df.set_index('location', inplace=True)
    return df[['lat', 'long', 'alt']].drop_duplicates()


def get_locations_south_korea() -> list:
    return list(set(get_df_south_korea()['location']))


def get_locations_coordinates_south_korea() -> pd.DataFrame:
    df = get_df_south_korea()
    df.set_index('location', inplace=True)
    return df[['lat', 'long', 'alt']].drop_duplicates()
