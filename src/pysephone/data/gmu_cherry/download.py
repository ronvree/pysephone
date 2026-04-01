from pathlib import Path

import pandas as pd


_DATA_DIR = Path(__file__).parent / 'data'

COLUMNS = ('location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy')


def get_data_japan() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'japan.csv')


def get_data_kyoto() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'kyoto.csv')


def get_data_liestal() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'liestal.csv')


def get_data_meteoswiss() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'meteoswiss.csv')


def get_data_south_korea() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'south_korea.csv')


def get_data_washingtondc() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / 'washingtondc.csv')
