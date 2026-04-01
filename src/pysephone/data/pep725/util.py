from pathlib import Path

import pandas as pd


# Metadata CSV files are bundled with the package
_METADATA_DIR = Path(__file__).parent / 'metadata'


def read_df_species() -> pd.DataFrame:
    return pd.read_csv(_METADATA_DIR / 'species.csv', sep=';', index_col='species_code')


def read_df_countries() -> pd.DataFrame:
    return pd.read_csv(_METADATA_DIR / 'countries.csv', sep=';', index_col='country_code')


def read_df_species_subgroups() -> pd.DataFrame:
    return pd.read_csv(_METADATA_DIR / 'species_subgroups.csv', sep=';', index_col=('species_code', 'subgroup_code'))


def read_df_species_entries() -> pd.DataFrame:
    return pd.read_csv(_METADATA_DIR / 'species_entries.csv', sep=';')
