from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pysephone.constants import KEY_OBSERVATIONS
from pysephone.data.source import ObservationData, ObservationSource
from pysephone.data.pep725.util import (
    read_df_species,
    read_df_countries,
    read_df_species_subgroups,
    read_df_species_entries,
)
from pysephone.data.pep725.download import (
    PEP725Entry,
    check_entries_missing,
    download_entries,
    extract_entries,
    create_observations_df,
    create_locations_df,
    create_events_df,
)
from pysephone.paths import get_observations_source_data_dir


class PEP725Source(ObservationSource):
    """
    Data source for the PEP725 Pan European Phenology dataset.

    PEP725 requires registration at http://pep725.eu to obtain credentials.
    Place your credentials in a file at:
        <data_root>/data/observations/pep725/credentials.txt
    formatted as: <email> <password>

    Supported cfg keys:
        force_download (bool): Re-download all entries even if already present. Default: False.
        verbose (bool):        Show progress bars during download/extraction. Default: True.
    """

    KEY = 'pep725'

    def _get_data(self, cfg: Mapping[str, Any], root: Path) -> ObservationData:
        force_download: bool = cfg.get('force_download', False)
        verbose: bool = cfg.get('verbose', True)

        # Load metadata from bundled CSVs
        df_species = read_df_species()
        df_countries = read_df_countries()
        df_species_subgroups = read_df_species_subgroups()
        df_entries = read_df_species_entries()

        # Build the full set of entries to obtain
        entries = {
            PEP725Entry(
                species_key=df_species.loc[species_code].key,
                species_code=species_code,
                subgroup_code=subgroup_code,
                country_code=country_code,
                species_name=df_species.loc[species_code].species_name,
                subgroup_name=df_species_subgroups.loc[species_code, subgroup_code].subgroup_name,
                country_name=df_countries.loc[country_code].country_name,
            )
            for _, species_code, subgroup_code, country_code in df_entries.itertuples()
        }

        credentials_path = get_observations_source_data_dir(root, self.KEY) / 'credentials.txt'

        # Determine which entries need downloading or extracting
        if force_download:
            entries_to_download = entries
            entries_to_extract = set()
        else:
            missing = check_entries_missing(entries, root, verbose=verbose)
            entries_to_download = set()
            entries_to_extract = set()
            for entry in missing['data']:
                if entry not in missing['download']:
                    entries_to_extract.add(entry)
                else:
                    entries_to_download.add(entry)

        # Download and extract as needed
        result_download = download_entries(entries_to_download, root, credentials_path, verbose=verbose)
        entries_to_extract |= result_download['successful']
        entries_failed = result_download['failed']

        extract_entries(entries_to_extract, root, verbose=verbose)

        included = entries - entries_failed

        return ObservationData(
            observations=self._build_observations(included, root),
            events=self._build_events(included, root),
            locations=self._build_locations(included, root),
            species=self._build_species(df_species),
            subgroups=self._build_subgroups(df_species_subgroups),
        )

    def _build_observations(self, entries: set, root: Path) -> pd.DataFrame:
        """
        Index:   src, loc_id, year, species_id, subgroup_id, obs_type
        Columns: date (datetime)
        """
        df = create_observations_df(entries, root).reset_index()
        df['src'] = self.KEY
        df[KEY_OBSERVATIONS] = (
            pd.to_datetime(df['year'].astype(str)) + pd.to_timedelta(df['DAY'] - 1, unit='D')
        )
        return df.set_index(['src', 'loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type'])[[KEY_OBSERVATIONS]]

    def _build_events(self, entries: set, root: Path) -> pd.DataFrame:
        """
        Index:   src, event (BBCH code)
        Columns: description
        """
        df = create_events_df(entries, root).reset_index()
        df['src'] = self.KEY
        df = df.rename(columns={'bbch': 'event'})
        return df.set_index(['src', 'event'])[['description']]

    def _build_locations(self, entries: set, root: Path) -> pd.DataFrame:
        """
        Index:   src, loc_id
        Columns: lat, lon, loc_name, country_code
        """
        df = create_locations_df(entries, root).reset_index()
        df['src'] = self.KEY
        df = df.rename(columns={'LAT': 'lat', 'LON': 'lon', 'NAME': 'loc_name'})
        return df.set_index(['src', 'loc_id'])[['lat', 'lon', 'loc_name', 'country_code']]

    def _build_species(self, df_species: pd.DataFrame) -> pd.DataFrame:
        """
        Index:   src, species_id (species_code)
        Columns: key, species_name, species
        """
        df = df_species.reset_index()
        df['src'] = self.KEY
        df = df.rename(columns={'species_code': 'species_id'})
        return df.set_index(['src', 'species_id'])[['key', 'species_name', 'species']]

    def _build_subgroups(self, df_species_subgroups: pd.DataFrame) -> pd.DataFrame:
        """
        Index:   src, subgroup_id ("{species_code}_{subgroup_code}")
        Columns: species_id, subgroup_name, is_cultivar
        """
        df = df_species_subgroups.reset_index()
        df['src'] = self.KEY
        df['subgroup_id'] = df['subgroup_code']
        df = df.rename(columns={'species_code': 'species_id'})
        return df.set_index(['src', 'subgroup_id'])[['species_id', 'subgroup_name', 'is_cultivar']]
