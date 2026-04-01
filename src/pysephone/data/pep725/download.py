import time
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from pysephone.paths import get_observations_source_data_dir


_LOGIN_URL = 'http://pep725.eu/login.php'
_URL_TEMPLATE = 'http://pep725.eu/data_download/download.php?id={country_code}_{species_code:03d}_{subgroup_code:03d}'
_DATA_FOLDER_TEMPLATE = 'PEP725_{country_code}_{species_code:03d}_{subgroup_code:03d}'
_FN_TEMPLATE_DOWNLOAD = 'PEP725_{country_code}_{species_code:03d}_{subgroup_code:03d}.tar.gz'
_FN_BBCH = 'PEP725_BBCH.csv'
_FN_TEMPLATE_STATIONS = 'PEP725_{country_code}_stations.csv'

# Delay between download requests to avoid overloading the PEP725 server
_DOWNLOAD_DELAY = 0.01


@dataclass(frozen=True)
class PEP725Entry:
    """
    Represents a single downloadable PEP725 data entry, uniquely identified
    by species code, subgroup code, and country code.
    """
    species_key: str
    species_code: int
    subgroup_code: int
    country_code: str
    species_name: str
    subgroup_name: str
    country_name: str

    @property
    def download_url(self) -> str:
        return _URL_TEMPLATE.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )

    def path_download_file(self, root: Path) -> Path:
        fn = _FN_TEMPLATE_DOWNLOAD.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )
        return get_observations_source_data_dir(root, 'pep725') / 'downloads' / fn

    def path_data_folder(self, root: Path) -> Path:
        fn = _DATA_FOLDER_TEMPLATE.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )
        return get_observations_source_data_dir(root, 'pep725') / 'data' / self.species_key / fn

    def get_stations_df(self, root: Path) -> pd.DataFrame:
        """
        Load station metadata for the country of this entry.

        Returns a DataFrame indexed by PEP_ID with columns:
        National_ID, LON, LAT, ALT, NAME, country_code.
        """
        path = self.path_data_folder(root) / _FN_TEMPLATE_STATIONS.format(country_code=self.country_code)
        df = pd.read_csv(path, sep=';', index_col='PEP_ID')
        df['country_code'] = self.country_code
        return df

    def get_data_df(self, root: Path, set_index: bool = True) -> pd.DataFrame:
        """
        Load phenological observations for this entry.

        Downloads contain multiple CSVs; the observations file is identified by
        its name matching PEP725_{CC}_*.csv (excluding the stations file).

        Returns a DataFrame with columns PEP_ID, YEAR, BBCH, DAY.
        If set_index is True, indexed by (PEP_ID, YEAR).
        """
        cc = self.country_code
        files = [
            p for p in self.path_data_folder(root).iterdir()
            if p.name.startswith(f'PEP725_{cc}_') and p.name != f'PEP725_{cc}_stations.csv'
        ]
        assert len(files) == 1, f'Expected one observations file, found: {files}'
        df = pd.read_csv(files[0], sep=';')
        if set_index:
            df.set_index(['PEP_ID', 'YEAR'], inplace=True)
        return df

    def get_bbch_df(self, root: Path) -> pd.DataFrame:
        """
        Load BBCH phenological stage definitions for this entry.

        Returns a DataFrame indexed by BBCH code with a description column.
        """
        df = pd.read_csv(self.path_data_folder(root) / _FN_BBCH, sep=';')
        df.set_index('bbch', inplace=True)
        # BBCH 50 definition is missing from the source files
        df.loc[50] = 'Flower buds present, still enclosed by leaves (oilseed rape)'
        df.sort_index(inplace=True)
        return df


def check_entries_missing(entries: set, root: Path, verbose: bool = True) -> dict:
    """
    Check which entries are missing their data folder or raw download file.

    Returns a dict with keys 'data' and 'download', each a list of entries.
    """
    missing_data = []
    missing_download = []

    iterable = tqdm(entries, desc='Checking for missing PEP725 data') if verbose else entries

    for entry in iterable:
        if not entry.path_data_folder(root).exists():
            missing_data.append(entry)
        if not entry.path_download_file(root).exists():
            missing_download.append(entry)

    return {'data': missing_data, 'download': missing_download}


def download_entries(entries: set, root: Path, credentials_path: Path, verbose: bool = True) -> dict:
    """
    Download raw .tar.gz files for the given entries using PEP725 credentials.

    Returns a dict with keys 'successful' and 'failed' (sets of PEP725Entry).
    """
    successful = set()
    failed = set()

    if not entries:
        return {'successful': successful, 'failed': failed}

    username, password = _read_credentials(credentials_path)
    session = requests.Session()
    session.post(_LOGIN_URL, data={'email': username, 'pwd': password, 'submit': 'Login'})

    iterable = tqdm(entries, desc='Downloading PEP725 data') if verbose else entries
    num_failed = 0

    for entry in iterable:
        response = session.get(entry.download_url, cookies=session.cookies.get_dict())
        if response.headers['Content-Type'] == 'application/x-gzip':
            path = entry.path_download_file(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            successful.add(entry)
        else:
            num_failed += 1
            failed.add(entry)

        if verbose:
            iterable.set_postfix({'n_failed': f'{num_failed}/{len(entries)}'})

        time.sleep(_DOWNLOAD_DELAY)

    return {'successful': successful, 'failed': failed}


def extract_entries(entries: set, root: Path, verbose: bool = True) -> None:
    """
    Extract downloaded .tar.gz files into their respective data folders.
    Also sanitizes station CSVs after extraction.
    """
    if not entries:
        return

    iterable = tqdm(entries, desc='Extracting PEP725 data') if verbose else entries

    for entry in iterable:
        data_folder = entry.path_data_folder(root)
        data_folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(entry.path_download_file(root), 'r:gz') as f:
            f.extractall(path=data_folder)

        stations_path = data_folder / _FN_TEMPLATE_STATIONS.format(country_code=entry.country_code)
        _sanitize_stations_csv(stations_path)


def create_observations_df(entries: set, root: Path) -> pd.DataFrame:
    """
    Concatenate observation DataFrames from all entries into a single DataFrame.

    Indexed by (loc_id, year, species_id, subgroup_id, obs_type).
    """
    dfs = []
    for entry in entries:
        df = entry.get_data_df(root, set_index=False)
        df['species_id'] = entry.species_code
        df['subgroup_id'] = entry.subgroup_code
        df = df.rename(columns={'PEP_ID': 'loc_id', 'YEAR': 'year', 'BBCH': 'obs_type'})
        df['obs_type'] = 'BBCH_' + df['obs_type'].astype(str)
        df.set_index(['loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type'], inplace=True)
        dfs.append(df)
    return pd.concat(dfs)


def create_locations_df(entries: set, root: Path) -> pd.DataFrame:
    """
    Concatenate station DataFrames, picking one entry per country.

    Indexed by loc_id with columns LON, LAT, ALT, NAME, country_code.
    """
    picked = {entry.country_code: entry for entry in entries}
    df = pd.concat([e.get_stations_df(root) for e in picked.values()])
    df.index.name = 'loc_id'
    return df


def create_events_df(entries: set, root: Path) -> pd.DataFrame:
    """
    Load BBCH event definitions from the first available entry.

    All entries share the same BBCH definitions, so one is sufficient.
    """
    entry = next(iter(entries))
    return entry.get_bbch_df(root)


def _read_credentials(path: Path) -> tuple[str, str]:
    assert path.exists(), f'PEP725 credentials file not found: {path}'
    tokens = path.read_text().split(' ')
    return tokens[0], ' '.join(tokens[1:])


def _sanitize_stations_csv(path: Path) -> None:
    """
    Some PEP725 station names contain semicolons, breaking semicolon-delimited CSVs.
    Ensures each row has exactly 6 semicolons by replacing excess ones with spaces.
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    sanitized = []
    for line in lines:
        parts = line.split(';')
        if len(parts) > 6:
            sanitized.append(';'.join(parts[:6]) + ' ' + ' '.join(parts[6:]))
        else:
            sanitized.append(line)
    path.write_text('\n'.join(sanitized), encoding='utf-8')
