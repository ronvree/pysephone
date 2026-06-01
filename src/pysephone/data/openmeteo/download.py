import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests_cache
import tables  # Required by pandas HDFStore with PyTables backend
from retry_requests import retry
from tqdm import tqdm

from pysephone.paths import get_data_root, get_products_data_dir


"""
    Code for downloading and accessing OpenMeteo data.

    DESCRIPTION:

    OpenMeteo data can be requested per entry.

    Data entries are characterized by:
        - Data source key (src_key)
        - Location key as defined by the data source (loc_id)
        - Year

    Each request obtains temperature/climate data from OpenMeteo for that year at
    the specified location. Location coordinates are provided when creating entries.

    Given these entries, checks are made whether data is already present or should
    be downloaded. Data is stored in per-variable HDF5 stores under:

        <data_root>/data/products/openmeteo/

"""

_STORE_NAME_TEMPLATE = "openmeteo_data_store_{step}_{data_key}.h5"

# HDF5 key template — entries are bucketed into groups to stay within PyTables limits
_STORE_KEY_TEMPLATE = "/openmeteo/{step}/{data_key}/group_{group_key}/src_{src_key}/loc_{loc_id}/year_{year}"

# API key filename
_API_KEY_FILENAME = 'openmeteo_api_key.txt'

# Delay between download requests
# _DOWNLOAD_DELAY = 0.01  # seconds
_DOWNLOAD_DELAY = 1  # seconds

# Data compression settings (DON'T CHANGE — would corrupt existing stores)
_STORE_COMP_LIB = 'zlib'
_STORE_COMP_LVL = 5
_STORE_FORMAT = 'table'


def _get_openmeteo_dir(root: Path) -> Path:
    return get_products_data_dir(root) / 'openmeteo'


@dataclass
class OpenMeteoEntry:
    """
    Represents a single OpenMeteo data entry.

    Entries are uniquely identified by (step, data_key, src_key, loc_id, year):
      - step:     temporal resolution (e.g. 'hourly', 'daily')
      - data_key: variable name as used by the OpenMeteo API
      - src_key:  phenology data source key (matches ObservationSource.KEY)
      - loc_id:   location identifier within that source
      - year:     calendar year to retrieve
    """

    step: str
    data_key: str
    src_key: str
    loc_id: int
    loc_name: str
    lat: float
    lon: float
    year: int

    def __hash__(self):
        return hash((self.step, self.data_key, self.src_key, self.loc_id, self.year))


class OpenMeteoStores:
    """
    Manages access to per-variable HDF5 stores for OpenMeteo data.

    Usage:
        with OpenMeteoStores(step_key_pairs, root) as stores:
            df = stores[entry]          # read
            stores[entry] = df          # write
            entry in stores             # check presence
    """

    def __init__(self, step_key_pairs: list, root: Path = None):
        if root is None:
            root = get_data_root()
        data_dir = _get_openmeteo_dir(root)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._stores = {
            (step, key): self._load_store(step, key) for step, key in step_key_pairs
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item) -> pd.DataFrame:
        key = self._get_store_key(self._to_tuple(item))
        store = self._get_store(self._to_tuple(item))
        result = store.get(key=key)
        assert isinstance(result, pd.DataFrame)
        return result

    def __setitem__(self, item, df: pd.DataFrame):
        key = self._get_store_key(self._to_tuple(item))
        store = self._get_store(self._to_tuple(item))
        df.to_hdf(store, key=key, format=_STORE_FORMAT,
                  complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL)

    def __contains__(self, item) -> bool:
        key = self._get_store_key(self._to_tuple(item))
        store = self._get_store(self._to_tuple(item))
        return key in store

    def close(self):
        for store in self._stores.values():
            store.close()

    def _get_store(self, item: tuple) -> pd.HDFStore:
        step, data_key, src_key, loc_id, year = item
        return self._stores[step, data_key]

    def _load_store(self, step: str, key: str) -> pd.HDFStore:
        path = self._data_dir / _STORE_NAME_TEMPLATE.format(step=step, data_key=key)
        return pd.HDFStore(str(path), complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL)

    @staticmethod
    def _to_tuple(item) -> tuple:
        if isinstance(item, OpenMeteoEntry):
            return item.step, item.data_key, item.src_key, item.loc_id, item.year
        return item

    @staticmethod
    def _get_store_key(item: tuple) -> str:
        step, data_key, src_key, loc_id, year = item
        group_key = OpenMeteoStores._assign_group_key(item)
        return _STORE_KEY_TEMPLATE.format(
            step=step,
            data_key=data_key,
            group_key=group_key,
            src_key=src_key,
            loc_id=loc_id,
            year=year,
        )

    @staticmethod
    def _assign_group_key(item: tuple) -> str:
        """
        Distribute entries across ≤4096 groups (PyTables recommends group sizes <16000).
        Uses a hash of (src_key, loc_id, year) for deterministic, run-independent assignment.
        """
        step, data_key, src_key, loc_id, year = item
        s = f'{src_key}-{loc_id}-{year}'
        h = hashlib.sha1(bytearray(s, encoding='utf-8'), usedforsecurity=False).hexdigest()
        return h[:3]

    def _transfer_store(self, name: str = 'openmeteo_data_store.h5'):
        """Migrate data from a legacy single-file store into the per-variable stores."""
        path = self._data_dir / name
        with pd.HDFStore(str(path), complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL) as store:
            for key in tqdm(store.keys()):
                tokens = key.split('/')
                step = tokens[2]
                data_key = tokens[3]
                df = store.get(key=key)
                assert isinstance(df, pd.DataFrame)
                if (step, data_key) not in self._stores:
                    print('Store is not completely transferred!', (step, data_key))
                    continue
                df.to_hdf(self._stores[step, data_key], key=key, format=_STORE_FORMAT,
                          complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL)


DOWNLOAD_MODES = [
    None,       # Default — download all missing data
    'forced',   # Download all data regardless of whether it is already present
    'skip',     # Skip downloading data
]


def get_openmeteo_data(
    entries: set,
    verbose: bool = True,
    download_mode: str = None,
    root: Path = None,
) -> dict:
    """
    Download and cache OpenMeteo data for the given entries.

    Args:
        entries:       Set of OpenMeteoEntry objects to fetch.
        verbose:       Show progress bars.
        download_mode: None (default, download missing), 'forced', or 'skip'.
        root:          Data root directory. Defaults to get_data_root().

    Returns:
        Empty dict (reserved for future result info).
    """
    assert download_mode in DOWNLOAD_MODES, f'Unrecognized download mode: {download_mode}'

    if root is None:
        root = get_data_root()

    if not entries:
        return {}

    step_key_pairs = list({(e.step, e.data_key) for e in entries})

    with OpenMeteoStores(step_key_pairs, root=root) as stores:
        match download_mode:
            case None:
                missing = _check_entries_missing(entries, stores, verbose=verbose)
                _download_entries(missing, stores, root=root, verbose=verbose)
            case 'forced':
                _download_entries(entries, stores, root=root, verbose=verbose)
            case 'skip':
                pass
            case _:
                raise OpenMeteoDownloadException(f'Unrecognized download mode: {download_mode}')

    return {}


def _check_entries_missing(entries: set, stores: OpenMeteoStores, verbose: bool = True) -> set:
    iterable = tqdm(entries, desc='Checking for missing meteo data') if verbose else entries
    missing = set()
    for entry in iterable:
        if entry not in stores:
            missing.add(entry)
        if verbose:
            iterable.set_postfix({
                'step': entry.step,
                'data_key': entry.data_key,
                'location_id': f'({entry.src_key}, {entry.loc_id})',
                'year': entry.year,
                'n_missing': f'{len(missing)}/{len(entries)}',
            })
    return missing


def _require_openmeteo():
    """Import openmeteo-requests lazily. Only needed to download new OpenMeteo
    data, not to read data already cached on disk."""
    try:
        import openmeteo_requests  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openmeteo-requests is required to download OpenMeteo data. "
            'Install with: pip install "pysephone[openmeteo]"'
        ) from exc
    return openmeteo_requests


def _download_entries(
    entries: set,
    stores: OpenMeteoStores,
    root: Path,
    verbose: bool = True,
) -> None:
    """Download the given entries from the OpenMeteo API and write them to stores."""
    if not entries:
        return

    openmeteo_requests = _require_openmeteo()
    from openmeteo_requests.Client import OpenMeteoRequestsError

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    api_key = _get_api_key(root)
    commercial = api_key is not None

    if api_key is None:
        print('No OpenMeteo API key found. Proceeding with the non-commercial API limitations')
    else:
        print('OpenMeteo API key found. Using commercial API')

    url = _get_base_url(commercial=commercial)

    iterable = tqdm(entries, desc='Downloading openmeteo data') if verbose else entries
    num_failed = 0

    for entry in iterable:
        params = {
            'latitude': entry.lat,
            'longitude': entry.lon,
            'start_date': f'{entry.year}-01-01',
            'end_date': f'{entry.year}-12-31',
            'models': 'era5',
        }
        if entry.step == 'hourly':
            params['hourly'] = entry.data_key
        elif entry.step == 'daily':
            params['daily'] = entry.data_key
        if commercial:
            params['apikey'] = api_key

        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            match entry.step:
                case 'hourly':
                    data = response.Hourly()
                case 'daily':
                    data = response.Daily()
                case _:
                    raise OpenMeteoDownloadException(f'Unsupported step value: {entry.step!r}')

            df = pd.DataFrame({
                'date': pd.date_range(
                    start=pd.to_datetime(data.Time(), unit='s', utc=True),
                    end=pd.to_datetime(data.TimeEnd(), unit='s', utc=True),
                    freq=pd.Timedelta(seconds=data.Interval()),
                    inclusive='left',
                ),
                entry.data_key: data.Variables(0).ValuesAsNumpy(),
            })

            stores[entry] = df

        except OpenMeteoRequestsError as exc:
            num_failed += 1
            print(exc)

        if verbose:
            iterable.set_postfix({
                'step': entry.step,
                'data_key': entry.data_key,
                'location_id': f'({entry.src_key}, {entry.loc_id})',
                'year': entry.year,
                'n_failed': f'{num_failed}/{len(entries)}',
            })

        time.sleep(_DOWNLOAD_DELAY)


def _get_base_url(commercial: bool = False) -> str:
    prefix = 'customer-' if commercial else ''
    return f'https://{prefix}archive-api.open-meteo.com/v1/archive'


def _get_api_key(root: Path) -> str | None:
    path = _get_openmeteo_dir(root) / _API_KEY_FILENAME
    return path.read_text().strip() if path.exists() else None


class OpenMeteoDownloadException(Exception):
    pass
