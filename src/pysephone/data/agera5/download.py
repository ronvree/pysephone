import gc
import hashlib
import math
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tables  # Required by pandas HDFStore with PyTables backend
from tqdm import tqdm

from pysephone.paths import get_data_root, get_products_data_dir


# Default geographic tile size (degrees) for chunking CDS requests.
# 3.0° at 365 days × 1 variable lands well under the 400-cost-unit per-request
# limit for sis-agrometeorological-indicators.  Shrink via the --tile-deg flag
# if a region's cost still exceeds the limit.
DEFAULT_TILE_DEG: float = 3.0


"""
    Code for downloading and accessing AgERA5 data from the Copernicus
    Climate Data Store (CDS).

    DESCRIPTION:

    AgERA5 is a daily, agriculture-oriented derivative of ERA5 published on
    CDS as dataset 'sis-agrometeorological-indicators'.  It is 0.1° resolved
    (downscaled from ERA5's 0.25° with a lapse-rate elevation correction),
    aggregated in local solar time, and ships derived fields (Penman-Monteith
    inputs, RH at fixed local hours, etc.).

    Unlike OpenMeteo, CDS is queue-based — per-(loc, year) requests are
    impractical.  This module bundles entries by (variable, src_key, year)
    and submits one CDS request per bundle covering the bounding box of all
    locations in that bundle.  The returned NetCDFs are sampled at each
    location and persisted in per-variable HDF5 stores under:

        <data_root>/data/products/agera5/

    Authentication uses cdsapi's own convention (~/.cdsapirc or the
    CDSAPI_URL / CDSAPI_KEY environment variables) — no API key handling
    is done here.
"""

_DATASET_NAME = 'sis-agrometeorological-indicators'
# v2.0 (released 2025-05-20) fixes rogue Tmin-24h values and the Tmean>Tmax
# issue present in v1.1. v1.1 stops being updated on 2026-06-17.
# See https://forum.ecmwf.int/t/agera5-version-1-1-to-be-deprecated-please-use-agera5-version-2-if-not-already/14940
_DATASET_VERSION = '2_0'

_STORE_NAME_TEMPLATE = "agera5_data_store_{data_key}.h5"

# HDF5 key template — entries are bucketed into groups to stay within PyTables limits
_STORE_KEY_TEMPLATE = "/agera5/daily/{data_key}/group_{group_key}/src_{src_key}/loc_{loc_id}/year_{year}"

# Data compression settings (DON'T CHANGE — would corrupt existing stores)
_STORE_COMP_LIB = 'zlib'
_STORE_COMP_LVL = 5
_STORE_FORMAT = 'table'


# Mapping from user-facing data_key (= variable name in the resulting AgERA5
# NetCDF file) to the CDS request parameters that produce it.
# Add entries here as you need more variables.
_VARIABLE_SPECS: dict[str, dict] = {
    'Temperature_Air_2m_Mean_24h':
        {'variable': '2m_temperature', 'statistic': '24_hour_mean'},
    'Temperature_Air_2m_Max_24h':
        {'variable': '2m_temperature', 'statistic': '24_hour_maximum'},
    'Temperature_Air_2m_Min_24h':
        {'variable': '2m_temperature', 'statistic': '24_hour_minimum'},
    'Precipitation_Flux':
        {'variable': 'precipitation_flux'},
    'Solar_Radiation_Flux':
        {'variable': 'solar_radiation_flux'},
    'Vapour_Pressure_Mean':
        {'variable': 'vapour_pressure', 'statistic': '24_hour_mean'},
    'Wind_Speed_10m_Mean':
        {'variable': '10m_wind_speed', 'statistic': '24_hour_mean'},
    'Dew_Point_Temperature_2m_Mean':
        {'variable': '2m_dewpoint_temperature', 'statistic': '24_hour_mean'},
}


def _get_agera5_dir(root: Path) -> Path:
    return get_products_data_dir(root) / 'agera5'


@dataclass
class AgEra5Entry:
    """
    Represents a single AgERA5 data entry.

    Entries are uniquely identified by (data_key, src_key, loc_id, year):
      - data_key: variable name as it appears inside the AgERA5 NetCDF
                  (e.g. 'Temperature_Air_2m_Mean_24h').  Must be a key of
                  :data:`_VARIABLE_SPECS`.
      - src_key:  phenology data source key (matches ObservationSource.KEY)
      - loc_id:   location identifier within that source
      - year:     calendar year to retrieve

    ``loc_name`` is purely informational (not used in CDS requests or hashing)
    and may be ``None`` for sources that don't provide one.
    """

    data_key: str
    src_key: str
    loc_id: int
    lat: float
    lon: float
    year: int
    loc_name: str | None = None

    def __hash__(self):
        return hash((self.data_key, self.src_key, self.loc_id, self.year))


class AgEra5Stores:
    """
    Manages access to per-variable HDF5 stores for AgERA5 data.

    Usage:
        with AgEra5Stores(data_keys, root) as stores:
            df = stores[entry]          # read
            stores[entry] = df          # write
            entry in stores             # check presence
    """

    def __init__(self, data_keys: list, root: Path = None):
        if root is None:
            root = get_data_root()
        data_dir = _get_agera5_dir(root)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._stores = {key: self._load_store(key) for key in data_keys}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item) -> pd.DataFrame:
        tup = self._to_tuple(item)
        key = self._get_store_key(tup)
        store = self._get_store(tup)
        result = store.get(key=key)
        assert isinstance(result, pd.DataFrame)
        return result

    def __setitem__(self, item, df: pd.DataFrame):
        tup = self._to_tuple(item)
        key = self._get_store_key(tup)
        store = self._get_store(tup)
        df.to_hdf(store, key=key, format=_STORE_FORMAT,
                  complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL)

    def __contains__(self, item) -> bool:
        tup = self._to_tuple(item)
        key = self._get_store_key(tup)
        store = self._get_store(tup)
        return key in store

    def close(self):
        for store in self._stores.values():
            store.close()

    def flush(self) -> None:
        """Force pending HDFStore writes to disk.

        Without this, pandas/PyTables buffers writes until ``close()``.  Long
        resumable downloads (CDS minutes per tile) call this after each tile
        so a kernel restart only loses the currently in-flight bundle, not
        everything written this run.
        """
        for store in self._stores.values():
            store.flush(fsync=True)

    def _get_store(self, item: tuple) -> pd.HDFStore:
        data_key, src_key, loc_id, year = item
        return self._stores[data_key]

    def _load_store(self, key: str) -> pd.HDFStore:
        path = self._data_dir / _STORE_NAME_TEMPLATE.format(data_key=key)
        return pd.HDFStore(str(path), complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL)

    @staticmethod
    def _to_tuple(item) -> tuple:
        if isinstance(item, AgEra5Entry):
            return item.data_key, item.src_key, item.loc_id, item.year
        return item

    @staticmethod
    def _get_store_key(item: tuple) -> str:
        data_key, src_key, loc_id, year = item
        group_key = AgEra5Stores._assign_group_key(item)
        return _STORE_KEY_TEMPLATE.format(
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
        data_key, src_key, loc_id, year = item
        s = f'{src_key}-{loc_id}-{year}'
        h = hashlib.sha1(bytearray(s, encoding='utf-8'), usedforsecurity=False).hexdigest()
        return h[:3]


DOWNLOAD_MODES = [
    None,       # Default — download all missing data
    'forced',   # Download all data regardless of whether it is already present
    'skip',     # Skip downloading data
]


def build_entries(observations, calendar, data_keys) -> set:
    """Build the full set of AgEra5Entry objects required for the given observations.

    For each observation, the calendar defines a season window which may span
    multiple calendar years (e.g. a winter season starting October Y-1 and
    ending September Y); one entry is created per (data_key, src_key, loc_id,
    year) tuple covering that window.

    This is the same set the provider's :meth:`AgEra5Features.download` would
    build internally — factoring it out lets callers merge entry sets across
    multiple `Observations` instances (e.g. all BloomBench datasets), so
    cross-source tile deduplication happens at the bundling step.
    """
    entries: set = set()
    for key in data_keys:
        for index in observations.iter_index():
            src, loc_id, year, species_id, subgroup_id = index

            season = calendar.get_season_info(
                year=year,
                species_id=species_id,
                subgroup_id=subgroup_id,
                src=src,
                loc_id=loc_id,
            )
            season_start = season['season_start']
            season_end = season['season_end']

            coords = observations.get_location_coordinates((src, loc_id))

            EPOCH_YEAR = 1970
            year_min = int(season_start.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR
            year_max = int(season_end.astype('datetime64[Y]').astype(int)) + EPOCH_YEAR

            try:
                loc_name = observations.get_location_name((src, loc_id))
            except KeyError:
                # Source doesn't expose a loc_name column — fine, it's optional.
                loc_name = None

            for year_ in range(year_min, year_max + 1):
                entries.add(AgEra5Entry(
                    data_key=key,
                    src_key=src,
                    loc_id=loc_id,
                    loc_name=loc_name,
                    lat=coords['lat'],
                    lon=coords['lon'],
                    year=year_,
                ))
    return entries


def estimate_requests(
    entries: set,
    root: Path = None,
    tile_deg: float = DEFAULT_TILE_DEG,
) -> dict:
    """Estimate how much work :func:`get_agera5_data` would do for these entries.

    Touches the on-disk HDF5 store to check which entries are already cached,
    but issues no CDS calls.  Useful for pre-flight inspection: how many CDS
    requests will the next download trigger?

    Returns:
        Dict with keys ``total_entries``, ``missing_entries``, ``num_bundles``.
    """
    if root is None:
        root = get_data_root()
    if not entries:
        return {'total_entries': 0, 'missing_entries': 0, 'num_bundles': 0}

    data_keys = list({e.data_key for e in entries})
    with AgEra5Stores(data_keys, root=root) as stores:
        missing = _check_entries_missing(entries, stores, verbose=False)
    bundles = _bundle_entries(missing, tile_deg=tile_deg)
    return {
        'total_entries': len(entries),
        'missing_entries': len(missing),
        'num_bundles': len(bundles),
    }


def get_agera5_data(
    entries: set,
    verbose: bool = True,
    download_mode: str = None,
    root: Path = None,
    tile_deg: float = DEFAULT_TILE_DEG,
) -> dict:
    """
    Download and cache AgERA5 data for the given entries.

    Args:
        entries:       Set of AgEra5Entry objects to fetch.
        verbose:       Show progress bars.
        download_mode: None (default, download missing), 'forced', or 'skip'.
        root:          Data root directory. Defaults to get_data_root().
        tile_deg:      Geographic tile size in degrees for chunking CDS
                       requests.  Smaller tiles → more requests but lower
                       cost per request.  See DEFAULT_TILE_DEG.

    Returns:
        Empty dict (reserved for future result info).
    """
    assert download_mode in DOWNLOAD_MODES, f'Unrecognized download mode: {download_mode}'

    if root is None:
        root = get_data_root()

    if not entries:
        return {}

    data_keys = list({e.data_key for e in entries})

    with AgEra5Stores(data_keys, root=root) as stores:
        match download_mode:
            case None:
                missing = _check_entries_missing(entries, stores, verbose=verbose)
                _download_entries(missing, stores, verbose=verbose, tile_deg=tile_deg)
            case 'forced':
                _download_entries(entries, stores, verbose=verbose, tile_deg=tile_deg)
            case 'skip':
                pass
            case _:
                raise AgEra5DownloadException(f'Unrecognized download mode: {download_mode}')

    return {}


def _check_entries_missing(entries: set, stores: AgEra5Stores, verbose: bool = True) -> set:
    iterable = tqdm(entries, desc='Checking for missing AgERA5 data') if verbose else entries
    missing = set()
    for entry in iterable:
        if entry not in stores:
            missing.add(entry)
        if verbose:
            iterable.set_postfix({
                'data_key': entry.data_key,
                'location_id': f'({entry.src_key}, {entry.loc_id})',
                'year': entry.year,
                'n_missing': f'{len(missing)}/{len(entries)}',
            })
    return missing


def _tile_id(lat: float, lon: float, tile_deg: float) -> tuple[int, int]:
    """Discretize a (lat, lon) into an integer tile bin."""
    return (math.floor(lat / tile_deg), math.floor(lon / tile_deg))


def _bundle_entries(entries: set, tile_deg: float = DEFAULT_TILE_DEG) -> dict:
    """Group entries by (data_key, year, tile_id) — one CDS request per bundle.

    Bundling across src_keys is intentional: PEP725_Apple and PEP725_Pear
    that share station coordinates produce a single CDS download that fans
    out to both source's HDF5 records.  Bundling by tile keeps each request
    small enough to stay under CDS's per-request cost limit.
    """
    bundles: dict = defaultdict(list)
    for entry in entries:
        tile = _tile_id(entry.lat, entry.lon, tile_deg)
        bundles[(entry.data_key, entry.year, tile)].append(entry)
    return bundles


def _bbox_for_bundle(bundle: list, pad_deg: float = 0.2) -> tuple[float, float, float, float]:
    """Return [north, west, south, east] covering all locations, padded slightly."""
    lats = [e.lat for e in bundle]
    lons = [e.lon for e in bundle]
    north = min(90.0, max(lats) + pad_deg)
    south = max(-90.0, min(lats) - pad_deg)
    east = min(180.0, max(lons) + pad_deg)
    west = max(-180.0, min(lons) - pad_deg)
    return north, west, south, east


def _download_entries(
    entries: set,
    stores: AgEra5Stores,
    verbose: bool = True,
    tile_deg: float = DEFAULT_TILE_DEG,
) -> None:
    """Download the given entries from CDS, one request per (data_key, year, tile) bundle."""
    if not entries:
        return

    cdsapi = _require_cdsapi()
    xr = _require_xarray()

    client = cdsapi.Client(quiet=not verbose, progress=verbose)

    bundles = _bundle_entries(entries, tile_deg=tile_deg)
    iterable = tqdm(bundles.items(), desc='Downloading AgERA5 bundles') if verbose else bundles.items()
    num_failed = 0

    for (data_key, year, tile), bundle in iterable:
        if data_key not in _VARIABLE_SPECS:
            num_failed += len(bundle)
            print(
                f"Unknown AgERA5 variable {data_key!r}. "
                f"Add it to _VARIABLE_SPECS in agera5/download.py."
            )
            continue

        spec = _VARIABLE_SPECS[data_key]
        bbox = _bbox_for_bundle(bundle)

        try:
            _download_and_store_bundle(
                client=client, xr=xr, stores=stores,
                bundle=bundle, data_key=data_key, year=year,
                spec=spec, bbox=bbox,
            )
            # Persist this bundle's writes immediately — a kernel restart now
            # only loses the *next* in-flight tile, not everything in this run.
            stores.flush()
        except Exception as exc:
            num_failed += len(bundle)
            print(f'CDS request failed for ({data_key}, year={year}, tile={tile}): {exc}')

        if verbose:
            iterable.set_postfix({
                'data_key': data_key,
                'year': year,
                'tile': tile,
                'bundle_size': len(bundle),
                'n_failed': num_failed,
            })


def _download_and_store_bundle(
    *,
    client,
    xr,
    stores: AgEra5Stores,
    bundle: list,
    data_key: str,
    year: int,
    spec: dict,
    bbox: tuple[float, float, float, float],
) -> None:
    """Submit one CDS request, then sample each entry in the bundle from the result."""
    request = {
        'version': _DATASET_VERSION,
        'format': 'zip',
        'variable': spec['variable'],
        'year': str(year),
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'area': list(bbox),  # [N, W, S, E]
    }
    if 'statistic' in spec:
        request['statistic'] = spec['statistic']
    if 'time' in spec:
        request['time'] = spec['time']

    # ignore_cleanup_errors=True: on Windows the NetCDF backend can hang on to
    # file handles past xarray's .close(), so rmtree of the temp dir may raise
    # WinError 32.  Swallow that — the actual HDF5 writes already succeeded.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        zip_path = tmp_path / f'{data_key}_{year}.zip'
        client.retrieve(_DATASET_NAME, request, str(zip_path))

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        nc_files = sorted(tmp_path.glob('*.nc'))
        if not nc_files:
            raise AgEra5DownloadException(
                f'No NetCDF files extracted from CDS response for '
                f'({data_key}, year={year}).'
            )

        # Open each NetCDF individually, load eagerly, then combine in memory.
        # Avoids the dask dependency that xr.open_mfdataset requires, and
        # releases each file handle before opening the next — important on
        # Windows where lingering handles block temp-dir cleanup.
        # combine_attrs='override': AgERA5 daily NetCDFs each timestamp their
        # own `history` attribute, so attrs across files always differ.  We
        # keep the first file's attrs and skip the conflict check.
        loaded = []
        for nc in nc_files:
            with xr.open_dataset(nc) as ds_single:
                loaded.append(ds_single.load())
        ds = xr.combine_by_coords(loaded, combine_attrs='override')
        try:
            _persist_bundle_from_dataset(ds, stores, bundle, data_key)
        finally:
            ds.close()
        # Force release of any lingering refs to xarray-backed file handles
        # before TemporaryDirectory tries to rmtree.
        gc.collect()


def _persist_bundle_from_dataset(ds, stores: AgEra5Stores, bundle: list, data_key: str) -> None:
    """Sample each entry's point from the opened NetCDF dataset and write to the store."""
    lat_dim = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_dim = 'lon' if 'lon' in ds.coords else 'longitude'

    if data_key not in ds.data_vars:
        # Fall back to the first data variable if the NetCDF labels things differently.
        available = list(ds.data_vars)
        if len(available) != 1:
            raise AgEra5DownloadException(
                f'Expected variable {data_key!r} in NetCDF, found {available}.'
            )
        nc_var = available[0]
    else:
        nc_var = data_key

    lon_min = float(ds[lon_dim].min())
    lon_max = float(ds[lon_dim].max())
    # AgERA5 may use either -180–180 or 0–360 longitude.
    use_360 = lon_max > 180.0

    for entry in bundle:
        lon_q = entry.lon
        if use_360 and lon_q < 0:
            lon_q = lon_q + 360.0
        elif (not use_360) and lon_q > 180:
            lon_q = lon_q - 360.0

        point = ds[nc_var].sel(
            {lat_dim: entry.lat, lon_dim: lon_q},
            method='nearest',
        )
        values = point.values
        # AgERA5 time axis is daily; coerce to pandas datetimes
        dates = pd.to_datetime(point['time'].values, utc=True)

        df = pd.DataFrame({
            'date': dates,
            data_key: values,
        })
        stores[entry] = df


def _require_cdsapi():
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required to download AgERA5 data. "
            "Install with: pip install cdsapi"
        ) from exc
    return cdsapi


def _require_xarray():
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xarray (with netCDF4) is required to read AgERA5 NetCDF files. "
            "Install with: pip install xarray netCDF4"
        ) from exc
    return xr


class AgEra5DownloadException(Exception):
    pass
