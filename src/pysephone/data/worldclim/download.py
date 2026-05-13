"""
WorldClim v2.1 BIOCLIM raster download, extraction, and point sampling.

WorldClim ships 19 bioclimatic variables (BIO1..BIO19) as global GeoTIFF rasters
in WGS84 at several spatial resolutions (30s ≈ 1km, 2.5m, 5m, 10m).  This module:

- Downloads the resolution-specific archive from ``geodata.ucdavis.edu`` (cached
  on disk; never re-fetched if the extracted rasters are already present).
- Extracts each BIOCLIM TIFF into a per-resolution directory.
- Samples values at arbitrary ``(lat, lon)`` points using :mod:`rasterio`.
- Caches sampled values per ``(lat, lon, resolution)`` in a JSON file so that
  repeated lookups never need to re-open the rasters.

``rasterio`` is imported lazily — the rest of pysephone does not require it.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm

from pysephone.paths import get_data_root, get_products_data_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLDCLIM_BASE_URL = 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base'

# Spatial resolutions WorldClim publishes for the BIOCLIM product.
# '30s' is the ~1km resolution requested most often for species/phenology work.
WORLDCLIM_RESOLUTIONS: Tuple[str, ...] = ('30s', '2.5m', '5m', '10m')
DEFAULT_RESOLUTION: str = '30s'

NUM_BIOCLIM_VARS: int = 19
BIOCLIM_VAR_INDICES: Tuple[int, ...] = tuple(range(1, NUM_BIOCLIM_VARS + 1))

# Stable feature keys (the canonical "bio_<n>" form) and human-readable labels.
# WorldClim numbers them 1..19; we expose both forms so callers can pick either
# the compact key or the descriptive name.
BIOCLIM_VAR_NAMES: Dict[int, str] = {
    1: 'annual_mean_temperature',
    2: 'mean_diurnal_range',
    3: 'isothermality',
    4: 'temperature_seasonality',
    5: 'max_temperature_warmest_month',
    6: 'min_temperature_coldest_month',
    7: 'temperature_annual_range',
    8: 'mean_temperature_wettest_quarter',
    9: 'mean_temperature_driest_quarter',
    10: 'mean_temperature_warmest_quarter',
    11: 'mean_temperature_coldest_quarter',
    12: 'annual_precipitation',
    13: 'precipitation_wettest_month',
    14: 'precipitation_driest_month',
    15: 'precipitation_seasonality',
    16: 'precipitation_wettest_quarter',
    17: 'precipitation_driest_quarter',
    18: 'precipitation_warmest_quarter',
    19: 'precipitation_coldest_quarter',
}

BIOCLIM_KEYS: Tuple[str, ...] = tuple(f'bio_{i}' for i in BIOCLIM_VAR_INDICES)


DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2.0
DEFAULT_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def worldclim_data_dir(root: Optional[Path] = None) -> Path:
    """Root directory for WorldClim products under the data tree."""
    root = root if root is not None else get_data_root()
    return get_products_data_dir(root) / 'worldclim'


def bioclim_resolution_dir(resolution: str, root: Optional[Path] = None) -> Path:
    """Directory holding the extracted BIOCLIM rasters for *resolution*."""
    _validate_resolution(resolution)
    return worldclim_data_dir(root) / f'wc2.1_{resolution}_bio'


def bioclim_archive_path(resolution: str, root: Optional[Path] = None) -> Path:
    """Path to the downloaded BIOCLIM zip for *resolution*."""
    _validate_resolution(resolution)
    return worldclim_data_dir(root) / f'wc2.1_{resolution}_bio.zip'


def bioclim_raster_path(
    resolution: str, var: int, root: Optional[Path] = None
) -> Path:
    """Path to a single BIOCLIM raster (BIO*var*) for *resolution*."""
    _validate_resolution(resolution)
    _validate_var(var)
    return bioclim_resolution_dir(resolution, root) / f'wc2.1_{resolution}_bio_{var}.tif'


def default_cache_path(resolution: str, root: Optional[Path] = None) -> Path:
    """JSON cache for sampled values at *resolution*."""
    _validate_resolution(resolution)
    return worldclim_data_dir(root) / f'bioclim_samples_{resolution}.json'


# ---------------------------------------------------------------------------
# Disk-backed sample cache
# ---------------------------------------------------------------------------

class BioclimCache:
    """JSON-backed ``(lat, lon) -> {bio_1, ..., bio_19}`` cache.

    Both the on-disk and in-memory keys round the coordinate to *precision*
    decimals so small floating-point noise does not cause cache misses.
    Each value is a list of length :data:`NUM_BIOCLIM_VARS` ordered by BIO index
    (1..19); ``NaN`` is used for any variable not yet sampled.

    Note:
        One cache file is dedicated to a single spatial resolution because the
        ``(lat, lon)`` snap and the underlying raster values both change with
        resolution.  Mixing them in one file would silently corrupt lookups.
    """

    def __init__(
        self,
        resolution: str = DEFAULT_RESOLUTION,
        path: Optional[Path] = None,
        precision: int = 4,
        root: Optional[Path] = None,
    ) -> None:
        _validate_resolution(resolution)
        self._resolution = resolution
        self._path = Path(path) if path is not None else default_cache_path(resolution, root)
        self._precision = int(precision)
        self._cache: Dict[Tuple[float, float], List[float]] = {}
        self._load()

    # -- Public API --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def resolution(self) -> str:
        return self._resolution

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, coord: Tuple[float, float]) -> bool:
        return self._key(*coord) in self._cache

    def get(self, lat: float, lon: float) -> Optional[List[float]]:
        return self._cache.get(self._key(lat, lon))

    def put(self, lat: float, lon: float, values: Sequence[float]) -> None:
        if len(values) != NUM_BIOCLIM_VARS:
            raise ValueError(
                f'BioclimCache.put expected {NUM_BIOCLIM_VARS} values, got {len(values)}'
            )
        self._cache[self._key(lat, lon)] = [float(v) for v in values]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {f'{k[0]},{k[1]}': v for k, v in self._cache.items()}
        self._path.write_text(json.dumps(serialisable, indent=0), encoding='utf-8')

    # -- Internal ---------------------------------------------------------

    def _key(self, lat: float, lon: float) -> Tuple[float, float]:
        return round(float(lat), self._precision), round(float(lon), self._precision)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return
        for k, v in raw.items():
            lat_s, lon_s = k.split(',')
            if not isinstance(v, list) or len(v) != NUM_BIOCLIM_VARS:
                continue
            self._cache[(float(lat_s), float(lon_s))] = [float(x) for x in v]


# ---------------------------------------------------------------------------
# Archive download and extraction
# ---------------------------------------------------------------------------

def _archive_url(resolution: str) -> str:
    _validate_resolution(resolution)
    return f'{WORLDCLIM_BASE_URL}/wc2.1_{resolution}_bio.zip'


def _download_with_retries(
    url: str,
    dest: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    verbose: bool,
) -> None:
    """Stream *url* into *dest*, retrying on transient failures."""
    delay = backoff
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt == retries:
                        resp.raise_for_status()
                    wait = delay
                    retry_after = resp.headers.get('Retry-After')
                    if retry_after is not None:
                        try:
                            wait = max(wait, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    delay *= 2
                    continue
                resp.raise_for_status()

                total = int(resp.headers.get('Content-Length') or 0) or None
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + '.part')
                bar = (
                    tqdm(
                        total=total, unit='B', unit_scale=True,
                        desc=f'Downloading {dest.name}',
                    )
                    if verbose else None
                )
                try:
                    with tmp.open('wb') as f:
                        for chunk in resp.iter_content(chunk_size=DEFAULT_DOWNLOAD_CHUNK):
                            if not chunk:
                                continue
                            f.write(chunk)
                            if bar is not None:
                                bar.update(len(chunk))
                finally:
                    if bar is not None:
                        bar.close()
                tmp.replace(dest)
            return
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2

    raise RuntimeError('unreachable')


def _all_rasters_present(resolution: str, root: Optional[Path]) -> bool:
    return all(
        bioclim_raster_path(resolution, v, root).exists()
        for v in BIOCLIM_VAR_INDICES
    )


def download_bioclim_archive(
    resolution: str = DEFAULT_RESOLUTION,
    root: Optional[Path] = None,
    timeout: float = 120.0,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Fetch the BIOCLIM zip for *resolution* if not already cached.

    Returns the path to the on-disk zip.  Skips the download when the archive
    already exists and *force* is false.
    """
    archive = bioclim_archive_path(resolution, root)
    if archive.exists() and not force:
        return archive
    url = _archive_url(resolution)
    _download_with_retries(
        url, archive,
        timeout=timeout, retries=retries, backoff=backoff, verbose=verbose,
    )
    return archive


def extract_bioclim_archive(
    resolution: str = DEFAULT_RESOLUTION,
    root: Optional[Path] = None,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Extract the BIOCLIM zip into its per-resolution directory.

    Returns the directory containing the 19 ``wc2.1_<res>_bio_<n>.tif`` files.
    Skips extraction when every expected raster is already present and *force*
    is false.
    """
    out_dir = bioclim_resolution_dir(resolution, root)
    if _all_rasters_present(resolution, root) and not force:
        return out_dir

    archive = bioclim_archive_path(resolution, root)
    if not archive.exists():
        raise FileNotFoundError(
            f'BIOCLIM archive not found: {archive}. '
            'Call download_bioclim_archive() first.'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith('.tif')]
        it = tqdm(members, desc=f'Extracting {archive.name}', unit='file') if verbose else members
        for member in it:
            # Flatten any internal directory structure — write the basename
            # directly under out_dir so the bioclim_raster_path() layout holds.
            target = out_dir / Path(member).name
            if target.exists() and not force:
                continue
            with zf.open(member) as src, target.open('wb') as dst:
                while True:
                    chunk = src.read(DEFAULT_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
    return out_dir


def ensure_bioclim_rasters(
    resolution: str = DEFAULT_RESOLUTION,
    root: Optional[Path] = None,
    timeout: float = 120.0,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Ensure the 19 BIOCLIM rasters for *resolution* are on disk.

    Downloads the archive if missing, then extracts it.  Idempotent; returns
    the directory containing the rasters.
    """
    if _all_rasters_present(resolution, root) and not force:
        return bioclim_resolution_dir(resolution, root)
    download_bioclim_archive(
        resolution=resolution, root=root,
        timeout=timeout, verbose=verbose, force=force,
    )
    return extract_bioclim_archive(
        resolution=resolution, root=root, verbose=verbose, force=force,
    )


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def _import_rasterio():
    try:
        import rasterio  # noqa: WPS433 — optional, heavyweight dependency
    except ImportError as e:
        raise ImportError(
            "Sampling WorldClim rasters requires 'rasterio'. "
            "Install with: pip install rasterio"
        ) from e
    return rasterio


def sample_bioclim(
    coords: Sequence[Tuple[float, float]],
    resolution: str = DEFAULT_RESOLUTION,
    cache: Optional[BioclimCache] = None,
    root: Optional[Path] = None,
    verbose: bool = True,
    force: bool = False,
    variables: Optional[Sequence[int]] = None,
) -> Dict[Tuple[float, float], List[float]]:
    """Sample BIOCLIM values at a sequence of ``(lat, lon)`` points.

    Cached points are returned from disk; missing points are sampled from the
    extracted rasters and persisted to the cache after each raster pass so a
    crash partway through does not lose progress.

    Args:
        coords:     ``(lat, lon)`` points to sample.
        resolution: WorldClim resolution to read (default ``'30s'`` ≈ 1km).
        cache:      :class:`BioclimCache` to use (created at the default path
                    if ``None``).
        root:       Override the data root.
        verbose:    Show progress bars.
        force:      Re-sample even cached points.
        variables:  Subset of BIO indices (1..19) to sample.  ``None`` samples
                    all 19.  Note: the cache always stores 19 slots — unsampled
                    variables remain ``NaN``.

    Returns:
        ``{(lat_rounded, lon_rounded): [bio_1, ..., bio_19]}`` for every input
        point.  Slots not covered by *variables* (and not already cached) are
        ``NaN``.
    """
    rasterio = _import_rasterio()

    cache = cache if cache is not None else BioclimCache(resolution=resolution, root=root)
    if cache.resolution != resolution:
        raise ValueError(
            f'BioclimCache resolution mismatch: cache={cache.resolution!r}, '
            f'request={resolution!r}'
        )

    var_ids: List[int] = list(variables) if variables is not None else list(BIOCLIM_VAR_INDICES)
    for v in var_ids:
        _validate_var(v)

    prec = cache.precision

    # Decide which (lat, lon) need raster access this call.
    # A point is "fresh" if cached and not forced and already has every requested var.
    to_sample: List[Tuple[float, float]] = []
    out: Dict[Tuple[float, float], List[float]] = {}
    for lat, lon in coords:
        key = (round(float(lat), prec), round(float(lon), prec))
        cached = cache.get(*key)
        if cached is None or force:
            to_sample.append((lat, lon))
            out[key] = list(cached) if cached is not None else [float('nan')] * NUM_BIOCLIM_VARS
            continue
        # Cached: check whether every requested variable is filled in.
        needs = any(np.isnan(cached[v - 1]) for v in var_ids)
        if needs:
            to_sample.append((lat, lon))
            out[key] = list(cached)
        else:
            out[key] = list(cached)

    if not to_sample:
        return out

    # Ensure rasters are on disk before sampling.
    ensure_bioclim_rasters(resolution=resolution, root=root, verbose=verbose)

    # Sample one raster at a time — rasterio's dataset.sample() takes a list of
    # (x, y) pairs and yields a 1-D ndarray per point, so we get every value
    # for one variable in a single open/read cycle.
    rounded_keys = [
        (round(float(lat), prec), round(float(lon), prec))
        for lat, lon in to_sample
    ]
    xy = [(float(lon), float(lat)) for lat, lon in to_sample]

    var_iter = tqdm(var_ids, desc='Sampling BIOCLIM', unit='var') if verbose else var_ids
    for v in var_iter:
        tif = bioclim_raster_path(resolution, v, root)
        if not tif.exists():
            raise FileNotFoundError(
                f'BIOCLIM raster missing: {tif}. '
                'Call ensure_bioclim_rasters() to download/extract.'
            )
        with rasterio.open(tif) as src:
            nodata = src.nodata
            for key, sample in zip(rounded_keys, src.sample(xy)):
                val = float(sample[0])
                if nodata is not None and val == float(nodata):
                    val = float('nan')
                out[key][v - 1] = val

        # Persist progress after each raster so a partial run is not wasted.
        for key in rounded_keys:
            cache.put(*key, out[key])
        cache.save()

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_resolution(resolution: str) -> None:
    if resolution not in WORLDCLIM_RESOLUTIONS:
        raise ValueError(
            f'Unknown WorldClim resolution {resolution!r}. '
            f'Expected one of {WORLDCLIM_RESOLUTIONS}.'
        )


def _validate_var(var: int) -> None:
    if not (1 <= int(var) <= NUM_BIOCLIM_VARS):
        raise ValueError(
            f'BIOCLIM variable index must be in 1..{NUM_BIOCLIM_VARS}, got {var!r}'
        )
