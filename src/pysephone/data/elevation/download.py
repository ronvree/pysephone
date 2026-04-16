"""
Elevation lookup via the OpenMeteo Elevation API.

- API endpoint:  https://api.open-meteo.com/v1/elevation
- Data source:   SRTM / Copernicus GLO-30 digital elevation model
- Batching:      up to 100 points per request
- No authentication required

Elevations are cached to a JSON file on disk keyed by
``(round(lat, precision), round(lon, precision))`` so repeated runs across
datasets reuse already-fetched values.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm

from pysephone.paths import get_data_root, get_products_data_dir


_OPENMETEO_ELEVATION_URL = 'https://api.open-meteo.com/v1/elevation'
DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2.0        # seconds, doubled on each retry
DEFAULT_PAUSE_BETWEEN_BATCHES = 0.5  # polite pacing to stay under the burst limit


def default_cache_path() -> Path:
    return get_products_data_dir(get_data_root()) / 'elevation' / 'elevations.json'


class ElevationCache:
    """JSON-backed ``(lat, lon) -> elevation`` cache.

    Both the on-disk and in-memory representations round the coordinate key
    to *precision* decimals so small floating-point noise does not cause cache
    misses.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        precision: int = 4,
    ) -> None:
        self._path = Path(path) if path is not None else default_cache_path()
        self._precision = int(precision)
        self._cache: Dict[Tuple[float, float], float] = {}
        self._load()

    # -- Public API --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def precision(self) -> int:
        return self._precision

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, coord: Tuple[float, float]) -> bool:
        return self._key(*coord) in self._cache

    def get(self, lat: float, lon: float) -> Optional[float]:
        return self._cache.get(self._key(lat, lon))

    def put(self, lat: float, lon: float, elevation: float) -> None:
        self._cache[self._key(lat, lon)] = float(elevation)

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
            self._cache[(float(lat_s), float(lon_s))] = float(v)


def _request_batch(
    lats: str,
    lons: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> list:
    """GET a single elevation batch with retry on 429 / transient errors.

    Respects the ``Retry-After`` header when present.
    """
    delay = backoff
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                _OPENMETEO_ELEVATION_URL,
                params={'latitude': lats, 'longitude': lons},
                timeout=timeout,
            )
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
            continue

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
        return resp.json().get('elevation', [])

    raise RuntimeError('unreachable')


def fetch_elevations(
    coords: Sequence[Tuple[float, float]],
    cache: Optional[ElevationCache] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 30.0,
    verbose: bool = True,
    force: bool = False,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    pause_between_batches: float = DEFAULT_PAUSE_BETWEEN_BATCHES,
) -> Dict[Tuple[float, float], float]:
    """Fetch elevations for a sequence of ``(lat, lon)`` points.

    Cached points are returned from disk; missing points are fetched from the
    OpenMeteo API in batches, stored in the cache, and persisted **after every
    batch** so a rate-limit failure partway through does not lose progress.

    Args:
        coords:                 Points to look up as ``(lat, lon)`` tuples.
        cache:                  :class:`ElevationCache` to use (created at the
                                default path if ``None``).
        batch_size:             Points per API call (capped at 100).
        timeout:                HTTP timeout in seconds.
        verbose:                Show a progress bar.
        force:                  Re-fetch even cached points.
        retries:                Retry attempts on 429 / 5xx / network errors.
        backoff:                Base wait in seconds between retries (doubled
                                each attempt, overridden by ``Retry-After``).
        pause_between_batches:  Small sleep between successive batches to stay
                                under OpenMeteo's burst limit.

    Returns:
        ``{(lat_rounded, lon_rounded): elevation_m}`` for every input point.
    """
    cache = cache if cache is not None else ElevationCache()

    to_fetch: List[Tuple[float, float]] = []
    out: Dict[Tuple[float, float], float] = {}
    for lat, lon in coords:
        key = (round(lat, cache.precision), round(lon, cache.precision))
        if not force and key in cache:
            out[key] = cache.get(*key)
        else:
            to_fetch.append((lat, lon))

    if not to_fetch:
        return out

    batch_size = min(int(batch_size), DEFAULT_BATCH_SIZE)
    batches = [to_fetch[i:i + batch_size] for i in range(0, len(to_fetch), batch_size)]
    it = tqdm(batches, desc='Fetching elevations', unit='batch') if verbose else batches

    prec = cache.precision
    for bi, batch in enumerate(it):
        lats = ','.join(f'{lat:.{prec}f}' for lat, _ in batch)
        lons = ','.join(f'{lon:.{prec}f}' for _, lon in batch)
        elevs = _request_batch(
            lats, lons, timeout=timeout, retries=retries, backoff=backoff,
        )
        if len(elevs) != len(batch):
            raise RuntimeError(
                f'OpenMeteo returned {len(elevs)} elevations for a batch of {len(batch)}'
            )
        for (lat, lon), elev in zip(batch, elevs):
            cache.put(lat, lon, elev)
            out[(round(lat, prec), round(lon, prec))] = float(elev)

        # Persist after every batch — partial progress survives crashes
        cache.save()

        if pause_between_batches and bi + 1 < len(batches):
            time.sleep(pause_between_batches)

    return out
