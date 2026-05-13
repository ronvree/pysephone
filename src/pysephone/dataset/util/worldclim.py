"""
WorldClim BIOCLIM feature provider for use with Dataset.

Wraps :mod:`pysephone.data.worldclim` (BIOCLIM raster download + JSON sample
cache) to serve the 19 globally-interpolated WorldClim variables at each
dataset sample's location.  The data are climatological (1970-2000 baseline)
and therefore depend on ``(src, loc_id)`` only — not the observation year.

Two output layouts are supported:

- ``layout='dict'`` (default): one feature key per BIO variable —
  ``{'bio_1': np.array([..]), 'bio_2': np.array([..]), ...}``.  Useful when
  downstream models pick variables by name (e.g. only annual temperature and
  precipitation seasonality).
- ``layout='vector'``: a single key whose value is the full 19-D vector —
  ``{'bioclim': np.array([19])}``.  Useful when the bioclimatic profile is fed
  in as a single static covariate alongside time-series features.

Usage mirrors :class:`ElevationFeatures`:

- ``Dataset.download_features()`` triggers :meth:`download` which downloads
  and extracts the BIOCLIM rasters (if needed) and samples every uncached
  ``(lat, lon)`` to disk.
- ``Dataset.preload_features()`` triggers :meth:`preload` which builds the
  in-memory ``(src, loc_id) -> values`` map used by :meth:`get_data`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from pysephone.data.worldclim.download import (
    BIOCLIM_KEYS,
    BIOCLIM_VAR_INDICES,
    BIOCLIM_VAR_NAMES,
    DEFAULT_RESOLUTION,
    NUM_BIOCLIM_VARS,
    BioclimCache,
    ensure_bioclim_rasters,
    sample_bioclim,
)
from pysephone.dataset.util.provider import FeatureProvider


# ---------------------------------------------------------------------------
# Variable-spec normalization
# ---------------------------------------------------------------------------

def _normalize_variables(
    variables: Optional[Sequence],
) -> Tuple[List[int], List[str]]:
    """Resolve a user-provided variable spec to ``(indices, output_keys)``.

    Accepts:
      - ``None``                       → all 19 BIOCLIM variables, keys ``bio_1..bio_19``
      - integer indices                → e.g. ``[1, 12, 15]``
      - ``'bio_<n>'`` strings          → e.g. ``['bio_1', 'bio_12']``
      - descriptive names              → e.g. ``['annual_mean_temperature']``
    Mixing these forms in one list is allowed.

    Returns parallel lists: the underlying BIO indices (1..19), and the
    feature-key strings the provider will emit for each.
    """
    if variables is None:
        return list(BIOCLIM_VAR_INDICES), list(BIOCLIM_KEYS)

    name_to_index = {name: idx for idx, name in BIOCLIM_VAR_NAMES.items()}
    indices: List[int] = []
    keys: List[str] = []
    for spec in variables:
        if isinstance(spec, (int, np.integer)):
            idx = int(spec)
            if not (1 <= idx <= NUM_BIOCLIM_VARS):
                raise ValueError(
                    f'BIOCLIM index must be in 1..{NUM_BIOCLIM_VARS}, got {idx}'
                )
            indices.append(idx)
            keys.append(f'bio_{idx}')
        elif isinstance(spec, str):
            if spec.startswith('bio_'):
                try:
                    idx = int(spec[len('bio_'):])
                except ValueError as e:
                    raise ValueError(f'Cannot parse BIOCLIM key {spec!r}') from e
                if not (1 <= idx <= NUM_BIOCLIM_VARS):
                    raise ValueError(
                        f'BIOCLIM index in {spec!r} must be 1..{NUM_BIOCLIM_VARS}'
                    )
                indices.append(idx)
                keys.append(f'bio_{idx}')
            elif spec in name_to_index:
                idx = name_to_index[spec]
                indices.append(idx)
                # When the user asks by descriptive name, return them by that
                # name — this is the only way to round-trip the input.
                keys.append(spec)
            else:
                raise ValueError(
                    f'Unknown BIOCLIM variable {spec!r}. Expected an int 1..19, '
                    f"a 'bio_<n>' key, or one of {sorted(name_to_index)}."
                )
        else:
            raise TypeError(
                f'Unsupported BIOCLIM variable spec type: {type(spec).__name__}'
            )

    return indices, keys


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class WorldClimFeatures(FeatureProvider):
    """WorldClim BIOCLIM bioclimatic features per dataset location.

    Each BIO variable is a single scalar per ``(lat, lon)``, so the provider
    emits length-1 arrays in dict layout (one per variable) or a single
    length-:data:`NUM_BIOCLIM_VARS` array in vector layout.

    Args:
        resolution: WorldClim resolution to read.  Default ``'30s'`` (~1km),
                    matching the BIOCLIM product most commonly used for
                    species-distribution / phenology work.  Other valid values:
                    ``'2.5m'``, ``'5m'``, ``'10m'``.
        variables:  Subset of BIO variables to expose, as integer indices
                    (1..19), ``'bio_<n>'`` keys, or descriptive names from
                    :data:`BIOCLIM_VAR_NAMES`.  ``None`` exposes all 19.
        layout:     ``'dict'`` (default) — one feature per variable, keyed by
                    the resolved name.  ``'vector'`` — a single feature whose
                    value is the concatenated vector across *variables*.
        vector_key: Feature key used in ``layout='vector'`` mode.
        precision:  Decimal places used for the ``(lat, lon)`` cache key.
                    4 ≈ 11 m at the equator, finer than even the 30s raster.
        missing:    What to emit when the cache has no value for a sample:

                    - ``'error'`` (default) — raise :class:`RuntimeError`.
                    - ``'nan'``    — emit NaN values.
        cache_path: Override the JSON sample cache path.  Defaults to
                    ``<data_root>/data/products/worldclim/bioclim_samples_<res>.json``.
        root:       Override the data root.
    """

    DEFAULT_VECTOR_KEY: str = 'bioclim'

    def __init__(
        self,
        resolution: str = DEFAULT_RESOLUTION,
        variables: Optional[Sequence] = None,
        layout: str = 'dict',
        vector_key: str = DEFAULT_VECTOR_KEY,
        precision: int = 4,
        missing: str = 'error',
        cache_path: Optional[Path] = None,
        root: Optional[Path] = None,
    ) -> None:
        if layout not in ('dict', 'vector'):
            raise ValueError(f"layout must be 'dict' or 'vector', got {layout!r}")
        if missing not in ('error', 'nan'):
            raise ValueError(f"missing must be 'error' or 'nan', got {missing!r}")

        self._resolution = resolution
        self._var_indices, self._var_keys = _normalize_variables(variables)
        self._layout = layout
        self._vector_key = str(vector_key)
        self._missing = missing
        self._root = root

        self._cache = BioclimCache(
            resolution=resolution,
            path=cache_path,
            precision=precision,
            root=root,
        )

        # (src, loc_id) -> np.ndarray of length len(var_indices), dtype float32
        self._loc_values: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        src, loc_id, _year, _species_id, _subgroup_id = index
        key = (src, loc_id)
        if key not in self._loc_values:
            if self._missing == 'nan':
                vec = np.full(len(self._var_indices), np.nan, dtype=np.float32)
            else:
                raise RuntimeError(
                    f'WorldClimFeatures: no BIOCLIM values loaded for '
                    f'({src!r}, {loc_id!r}). Call Dataset.download_features() / '
                    'preload_features(), or preload(observations) directly.'
                )
        else:
            vec = self._loc_values[key]

        if self._layout == 'vector':
            return {self._vector_key: vec.copy()}
        # Dict layout: each variable becomes its own length-1 array, matching
        # the convention used by ElevationFeatures so callers can stack scalar
        # features uniformly.
        return {
            k: np.asarray([vec[i]], dtype=np.float32)
            for i, k in enumerate(self._var_keys)
        }

    def download(
        self,
        observations,
        verbose: bool = True,
        download_mode: Optional[str] = None,
    ) -> None:
        """Ensure rasters are on disk and sample every uncached location.

        Args:
            download_mode:
                - ``None``     — sample only missing values (default).
                - ``'forced'`` — re-sample every point.
                - ``'skip'``   — skip both the archive download and sampling.
        """
        if download_mode == 'skip':
            return

        # Make sure the BIOCLIM rasters are unpacked locally before sampling.
        # ensure_bioclim_rasters() is idempotent when both archive and rasters
        # are present, so this is cheap on repeat calls.
        ensure_bioclim_rasters(
            resolution=self._resolution,
            root=self._root,
            verbose=verbose,
            force=False,
        )

        coords = [
            (lat, lon)
            for _src, _loc, lat, lon in self._collect_locations(observations)
        ]
        sample_bioclim(
            coords,
            resolution=self._resolution,
            cache=self._cache,
            root=self._root,
            verbose=verbose,
            force=(download_mode == 'forced'),
            variables=self._var_indices,
        )

    def preload(self, observations, verbose: bool = True) -> None:
        """Populate the in-memory ``(src, loc_id) -> values`` map from the
        disk cache.  Samples whose coordinate is still missing will raise
        :class:`RuntimeError` (or emit NaN) at :meth:`get_data` time depending
        on the ``missing`` policy."""
        needed = self._collect_locations(observations)
        it = tqdm(needed, desc='Preloading BIOCLIM') if verbose else needed
        for src, loc_id, lat, lon in it:
            full = self._cache.get(lat, lon)
            if full is None:
                continue
            # Project the cached 19-vector down to the configured subset.
            vec = np.asarray(
                [full[i - 1] for i in self._var_indices],
                dtype=np.float32,
            )
            self._loc_values[(src, loc_id)] = vec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._loc_values.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> str:
        return self._resolution

    @property
    def variables(self) -> List[int]:
        return list(self._var_indices)

    @property
    def variable_keys(self) -> List[str]:
        return list(self._var_keys)

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def cache_path(self) -> Path:
        return self._cache.path

    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_locations(
        self, observations
    ) -> List[Tuple[str, str, float, float]]:
        seen: set = set()
        out: List[Tuple[str, str, float, float]] = []
        for ix in observations.iter_index():
            src, loc_id = ix[0], ix[1]
            if (src, loc_id) in seen:
                continue
            seen.add((src, loc_id))
            coords = observations.get_location_coordinates((src, loc_id))
            out.append((src, loc_id, float(coords['lat']), float(coords['lon'])))
        return out
