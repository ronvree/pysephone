"""
Generic on-disk feature cache for any FeatureProvider.

FeatureCache wraps the output of any :class:`FeatureProvider` into a single
compressed ``.npz`` file.  Once built, the cache can be loaded instantly with
no open file handles — no HDF5 locking, no preload step in notebooks.

Typical usage
-------------
Build once (offline or in a setup cell)::

    from pysephone.dataset.util.feature_cache import FeatureCache
    from pysephone.dataset.util.openmeteo import OpenMeteoFeatures
    from pysephone.dataset.util.calendar import Calendar
    from pysephone.dataset.dataset import Dataset

    cal   = Calendar()
    feats = OpenMeteoFeatures(calendar=cal, data_keys=['temperature_2m_mean'])

    obs = Dataset.load('PEP725_Apple', calendar=cal).observations
    feats.download(obs)

    path = FeatureCache.default_path('PEP725_Apple', feats.data_keys)
    FeatureCache.build(feats, obs, path=path)

Use anywhere::

    cache = FeatureCache.load(path)
    ds    = Dataset.load('PEP725_Apple', calendar=cal, feature_providers=[cache])

Storage format
--------------
A single ``.npz`` (compressed numpy) file containing:

* ``_idx_src``, ``_idx_loc_id``, ``_idx_year``, ``_idx_species_id``,
  ``_idx_subgroup_id`` — parallel arrays encoding the sample index.
* ``data__<key>`` — feature array per data key.  If all seasons have the same
  length the array is 2-D ``(N, T)``; if lengths vary it is a flat 1-D array
  (CSR format) paired with an ``offsets__<key>`` array of shape ``(N+1,)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pysephone.dataset.util.provider import FeatureProvider
from pysephone.paths import get_products_data_dir, get_data_root


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(root: Optional[Path] = None) -> Path:
    if root is None:
        root = get_data_root()
    return get_products_data_dir(root) / 'feature_cache'


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

class FeatureCache(FeatureProvider):
    """In-memory feature cache backed by a compressed numpy file.

    After :meth:`load` every :meth:`get_data` call is a pure dict/array
    lookup — no I/O, no file handles, no locking.

    Args:
        index_map:  Mapping from 5-tuple sample index to integer row position.
        data:       ``{key: array}`` — either ``(N, T)`` for fixed-length
                    seasons, or flat ``(M,)`` for ragged seasons.
        offsets:    ``{key: (N+1,)}`` CSR offsets for ragged keys.  Empty if
                    all keys are fixed-length.
        data_keys:  Ordered list of feature key names.
    """

    def __init__(
        self,
        index_map: Dict[Tuple, int],
        data: Dict[str, np.ndarray],
        offsets: Dict[str, np.ndarray],
        data_keys: List[str],
    ) -> None:
        self._index_map = index_map
        self._data      = data
        self._offsets   = offsets
        self._data_keys = data_keys

    # ------------------------------------------------------------------
    # FeatureProvider interface
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return cached feature arrays for *index*.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)``

        Raises:
            KeyError: if *index* was not present when the cache was built.
        """
        try:
            row = self._index_map[index]
        except KeyError:
            raise KeyError(
                f"Index {index} not found in feature cache. "
                "Rebuild the cache to include this entry."
            )
        result = {}
        for k in self._data_keys:
            arr = self._data[k]
            if k in self._offsets:
                start = int(self._offsets[k][row])
                end   = int(self._offsets[k][row + 1])
                result[k] = arr[start:end].copy()
            else:
                result[k] = arr[row].copy()
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data_keys(self) -> List[str]:
        return list(self._data_keys)

    def __len__(self) -> int:
        return len(self._index_map)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        provider: FeatureProvider,
        observations_or_dataset,
        path: Optional[Path] = None,
        verbose: bool = True,
    ) -> 'FeatureCache':
        """Compute and cache features for every entry in *observations_or_dataset*.

        Args:
            provider:               Any :class:`FeatureProvider` — OpenMeteo,
                                    elevation, satellite, etc.
            observations_or_dataset: Anything exposing ``iter_index()``.
            path:                   Where to save the cache.  ``None`` keeps it
                                    in memory only.
            verbose:                Show a tqdm progress bar.

        Returns:
            A ready-to-use :class:`FeatureCache` instance.
        """
        indices = list(observations_or_dataset.iter_index())

        # Peek at the first entry to discover data keys
        if not indices:
            raise ValueError("observations_or_dataset has no entries.")
        sample_data = provider.get_data(indices[0])
        data_keys   = list(sample_data.keys())

        # Accumulate feature arrays
        buckets: Dict[str, List[np.ndarray]] = {k: [] for k in data_keys}
        it = tqdm(indices, desc='Building feature cache') if verbose else indices
        for index in it:
            feats = provider.get_data(index)
            for k in data_keys:
                buckets[k].append(np.asarray(feats[k], dtype=np.float32))

        # Pack into fixed-length (2-D) or ragged (CSR) arrays
        data: Dict[str, np.ndarray]    = {}
        offsets: Dict[str, np.ndarray] = {}
        for k, arrays in buckets.items():
            lengths = [len(a) for a in arrays]
            if len(set(lengths)) == 1:
                data[k] = np.stack(arrays)             # (N, T)
            else:
                flat         = np.concatenate(arrays)  # (M,)
                offs         = np.zeros(len(arrays) + 1, dtype=np.int64)
                offs[1:]     = np.cumsum(lengths)
                data[k]    = flat
                offsets[k] = offs

        index_map = {idx: i for i, idx in enumerate(indices)}
        cache = cls(index_map, data, offsets, data_keys)

        if path is not None:
            cache.save(path)
            if verbose:
                print(f"Feature cache saved → {path}")

        return cache

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write cache to *path* (compressed numpy format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        indices = sorted(self._index_map, key=lambda i: self._index_map[i])
        save_dict: Dict[str, np.ndarray] = {
            '_idx_src':        np.array([str(i[0])  for i in indices]),
            '_idx_loc_id':     np.array([str(i[1])  for i in indices]),
            '_idx_year':       np.array([int(i[2])  for i in indices], dtype=np.int32),
            '_idx_species_id': np.array([int(i[3])  for i in indices], dtype=np.int32),
            '_idx_subgroup_id':np.array([int(i[4])  for i in indices], dtype=np.int32),
        }
        for k, arr in self._data.items():
            save_dict[f'data__{k}'] = arr
        for k, off in self._offsets.items():
            save_dict[f'offsets__{k}'] = off

        np.savez_compressed(str(path), **save_dict)

    @classmethod
    def load(cls, path: Path) -> 'FeatureCache':
        """Load a previously saved cache from *path*.

        Args:
            path: Path to the ``.npz`` file (extension optional).
        """
        path = Path(path)
        if path.suffix not in ('.npz', ''):
            path = path.with_suffix('.npz')
        if not path.exists() and not path.with_suffix('.npz').exists():
            raise FileNotFoundError(f"Feature cache not found: {path}")

        npz = np.load(str(path) if path.suffix == '.npz' else str(path.with_suffix('.npz')))

        # Reconstruct index map
        srcs         = npz['_idx_src']
        loc_ids      = npz['_idx_loc_id']
        years        = npz['_idx_year']
        species_ids  = npz['_idx_species_id']
        subgroup_ids = npz['_idx_subgroup_id']

        def _coerce_loc(v: str):
            try:
                return int(v)
            except ValueError:
                return str(v)

        indices = [
            (str(srcs[i]), _coerce_loc(loc_ids[i]),
             int(years[i]), int(species_ids[i]), int(subgroup_ids[i]))
            for i in range(len(srcs))
        ]
        index_map = {idx: i for i, idx in enumerate(indices)}

        # Reconstruct data / offsets
        data: Dict[str, np.ndarray]    = {}
        offsets: Dict[str, np.ndarray] = {}
        data_keys: List[str]           = []

        for fname in npz.files:
            if fname.startswith('data__'):
                k = fname[len('data__'):]
                data[k] = npz[fname]
                if k not in data_keys:
                    data_keys.append(k)
            elif fname.startswith('offsets__'):
                k = fname[len('offsets__'):]
                offsets[k] = npz[fname]

        return cls(index_map, data, offsets, data_keys)

    @classmethod
    def exists(cls, path: Path) -> bool:
        """Return ``True`` if a cache file exists at *path*."""
        path = Path(path)
        return path.exists() or path.with_suffix('.npz').exists()

    # ------------------------------------------------------------------
    # Convenience: canonical path
    # ------------------------------------------------------------------

    @staticmethod
    def default_path(
        dataset_key: str,
        data_keys: List[str],
        step: str = 'daily',
        root: Optional[Path] = None,
    ) -> Path:
        """Return the canonical cache path for a dataset / feature combination.

        Args:
            dataset_key: Registry key (e.g. ``'PEP725_Apple'``).
            data_keys:   Feature variable names.
            step:        Temporal resolution (``'daily'`` or ``'hourly'``).
            root:        Data root.  Defaults to ``get_data_root()``.
        """
        keys_str = '__'.join(sorted(data_keys))
        fname    = f'{dataset_key}__{step}__{keys_str}.npz'
        return _get_cache_dir(root) / fname
