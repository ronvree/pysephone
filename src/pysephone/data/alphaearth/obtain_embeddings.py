# """
# AlphaEarth (Google/DeepMind) Satellite Embedding V1 (Annual) sampler + on-disk cache.
#
# Data source (Earth Engine ImageCollection):
#   "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
# Bands:
#   A00..A63 (64D embedding)
#
# This module:
# - Given (lat, lon), fetches the 64D embedding for every available year
# - Stores embeddings in a persistent HDF5 file
# - Checks the HDF5 store before downloading anything
# - Lets you load embeddings back from disk by identifiers
#
# Prereqs:
#   pip install earthengine-api h5py numpy
#
# One-time auth (interactive):
#   python -c "import ee; ee.Authenticate()"
#
# Then:
#   python -c "import ee; ee.Initialize(project='YOUR_GCP_PROJECT')"
# (or set the project in code below)
#
# Notes:
# - Earth Engine access requires an EE account.
# - The dataset is tiled; we filter by point + year, then sample the pixel at that point.
# """
#
# from __future__ import annotations
#
# import os
# from dataclasses import dataclass
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple, Union
#
# import hashlib
# import numpy as np
# import h5py
#
# try:
#     import ee  # type: ignore
# except ImportError as e:
#     raise ImportError(
#         "Missing dependency 'earthengine-api'. Install with: pip install earthengine-api"
#     ) from e
#
#
# # ----------------------------
# # Configuration / constants
# # ----------------------------
#
# EE_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
# EMBED_DIM = 64
# BAND_NAMES = [f"A{i:02d}" for i in range(EMBED_DIM)]
#
#
# def _ymd_year_range(year: int) -> Tuple[str, str]:
#     """Return ISO date strings for [Jan 1, year) .. [Jan 1, year+1)."""
#     start = f"{year:04d}-01-01"
#     end = f"{year + 1:04d}-01-01"
#     return start, end
#
#
# def _stable_location_id(lat: float, lon: float, *, precision: int = 6) -> str:
#     """
#     Deterministic ID for a coordinate.
#     precision=6 ~ 0.11m at equator for latitude; good enough for caching by "same point".
#     """
#     lat_r = round(float(lat), precision)
#     lon_r = round(float(lon), precision)
#     raw = f"lat={lat_r:.{precision}f}|lon={lon_r:.{precision}f}|prec={precision}"
#     h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
#     return f"{lat_r:.{precision}f}_{lon_r:.{precision}f}_{h}"
#
# def _ensure_h5_exists(path: str) -> None:
#     """
#     Create an empty HDF5 file if it doesn't already exist.
#     Safe to call repeatedly.
#     """
#     if not os.path.exists(path):
#         with h5py.File(path, "w"):
#             pass
#
# @dataclass(frozen=True)
# class EmbeddingKey:
#     location_id: str
#     year: int
#
#
#
# def _default_h5_path() -> str:
#     """
#     Return 'alphaearth_embeddings.h5' located next to this Python file.
#     """
#     return str(Path(__file__).resolve().parent / "alphaearth_embeddings.h5")
#
#
# class AlphaEarthEmbeddingStore:
#     """
#     HDF5-backed cache for AlphaEarth embeddings sampled at points.
#     Layout:
#       /v1/locations/<location_id>/
#           attrs: lat, lon, precision, created_utc
#           /embeddings/<year>    (dataset shape (64,), dtype float32)
#           /meta/<year>          (attributes on a group; or attrs on embedding dataset)
#     """
#
#     def __init__(self, h5_path: str=None):
#         self.h5_path = h5_path or _default_h5_path()
#         _ensure_h5_exists(self.h5_path)
#
#     # -------------
#     # EE utilities
#     # -------------
#
#     @staticmethod
#     def ee_init(*, project: Optional[str] = None) -> None:
#         """
#         Initialize Earth Engine for this process.
#         If you haven't authenticated yet, run: ee.Authenticate()
#         """
#         # ee.Initialize(project=project)
#         try:
#             if project:
#                 ee.Initialize(project=project)
#             else:
#                 ee.Initialize()
#         except Exception as e:
#             raise RuntimeError(
#                 "Failed to initialize Earth Engine. "
#                 "Make sure you've authenticated (ee.Authenticate()) and have access."
#             ) from e
#
#     @staticmethod
#     def available_years() -> list[int]:
#         col = ee.ImageCollection(EE_COLLECTION_ID)
#         # Pull only time_start to keep payload small
#         info = col.aggregate_array("system:time_start").getInfo()
#         years = sorted({datetime.fromtimestamp(t / 1000, tz=timezone.utc).year for t in info})
#         return years
#
#     @staticmethod
#     def _fetch_embedding_from_ee(lat: float, lon: float, year: int, *, scale: int = 10) -> np.ndarray:
#         """
#         Fetch the 64D embedding for one year at a point by sampling the image pixel.
#         Returns float32 vector length 64.
#         """
#         point = ee.Geometry.Point([float(lon), float(lat)])
#         start, end = _ymd_year_range(year)
#
#         img = (
#             ee.ImageCollection(EE_COLLECTION_ID)
#             .filterDate(start, end)
#             .filterBounds(point)
#             .first()
#         )
#
#         # If nothing covers this point/year (rare), fail clearly.
#         if img is None:
#             raise ValueError(f"No embedding image found for year={year} at lat={lat}, lon={lon}")
#
#         # Sample exactly at the point. We select all embedding bands.
#         # sample() yields a FeatureCollection; take the first feature.
#         fc = ee.Image(img).select(BAND_NAMES).sample(
#             region=point,
#             scale=int(scale),
#             numPixels=1,
#             geometries=False,
#         )
#
#         feat = ee.Feature(fc.first())
#         props = feat.toDictionary(BAND_NAMES).getInfo()
#
#         if not props:
#             raise ValueError(f"No embedding values returned for year={year} at lat={lat}, lon={lon}")
#
#         # Ensure ordering A00..A63
#         vec = np.array([props.get(b) for b in BAND_NAMES], dtype=np.float32)
#
#         if vec.shape != (EMBED_DIM,) or np.any(np.isnan(vec)):
#             # Some missingness might come through as None -> nan
#             raise ValueError(
#                 f"Invalid embedding vector (shape={vec.shape}, nan={np.isnan(vec).any()}) "
#                 f"for year={year} at lat={lat}, lon={lon}"
#             )
#
#         return vec
#
#     # ----------------
#     # HDF5 operations
#     # ----------------
#
#     def __len__(self) -> int:
#         """
#         Return the total number of stored embeddings (across all locations and years).
#         """
#         count = 0
#         with self._open("r") as f:
#             base = "v1/locations"
#             if base not in f:
#                 return 0
#
#             for loc in f[base].values():
#                 if "embeddings" in loc:
#                     count += len(loc["embeddings"])
#
#         return count
#
#     def _open(self, mode: str = "a") -> h5py.File:
#         """
#         'a'  → read/write, create if missing
#         'r'  → read-only (safe because file is ensured to exist)
#         """
#         return h5py.File(self.h5_path, mode)
#
#     def _location_group(self, f: h5py.File, location_id: str) -> h5py.Group:
#         return f.require_group(f"v1/locations/{location_id}")
#
#     def has(self, key: EmbeddingKey) -> bool:
#         try:
#             with self._open("r") as f:
#                 path = f"v1/locations/{key.location_id}/embeddings/{key.year}"
#                 return path in f
#         except OSError:
#             return False
#
#     def save(
#         self,
#         key: EmbeddingKey,
#         vec: np.ndarray,
#         *,
#         lat: float,
#         lon: float,
#         precision: int,
#         scale: int,
#         source: str = EE_COLLECTION_ID,
#     ) -> None:
#         vec = np.asarray(vec, dtype=np.float32)
#         if vec.shape != (EMBED_DIM,):
#             raise ValueError(f"Expected embedding shape ({EMBED_DIM},), got {vec.shape}")
#
#         with self._open("a") as f:
#             lg = self._location_group(f, key.location_id)
#             lg.attrs.setdefault("lat", float(lat))
#             lg.attrs.setdefault("lon", float(lon))
#             lg.attrs.setdefault("precision", int(precision))
#             lg.attrs.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
#
#             eg = lg.require_group("embeddings")
#             ds_path = f"{key.year}"
#             if ds_path in eg:
#                 # already exists; don't overwrite silently
#                 return
#
#             ds = eg.create_dataset(
#                 ds_path,
#                 data=vec,
#                 dtype="float32",
#                 compression="gzip",
#                 compression_opts=4,
#                 shuffle=True,
#             )
#             ds.attrs["year"] = int(key.year)
#             ds.attrs["scale_m"] = int(scale)
#             ds.attrs["source"] = str(source)
#             ds.attrs["stored_utc"] = datetime.now(timezone.utc).isoformat()
#
#     def load(self, key: EmbeddingKey) -> np.ndarray:
#         with self._open("r") as f:
#             path = f"v1/locations/{key.location_id}/embeddings/{key.year}"
#             if path not in f:
#                 raise KeyError(f"Embedding not found in store: {path}")
#             return np.array(f[path][...], dtype=np.float32)
#
#     def list_years_for_location(self, location_id: str) -> List[int]:
#         with self._open("r") as f:
#             base = f"v1/locations/{location_id}/embeddings"
#             if base not in f:
#                 return []
#             years = sorted(int(k) for k in f[base].keys())
#             return years
#
#     # ----------------------
#     # High-level API methods
#     # ----------------------
#
#     def get_embeddings_all_years(
#         self,
#         lat: float,
#         lon: float,
#         *,
#         years: Union[str, Iterable[int]] = "all",
#         precision: int = 6,
#         scale: int = 10,
#         ee_project: Optional[str] = None,
#     ) -> Tuple[str, Dict[int, np.ndarray]]:
#         """
#         Main entrypoint:
#         - Computes a location_id from (lat, lon)
#         - For each year (all available, or user-specified), loads from HDF5 if present
#           otherwise downloads from EE and stores
#         Returns (location_id, {year: embedding_vector})
#
#         IMPORTANT: Always checks disk cache before downloading.
#         """
#         # Ensure EE is initialized for this process.
#         # (Safe to call multiple times; EE will usually no-op after first init.)
#         self.ee_init(project=ee_project)
#
#         location_id = _stable_location_id(lat, lon, precision=precision)
#
#         if years == "all":
#             year_list = self.available_years()
#         else:
#             year_list = sorted({int(y) for y in years})
#
#         out: Dict[int, np.ndarray] = {}
#
#         for y in year_list:
#             key = EmbeddingKey(location_id=location_id, year=int(y))
#
#             # 1) Try disk
#             if self.has(key):
#                 out[int(y)] = self.load(key)
#                 continue
#
#             # 2) Otherwise fetch + persist
#             vec = self._fetch_embedding_from_ee(lat, lon, int(y), scale=scale)
#             self.save(key, vec, lat=lat, lon=lon, precision=precision, scale=scale)
#             out[int(y)] = vec
#
#         return location_id, out
#
#
# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     import ee
#     ee.Authenticate()
#
#     # Example: sample Amsterdam-ish point
#     lat, lon = 52.370216, 4.895168
#
#     store = AlphaEarthEmbeddingStore()
#
#     # If you need to specify a billing/project for EE:
#     location_id, embs = store.get_embeddings_all_years(lat, lon, ee_project="YOUR_GCP_PROJECT")
#
#     print("location_id:", location_id)
#     print("years:", sorted(embs.keys()))
#     print("2024 embedding shape:", embs[max(embs.keys())].shape)


"""
Batched AlphaEarth embedding sampling for many lat/lon points using Earth Engine reduceRegions.

Key idea:
- For each year, sample a whole batch of points at once:
    image.reduceRegions(FeatureCollection(points), Reducer.first(), scale)
- Much faster than per-point getInfo()

Includes:
- stable location_id for each lat/lon
- HDF5 cache check BEFORE requesting EE (per year, per point)
- HDF5 persistence (per-location, per-year datasets like your earlier schema)

Install:
  pip install earthengine-api h5py numpy

Auth/init:
  python -c "import ee; ee.Authenticate()"
  python -c "import ee; ee.Initialize(project='YOUR_GCP_PROJECT')"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
import math

import h5py
import numpy as np

from tqdm import tqdm

from pysephone.paths import get_data_root, get_ee_project, get_products_data_dir


def _require_ee():
    """Import earthengine-api lazily.  It is only needed for download/fetch
    operations, not for store lookup, so we don't force it as a hard dep."""
    try:
        import ee  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "earthengine-api is required to fetch new AlphaEarth embeddings. "
            'Install with: pip install "pysephone[earthengine]" '
            "(or: pip install earthengine-api)"
        ) from exc
    return ee

EE_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBED_DIM = 64
BANDS = [f"A{i:02d}" for i in range(EMBED_DIM)]


def _ymd_year_range(year: int) -> Tuple[str, str]:
    return f"{year:04d}-01-01", f"{year + 1:04d}-01-01"


def _stable_location_id(lat: float, lon: float, *, precision: int = 6) -> str:
    lat_r = round(float(lat), precision)
    lon_r = round(float(lon), precision)
    raw = f"lat={lat_r:.{precision}f}|lon={lon_r:.{precision}f}|prec={precision}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{lat_r:.{precision}f}_{lon_r:.{precision}f}_{h}"


def _default_h5_path() -> str:
    return str(
        get_products_data_dir(get_data_root()) / 'alphaearth' / 'alphaearth_embeddings.h5'
    )


def _ensure_h5_exists(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with h5py.File(p, "w"):
            pass


@dataclass(frozen=True)
class PointSpec:
    lat: float
    lon: float
    location_id: str


class AlphaEarthEmbeddingStore:
    """
    HDF5 cache layout (same as earlier per-year dataset approach):
      /v1/locations/<location_id>/
          attrs: lat, lon, precision, created_utc
          /embeddings/<year>    dataset (64,) float32
    """

    def __init__(self, h5_path: Optional[str] = None):
        self.h5_path = h5_path or _default_h5_path()
        _ensure_h5_exists(self.h5_path)

    def _open(self, mode: str = "a") -> h5py.File:
        return h5py.File(self.h5_path, mode)

    def _location_group(self, f: h5py.File, location_id: str) -> h5py.Group:
        return f.require_group(f"v1/locations/{location_id}")

    def has_year(self, location_id: str, year: int) -> bool:
        with self._open("r") as f:
            path = f"v1/locations/{location_id}/embeddings/{year}"
            return path in f

    def save_year(
        self,
        *,
        location_id: str,
        year: int,
        vec: np.ndarray,
        lat: float,
        lon: float,
        precision: int,
        scale: int,
        source: str = EE_COLLECTION_ID,
    ) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (EMBED_DIM,):
            raise ValueError(f"Expected embedding shape ({EMBED_DIM},), got {vec.shape}")

        with self._open("a") as f:
            lg = self._location_group(f, location_id)
            lg.attrs.setdefault("lat", float(lat))
            lg.attrs.setdefault("lon", float(lon))
            lg.attrs.setdefault("precision", int(precision))
            lg.attrs.setdefault("created_utc", datetime.now(timezone.utc).isoformat())

            eg = lg.require_group("embeddings")
            name = str(int(year))
            if name in eg:
                return  # already cached

            ds = eg.create_dataset(
                name,
                data=vec,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            ds.attrs["year"] = int(year)
            ds.attrs["scale_m"] = int(scale)
            ds.attrs["source"] = str(source)
            ds.attrs["stored_utc"] = datetime.now(timezone.utc).isoformat()


def _chunked(seq: Sequence[PointSpec], batch_size: int) -> Iterable[List[PointSpec]]:
    for i in range(0, len(seq), batch_size):
        yield list(seq[i : i + batch_size])


def _parse_embedding_from_props(props: dict) -> Optional[np.ndarray]:
    """
    reduceRegions + Reducer.first() sometimes yields either:
      - props["A00"] ... props["A63"]
    or
      - props["A00_first"] ... props["A63_first"]

    Return None if missing/invalid.
    """
    # Try direct band names first
    if all(b in props for b in BANDS):
        vals = [props.get(b) for b in BANDS]
    else:
        # Try suffixed outputs
        suff = [f"{b}_first" for b in BANDS]
        if all(k in props for k in suff):
            vals = [props.get(k) for k in suff]
        else:
            return None

    if any(v is None for v in vals):
        return None

    vec = np.array(vals, dtype=np.float32)
    if vec.shape != (EMBED_DIM,) or np.any(np.isnan(vec)):
        return None
    return vec


def fetch_alphaearth_embeddings_batched(
    latlons: Sequence[Tuple[float, float]],
    *,
    years: Iterable[int],
    store: AlphaEarthEmbeddingStore,
    precision: int = 6,
    scale: int = 10,
    batch_size: int = 500,
    ee_project: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Fetch embeddings for MANY points in batches using reduceRegions.

    - Checks HDF5 BEFORE requesting EE:
        only missing (location_id, year) pairs are sent to EE.
    - Returns:
        {location_id: {year: embedding_vec}}

    Notes:
    - For very large runs, consider exporting tables (Export.table.*) instead of getInfo().
    - batch_size: 200-1000 typically works; too large may hit response limits/timeouts.
    """
    ee = _require_ee()
    # Initialize EE (safe to call multiple times). Resolve the project from the
    # explicit argument, then PYSEPHONE_EE_PROJECT / EARTHENGINE_PROJECT env vars.
    # If nothing is set, let Earth Engine resolve its own default project.
    project = get_ee_project(ee_project)
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    points: List[PointSpec] = []
    for (lat, lon) in latlons:
        lid = _stable_location_id(lat, lon, precision=precision)
        points.append(PointSpec(lat=float(lat), lon=float(lon), location_id=lid))

    # We'll build up results for points we fetch or (optionally) already had cached
    out: Dict[str, Dict[int, np.ndarray]] = {p.location_id: {} for p in points}

    # We open H5 in read mode once per year to reduce overhead of many opens
    years_list = [int(y) for y in years]

    for year in years_list:
        # 1) Identify missing points for this year (cache check)
        missing: List[PointSpec] = []
        with store._open("r") as f:
            for p in points:
                path = f"v1/locations/{p.location_id}/embeddings/{year}"
                if path in f:
                    # If you also want to load cached values into `out`, uncomment below:
                    # out[p.location_id][year] = np.array(f[path][...], dtype=np.float32)
                    continue
                missing.append(p)

        if not missing:
            continue

        # 2) Build the year image once
        start, end = _ymd_year_range(year)
        img = (
            ee.ImageCollection(EE_COLLECTION_ID)
            .filterDate(start, end)
            .select(BANDS)
            .mosaic()
        )

        # 3) Batch query missing points for this year
        # for batch in _chunked(missing, batch_size=batch_size):
        num_batches = math.ceil(len(missing) / batch_size)
        for batch in tqdm(
                _chunked(missing, batch_size=batch_size),
                desc=f"Year {year} (points: {len(missing)})",
                unit="batch",
                total=num_batches,
                leave=False,
        ):
            feats = [
                ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {"pid": p.location_id})
                for p in batch
            ]
            fc = ee.FeatureCollection(feats)

            # Server-side sampling: attach band values to each feature
            reduced = img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=int(scale),
            )

            # Pull results to client (one payload per batch)
            attempt = 0
            last_err = None
            while attempt < max_retries:
                try:
                    info = reduced.getInfo()
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    attempt += 1
            if last_err is not None:
                raise RuntimeError(
                    f"EE reduceRegions failed for year={year}, batch_size={len(batch)} "
                    f"after {max_retries} attempts. Last error: {last_err}"
                )

            features = info.get("features", [])
            # Map pid -> props
            for ft in features:
                props = ft.get("properties", {}) or {}
                pid = props.get("pid")
                if not pid:
                    continue

                vec = _parse_embedding_from_props(props)
                if vec is None:
                    # No data for this point/year (e.g., water mask). Skip.
                    continue

                # Persist (HDF5) and add to output
                # Find the original lat/lon for metadata
                # (For speed, build a dict outside if you want; this is fine for 10k.)
                p = next((pp for pp in batch if pp.location_id == pid), None)
                if p is None:
                    continue

                store.save_year(
                    location_id=pid,
                    year=year,
                    vec=vec,
                    lat=p.lat,
                    lon=p.lon,
                    precision=precision,
                    scale=scale,
                )
                out[pid][year] = vec

    return out


def lookup_embeddings_from_store(
    store: AlphaEarthEmbeddingStore,
    lat: float,
    lon: float,
    *,
    years: Optional[Iterable[int]] = None,
    precision: int = 6,
) -> Dict[int, np.ndarray]:
    """
    Look up cached AlphaEarth embeddings for a given lat/lon.

    - NO Earth Engine interaction
    - Uses the same location_id derivation as the fetch code
    - If years is None, returns all available years for that location

    Returns:
        {year: embedding_vec}

    Raises:
        KeyError if the location is not present in the store at all
    """
    location_id = _stable_location_id(lat, lon, precision=precision)

    out: Dict[int, np.ndarray] = {}

    with store._open("r") as f:
        loc_path = f"v1/locations/{location_id}"
        if loc_path not in f:
            raise KeyError(f"Location not found in store: {location_id}")

        emb_path = f"{loc_path}/embeddings"
        if emb_path not in f:
            return out

        emb_grp = f[emb_path]

        if years is None:
            years_iter = (int(y) for y in emb_grp.keys())
        else:
            years_iter = (int(y) for y in years)

        for y in years_iter:
            name = str(int(y))
            if name not in emb_grp:
                continue
            out[int(y)] = np.array(emb_grp[name][...], dtype=np.float32)

    return out


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    ee = _require_ee()
    ee.Authenticate()

    # Example: 3 points
    latlons = [
        (52.370216, 4.895168),
        (52.3676, 4.9041),
        (52.3791, 4.9003),
    ]

    store = AlphaEarthEmbeddingStore()  # default alphaearth_embeddings.h5 next to this file

    # Typical years (documented range is often 2017..2024 for this dataset)
    years = range(2017, 2025)

    res = fetch_alphaearth_embeddings_batched(
        latlons,
        years=years,
        store=store,
        batch_size=500,
        scale=10,
        # Supply your own Earth Engine GCP project here, or set the
        # PYSEPHONE_EE_PROJECT / EARTHENGINE_PROJECT environment variable.
        ee_project=None,
    )

    for lid, by_year in res.items():
        print(lid, "years:", sorted(by_year.keys()))

    out = lookup_embeddings_from_store(store, lat=52.370216, lon=4.895168)

    print(out)