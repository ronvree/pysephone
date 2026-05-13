import hashlib
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from pysephone.paths import get_observations_source_data_dir


KEY = 'usa_npn'

NPN_BASE = 'https://services.usanpn.org/npn_portal'
_URL_SPECIES = f'{NPN_BASE}/species/getSpecies.json'
_URL_PHENOMETRICS = f'{NPN_BASE}/observations/getSummarizedData.json'

_FN_SPECIES = 'species.csv'
_RAW_SUBDIR = 'raw'
_FN_TEMPLATE_PHENOMETRICS = 'individual_phenometrics_{start}_{end}_p{phenophase}_{key}.csv'

# Sentinel value used by USA-NPN for missing numeric fields.
NPN_MISSING = -9999

# Polite delay between successive POST requests (seconds).
_DOWNLOAD_DELAY = 0.05


def path_species_cache(root: Path) -> Path:
    return get_observations_source_data_dir(root, KEY) / _RAW_SUBDIR / _FN_SPECIES


def path_phenometrics_cache(
    root: Path,
    species_ids: Iterable[int],
    phenophase_id: int,
    start_date: str,
    end_date: str,
) -> Path:
    """
    Filename matches the convention used by `notebooks/usa_npn_*.ipynb`, so a
    notebook download and a provider download with the same parameters share
    one on-disk cache file.
    """
    species_ids = sorted(int(s) for s in species_ids)
    sha = hashlib.sha1(
        f'{species_ids}|{phenophase_id}|{start_date}|{end_date}'.encode()
    ).hexdigest()[:10]
    fn = _FN_TEMPLATE_PHENOMETRICS.format(
        start=start_date, end=end_date,
        phenophase=phenophase_id, key=sha,
    )
    return get_observations_source_data_dir(root, KEY) / _RAW_SUBDIR / fn


def fetch_species_table(
    root: Path,
    request_src: str = 'pysephone',
    force_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch the USA-NPN species catalogue, cached on disk."""
    cache = path_species_cache(root)
    if cache.exists() and not force_download:
        return pd.read_csv(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'GET {_URL_SPECIES}')
    resp = requests.get(_URL_SPECIES, params={'request_src': request_src}, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(cache, index=False)
    if verbose:
        print(f'  cached {cache.name}: {len(df):,} species')
    return df


def fetch_individual_phenometrics(
    root: Path,
    species_ids: Iterable[int],
    phenophase_id: int,
    start_date: str,
    end_date: str,
    request_src: str = 'pysephone',
    force_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    POST the bulk Individual Phenometrics request, cached by parameter hash.

    Wire-level endpoint is `observations/getSummarizedData.json` — that is the
    Individual Phenometrics endpoint despite the un-obvious name (matches the
    R `rnpn::npn_download_individual_phenometrics` helper).

    Returns one row per (Site × Individual × Year) for the requested phenophase,
    with `first_yes_*` and `last_yes_*` date components and other site fields.
    """
    cache = path_phenometrics_cache(root, species_ids, phenophase_id, start_date, end_date)
    if cache.exists() and not force_download:
        if verbose:
            print(f'[cache] {cache.name}')
        return pd.read_csv(cache, low_memory=False)

    species_ids = sorted(int(s) for s in species_ids)

    payload: dict = {
        'request_src': request_src,
        'start_date':  start_date,
        'end_date':    end_date,
    }
    for i, sid in enumerate(species_ids, start=1):
        payload[f'species_id[{i}]'] = sid
    payload['phenophase_id[1]'] = phenophase_id

    if verbose:
        print(f'POST {_URL_PHENOMETRICS}')
        print(f'  ({len(species_ids)} species, {start_date}..{end_date}, '
              f'phenophase={phenophase_id})')

    resp = requests.post(_URL_PHENOMETRICS, data=payload, timeout=600)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    if verbose:
        print(f'  -> {len(df):,} rows, saved to {cache.name}')

    time.sleep(_DOWNLOAD_DELAY)
    return df


def fetch_all_phenometrics(
    root: Path,
    species_ids: Iterable[int],
    phenophase_ids: Iterable[int],
    start_date: str,
    end_date: str,
    request_src: str = 'pysephone',
    force_download: bool = False,
    verbose: bool = True,
) -> dict[int, pd.DataFrame]:
    """Fetch one CSV per phenophase, returning {phenophase_id: dataframe}."""
    species_ids = list(species_ids)
    phenophase_ids = list(phenophase_ids)

    iterable = (
        tqdm(phenophase_ids, desc='Downloading USA-NPN phenometrics')
        if verbose and len(phenophase_ids) > 1
        else phenophase_ids
    )

    out: dict[int, pd.DataFrame] = {}
    for p in iterable:
        out[int(p)] = fetch_individual_phenometrics(
            root,
            species_ids=species_ids,
            phenophase_id=int(p),
            start_date=start_date,
            end_date=end_date,
            request_src=request_src,
            force_download=force_download,
            verbose=verbose,
        )
    return out


# ---------------------------------------------------------------------------
# Build helpers — turn raw NPN frames into ObservationData-compatible tables.
# Each returns a DataFrame keyed only by the source-level columns; the source
# class adds the `src` index level.
# ---------------------------------------------------------------------------

def _validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with -9999 / out-of-range date components and attach a parsed
    `first_yes_date` (datetime).
    """
    df = df[
        df['first_yes_month'].between(1, 12)
        & df['first_yes_day'].between(1, 31)
        & df['first_yes_year'].between(1900, 2100)
    ].copy()
    df['first_yes_date'] = pd.to_datetime(
        dict(
            year=df['first_yes_year'].astype(int),
            month=df['first_yes_month'].astype(int),
            day=df['first_yes_day'].astype(int),
        ),
        errors='coerce',
    )
    return df.dropna(subset=['first_yes_date'])


def obs_type_for(phenophase_id: int) -> str:
    """Stable, parse-friendly obs_type string for a USA-NPN phenophase."""
    return f'NPN_{int(phenophase_id)}'


def create_observations_df(phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack per-phenophase frames into a single observations DataFrame.

    Index:   loc_id (site_id), year, species_id, subgroup_id (individual_id), obs_type
    Columns: first_yes_date  (the source class renames this to KEY_OBSERVATIONS)
    """
    parts = []
    for phenophase_id, df in phenometrics.items():
        if df.empty:
            continue
        df = _validate_dates(df)
        if df.empty:
            continue
        df = df.assign(obs_type=obs_type_for(phenophase_id))
        parts.append(df[[
            'site_id', 'first_yes_year', 'species_id',
            'individual_id', 'obs_type', 'first_yes_date',
        ]])

    if not parts:
        return pd.DataFrame(
            {'first_yes_date': pd.Series(dtype='datetime64[ns]')},
            index=pd.MultiIndex.from_tuples(
                [], names=['loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type'],
            ),
        )

    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={
        'site_id':        'loc_id',
        'first_yes_year': 'year',
        'individual_id':  'subgroup_id',
    })
    df['loc_id']      = df['loc_id'].astype(int)
    df['year']        = df['year'].astype(int)
    df['species_id']  = df['species_id'].astype(int)
    df['subgroup_id'] = df['subgroup_id'].astype(int)

    # USA-NPN occasionally repeats (site, individual, year, phenophase). Keep
    # the earliest first-yes date as canonical.
    df = (df.sort_values(['loc_id', 'year', 'species_id', 'subgroup_id',
                          'obs_type', 'first_yes_date'])
            .drop_duplicates(
              subset=['loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type'],
              keep='first'))

    return df.set_index(
        ['loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type']
    )[['first_yes_date']]


def create_events_df(phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the events table. Each phenophase id becomes one event whose
    description is read from the raw frame (or falls back to the id itself
    if the frame is empty).

    Index:   event ('NPN_{phenophase_id}')
    Columns: description
    """
    rows = []
    for phenophase_id, df in phenometrics.items():
        desc = None
        if not df.empty and 'phenophase_description' in df.columns:
            descs = df['phenophase_description'].dropna().unique().tolist()
            if descs:
                desc = descs[0]
        if desc is None:
            desc = f'USA-NPN phenophase {phenophase_id}'
        rows.append({'event': obs_type_for(phenophase_id), 'description': desc})
    return pd.DataFrame(rows).set_index('event')[['description']]


def create_locations_df(phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the locations table from the union of sites seen across phenophases.

    Index:   loc_id (site_id)
    Columns: lat, lon, alt, state
    """
    parts = []
    for df in phenometrics.values():
        if df.empty:
            continue
        cols = ['site_id', 'latitude', 'longitude', 'elevation_in_meters', 'state']
        cols = [c for c in cols if c in df.columns]
        parts.append(df[cols].copy())

    if not parts:
        return pd.DataFrame(
            columns=['lat', 'lon', 'alt', 'state'],
            index=pd.Index([], name='loc_id'),
        )

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset='site_id')
    df = df.rename(columns={
        'site_id':              'loc_id',
        'latitude':             'lat',
        'longitude':            'lon',
        'elevation_in_meters':  'alt',
    })
    # Drop NPN sentinel elevations.
    if 'alt' in df.columns:
        df.loc[df['alt'] == NPN_MISSING, 'alt'] = pd.NA
    df['loc_id'] = df['loc_id'].astype(int)
    keep = [c for c in ['lat', 'lon', 'alt', 'state'] if c in df.columns]
    return df.set_index('loc_id')[keep]


_SPECIES_KEEP_COLS = (
    'genus', 'species', 'common_name', 'family_name', 'kingdom',
)


def create_species_df(df_species_catalogue: pd.DataFrame) -> pd.DataFrame:
    """
    Build the species table by projecting the NPN catalogue.

    Index:   species_id (int)
    Columns: genus, species, common_name, family_name, kingdom (whichever exist)
    """
    df = df_species_catalogue.copy()
    df.columns = [c.lower() for c in df.columns]
    df['species_id'] = df['species_id'].astype(int)
    keep = [c for c in _SPECIES_KEEP_COLS if c in df.columns]
    return df.drop_duplicates(subset='species_id').set_index('species_id')[keep]


def create_subgroups_df(phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    USA-NPN tracks observations per individual plant. Each `individual_id`
    becomes a subgroup belonging to its species.

    Index:   subgroup_id (individual_id)
    Columns: species_id
    """
    parts = []
    for df in phenometrics.values():
        if df.empty:
            continue
        if 'individual_id' not in df.columns or 'species_id' not in df.columns:
            continue
        parts.append(df[['individual_id', 'species_id']].copy())

    if not parts:
        return pd.DataFrame(
            {'species_id': pd.Series(dtype='int64')},
            index=pd.Index([], name='subgroup_id'),
        )

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset='individual_id')
    df = df.rename(columns={'individual_id': 'subgroup_id'})
    df['subgroup_id'] = df['subgroup_id'].astype(int)
    df['species_id']  = df['species_id'].astype(int)
    return df.set_index('subgroup_id')[['species_id']]


def filter_species_by_genus(
    df_species: pd.DataFrame,
    genera: Iterable[str] | None,
    plantae_only: bool = True,
) -> pd.DataFrame:
    """Restrict the NPN catalogue to the given genera (case-sensitive)."""
    df = df_species.copy()
    if plantae_only and 'kingdom' in df.columns:
        df = df[df['kingdom'].fillna('Plantae') == 'Plantae']
    if genera is not None:
        df = df[df['genus'].isin(set(genera))]
    return df.reset_index(drop=True)
