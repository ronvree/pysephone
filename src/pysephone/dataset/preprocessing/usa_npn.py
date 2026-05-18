import pandas as pd

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_OBSERVATIONS,
    KEY_SPECIES_ID,
)
from pysephone.data.source import ObservationData
from pysephone.dataset.util.func import (
    empty_df_like,
    filter_outliers,
    select_location,
    select_observation_type,
    select_species,
    select_year,
)


"""
    Preprocess USA-NPN ObservationData into dataframes compatible with the dataset classes.

    The USA-NPN source already builds tables on the standard MultiIndex
    (``src, loc_id, year, species_id, subgroup_id, obs_type``), so this module
    only handles filtering, outlier removal, and species-name extraction.
"""


def get_usa_npn_dataframes(
    data: ObservationData,
    remove_outliers: bool = True,
    filter_on_species=None,
    filter_on_years=None,
    filter_on_locations=None,
    filter_on_observation_types=None,
    filter_on_taxa=None,
    datetime_observations: bool = True,
) -> dict:
    """
    Args:
        data:                        ObservationData produced by USANPNSource.get_data().
        remove_outliers:             Remove per-obs_type outliers via symmetric quantile trimming.
        filter_on_species:           Optional iterable of (src, species_id) [or
                                     (src, species_id, subgroup_id)] tuples to keep.
        filter_on_years:             Optional iterable of year values to keep.
        filter_on_locations:         Optional iterable of (src, loc_id) tuples to keep.
        filter_on_observation_types: Optional iterable of obs_type strings to keep
                                     (e.g. ``'NPN_501'`` for open flowers).
        filter_on_taxa:              Optional iterable of taxonomic selectors translated
                                     to species_ids via ``data.species``. Each item is
                                     either a genus string (``'Malus'``) or a
                                     ``(genus, species_epithet)`` tuple (``('Prunus', 'persica')``).
        datetime_observations:       If True, keep KEY_OBSERVATIONS as datetime.
                                     If False, convert to integer day-of-year.

    Returns:
        dict with keys:
            'data':          observations DataFrame indexed by the standard multi-index.
            'locations':     locations DataFrame indexed by (src, loc_id).
            'species_names': ``{(src, species_id): 'Genus species'}`` mapping.
    """
    df_y_loc = data.locations.copy()
    df_y = data.observations.copy()

    if filter_on_taxa is not None:
        taxa_species = _species_ids_for_taxa(data, filter_on_taxa)
        if not taxa_species:
            df_y = empty_df_like(df_y)
        else:
            df_y = select_species(df_y, taxa_species)

    if filter_on_species is not None:
        df_y = select_species(df_y, filter_on_species)

    if filter_on_years is not None:
        df_y = select_year(df_y, filter_on_years)

    if filter_on_locations is not None:
        df_y = select_location(df_y, filter_on_locations)

    if filter_on_observation_types is not None:
        df_y = select_observation_type(df_y, filter_on_observation_types)

    if remove_outliers:
        df_y = filter_outliers(df_y)

    if not datetime_observations:
        df_y[KEY_OBSERVATIONS] = df_y[KEY_OBSERVATIONS].dt.dayofyear

    species_names = _build_species_names(data, df_y)

    return {'data': df_y, 'locations': df_y_loc, 'species_names': species_names}


def _species_ids_for_taxa(data: ObservationData, taxa) -> list:
    """Resolve a list of genus / (genus, species) selectors to (src, species_id) tuples."""
    if data.species is None or len(data.species) == 0:
        return []

    df_sp = data.species.reset_index()
    out: list = []
    seen: set = set()
    for t in taxa:
        if isinstance(t, str):
            mask = df_sp['genus'] == t
        elif isinstance(t, tuple) and len(t) == 2:
            genus, species_epithet = t
            mask = (df_sp['genus'] == genus) & (df_sp['species'] == species_epithet)
        else:
            raise ValueError(
                f'taxon selectors must be a genus string or (genus, species) tuple, got {t!r}'
            )
        for src, sid in df_sp.loc[mask, [KEY_DATA_SOURCE, KEY_SPECIES_ID]].itertuples(
            index=False, name=None,
        ):
            key = (src, int(sid))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _build_species_names(data: ObservationData, df_y: pd.DataFrame) -> dict:
    """Construct ``{(src, species_id): 'Genus species'}`` for species in df_y."""
    species_names: dict = {}
    if not hasattr(data, 'species') or data.species is None:
        return species_names

    present = set(df_y.index.get_level_values(KEY_SPECIES_ID).unique())
    for (src, species_id), row in data.species.iterrows():
        if species_id not in present:
            continue
        genus = row.get('genus')
        epithet = row.get('species')
        if not (isinstance(genus, str) and isinstance(epithet, str)):
            continue
        genus = genus.strip()
        epithet = epithet.strip()
        if not genus or not epithet:
            continue
        # USA-NPN stores epithets like 'persica' lowercase already, but be defensive.
        species_names[(src, species_id)] = f'{genus} {epithet.lower()}'
    return species_names
