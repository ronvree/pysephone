"""
PEP725 dataset definitions.

Each public builder returns an Observations object for one pre-configured
dataset variant.  Shared configuration lives at module level so it is easy
to spot and adjust without touching individual builders.
"""

from datetime import datetime
from functools import reduce
from typing import List

from pysephone.data.pep725.source import PEP725Source
from pysephone.dataset.observations import Observations
from pysephone.dataset.preprocessing.pep725 import get_pep725_dataframes

_SRC = PEP725Source.KEY  # 'pep725'

# ---------------------------------------------------------------------------
# Shared configuration blocks
# ---------------------------------------------------------------------------

# Crop Phenology Framework (CPF)
_CPF_YEARS: List[int] = list(range(1986, 2016))
_CPF_REMOVE_OUTLIERS: bool = True
_CPF_DO_AGG: bool = True
_CPF_AGG_METHOD: str = 'median'

# General benchmark (bm)
_BM_YEAR_MIN: int = 1986
_BM_YEAR_MAX: int = datetime.now().year - 1
_BM_YEARS: List[int] = list(range(_BM_YEAR_MIN, _BM_YEAR_MAX + 1))
_BM_DO_AGG: bool = True
_BM_AGG_METHOD: str = 'median'
_BM_REMOVE_OUTLIERS: bool = True
_BM_ASSERT_TARGET: bool = True


def _pep725_data():
    """Load raw PEP725 ObservationData (reads from disk / triggers download)."""
    return PEP725Source().get_data({})


# ---------------------------------------------------------------------------
# Helper: build a PEP725 observation dataset with benchmark defaults
# ---------------------------------------------------------------------------

def _build_bm(species_subgroups, obs_types: list, target_obs: str) -> Observations:
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=species_subgroups,
        remove_outliers=_BM_REMOVE_OUTLIERS,
        filter_on_observation_types=obs_types,
        filter_on_years=_BM_YEARS,
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'], species_names=dfs.get('species_names'))
    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement(target_obs)
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


def _cpf_obs_from_dfs(dfs: dict, require: list, shift_year_key: str = None) -> Observations:
    """Build a CPF Observations from get_pep725_dataframes output."""
    df_y = dfs['data']
    if shift_year_key:
        df_y = Observations.shift_year(df_y, shift_year_key, 1)
    obs = Observations(df_y, dfs['locations'], species_names=dfs.get('species_names'))
    obs = obs.select_by_observation_requirement(require)
    if _CPF_DO_AGG:
        obs = obs.aggregate_in_grid(method=_CPF_AGG_METHOD)
    return obs


# ---------------------------------------------------------------------------
# CPF winter crops
# ---------------------------------------------------------------------------

def build_test_dataset(**kwargs) -> Observations:
    # Same species as CPF_PEP725_winter_wheat, used for quick smoke-tests
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=(_SRC, 333, 300),
        remove_outliers=_CPF_REMOVE_OUTLIERS,
        filter_on_observation_types=['BBCH_0', 'BBCH_51'],
        filter_on_years=_CPF_YEARS,
        datetime_observations=True,
    )
    return _cpf_obs_from_dfs(dfs, ['BBCH_0', 'BBCH_51'], shift_year_key='BBCH_0')


def build_cpf_winter_wheat(**kwargs) -> Observations:
    # Triticum aestivum — species 333, subgroup 300
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=(_SRC, 333, 300),
        remove_outliers=_CPF_REMOVE_OUTLIERS,
        filter_on_observation_types=['BBCH_0', 'BBCH_51'],
        filter_on_years=_CPF_YEARS,
        datetime_observations=True,
    )
    return _cpf_obs_from_dfs(dfs, ['BBCH_0', 'BBCH_51'], shift_year_key='BBCH_0')


def build_cpf_winter_barley(**kwargs) -> Observations:
    # Hordeum vulgare — species 330, subgroup 300
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=(_SRC, 330, 300),
        remove_outliers=_CPF_REMOVE_OUTLIERS,
        filter_on_observation_types=['BBCH_0', 'BBCH_51'],
        filter_on_years=_CPF_YEARS,
        datetime_observations=True,
    )
    return _cpf_obs_from_dfs(dfs, ['BBCH_0', 'BBCH_51'], shift_year_key='BBCH_0')


def build_cpf_winter_rye(**kwargs) -> Observations:
    # Secale cereale — species 332, subgroup 300
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=(_SRC, 332, 300),
        remove_outliers=_CPF_REMOVE_OUTLIERS,
        filter_on_observation_types=['BBCH_0', 'BBCH_61'],
        filter_on_years=_CPF_YEARS,
        datetime_observations=True,
    )
    return _cpf_obs_from_dfs(dfs, ['BBCH_0', 'BBCH_61'], shift_year_key='BBCH_0')


# ---------------------------------------------------------------------------
# PEP725 fruit trees (benchmark)
# ---------------------------------------------------------------------------

def build_pep725_apple(**kwargs) -> Observations:
    # Malus x Domestica — species 220
    species_subgroups = [
        (_SRC, 220, 100),  # Early cultivar
        (_SRC, 220, 130),  # Late cultivar
        (_SRC, 220, 115),  # Middle cultivar
        (_SRC, 220, 433),  # Cox Orange Renette
        (_SRC, 220, 508),  # Elstar
        (_SRC, 220, 437),  # Golden Delicious
        (_SRC, 220, 430),  # Goldparm
        (_SRC, 220, 438),  # Gravensteiner
        (_SRC, 220, 509),  # Idared
        (_SRC, 220, 500),  # James Grieve
        (_SRC, 220, 501),  # Jonagold
        (_SRC, 220, 510),  # Jonathan
        (_SRC, 220, 503),  # Roter Boskoop
        (_SRC, 220, 506),  # Wei
        (_SRC, 220, 615),  # Granny Smith
        (_SRC, 220, 617),  # Bobovec
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_69', 'BBCH_87'], 'BBCH_60')


def build_pep725_pear(**kwargs) -> Observations:
    # Pyrus Communis — species 227
    species_subgroups = [
        (_SRC, 227, 100),  # Early cultivar
        (_SRC, 227, 130),  # Late cultivar
        (_SRC, 227, 590),  # Williams
        (_SRC, 227, 586),  # Bunte Julibirne
        (_SRC, 227, 585),  # Jakob
        (_SRC, 227, 587),  # Junsko Zlato
        (_SRC, 227, 589),  # Karamanka
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'], 'BBCH_60')


def build_pep725_peach(**kwargs) -> Observations:
    # Prunus Persica — species 202
    species_subgroups = [
        (_SRC, 202, 0),    # No group
        (_SRC, 202, 579),  # Alberta
        (_SRC, 202, 580),  # Dixired
        (_SRC, 202, 581),  # Hale
        (_SRC, 202, 578),  # Red Haven
        (_SRC, 202, 582),  # Springtime
    ]
    return _build_bm(species_subgroups, ['BBCH_60'], 'BBCH_60')


def build_pep725_almond(**kwargs) -> Observations:
    # Prunus Amygdalis — species 782
    species_subgroups = [
        (_SRC, 782, 0),  # No group
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'], 'BBCH_60')


def build_pep725_hazel(**kwargs) -> Observations:
    # Corylus Avellana — species 107
    species_subgroups = [
        (_SRC, 107, 0),  # No group
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_86'], 'BBCH_60')


def build_pep725_cherry(**kwargs) -> Observations:
    # Prunus Avium — species 222
    species_subgroups = [
        (_SRC, 222, 0),    # No group
        (_SRC, 222, 100),  # Early cultivar
        (_SRC, 222, 130),  # Late cultivar
        (_SRC, 222, 494),  # Regina
        (_SRC, 222, 495),  # Schwarze Knorpelkirsch
        (_SRC, 222, 618),  # Majska rana
        (_SRC, 222, 602),  # Germersdorfer
        (_SRC, 222, 603),  # Hedelfinger
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'], 'BBCH_60')


def build_pep725_cherry_no_gmu(match_radius_km: float = 5.0, **kwargs) -> Observations:
    """PEP725 Cherry (*Prunus avium*) with stations overlapping GMU Swiss
    cherry removed.

    Any PEP725 station whose nearest *GMU_Cherry_Switzerland* station is
    within ``match_radius_km`` (default 5 km) is dropped, deduplicating the
    same-tree overlap flagged by ``notebooks/gmu_pep725_cherry_overlap``.
    Only GMU Switzerland is considered — GMU Japan / South Korea are
    *P. yedoensis*, so no cross-species confusion is possible.
    """
    import numpy as np
    from pysephone.dataset.registry.gmu_cherry import build_gmu_cherry_switzerland

    pep_obs = build_pep725_cherry()
    gmu_obs = build_gmu_cherry_switzerland()

    gmu_locs = list(gmu_obs.locations)
    pep_locs = list(pep_obs.locations)
    if not gmu_locs or not pep_locs:
        return pep_obs

    gmu_lats = np.array([gmu_obs.get_location_coordinates(l)['lat'] for l in gmu_locs])
    gmu_lons = np.array([gmu_obs.get_location_coordinates(l)['lon'] for l in gmu_locs])
    pep_lats = np.array([pep_obs.get_location_coordinates(l)['lat'] for l in pep_locs])
    pep_lons = np.array([pep_obs.get_location_coordinates(l)['lon'] for l in pep_locs])

    R = 6371.0  # km
    lat1 = np.radians(pep_lats)[:, None]
    lat2 = np.radians(gmu_lats)[None, :]
    dlat = lat2 - lat1
    dlon = np.radians(gmu_lons[None, :] - pep_lons[:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))

    keep_mask = dist_km.min(axis=1) > match_radius_km
    kept = [loc for loc, k in zip(pep_locs, keep_mask) if k]
    return pep_obs.select_locations(kept)


def build_pep725_apricot(**kwargs) -> Observations:
    # Prunus Armeniaca — species 205
    species_subgroups = [
        (_SRC, 205, 0),  # No group
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_87'], 'BBCH_60')


def build_pep725_plum(**kwargs) -> Observations:
    # Prunus Domestica — species 225
    species_subgroups = [
        (_SRC, 225, 0),    # No group
        (_SRC, 225, 100),  # Early cultivar
        (_SRC, 225, 130),  # Late cultivar
        (_SRC, 225, 621),  # Besztercei
        (_SRC, 225, 595),  # Bosankska
        (_SRC, 225, 596),  # Dzanarika
        (_SRC, 225, 597),  # Pozegaca
        (_SRC, 225, 612),  # Renkloda
        (_SRC, 225, 614),  # Stanlay
    ]
    return _build_bm(species_subgroups, ['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'], 'BBCH_60')


def build_pep725_blackthorn(**kwargs) -> Observations:
    # Prunus Spinosa — species 123
    species_subgroups = [
        (_SRC, 123, 0),
    ]
    return _build_bm(species_subgroups, ['BBCH_60'], 'BBCH_60')


def build_pep725_oak(**kwargs) -> Observations:
    # Quercus robur — species 111
    species_subgroups = [
        (_SRC, 111, 0),  # No group
    ]
    return _build_bm(species_subgroups, ['BBCH_11', 'BBCH_94', 'BBCH_86', 'BBCH_95'], 'BBCH_11')


def build_cfm_zea_mays(**kwargs) -> Observations:
    # Zea mays (maize/corn) — species 440, subgroup 0
    dfs = get_pep725_dataframes(
        _pep725_data(),
        filter_on_species=(_SRC, 440, 0),
        remove_outliers=True,
        filter_on_observation_types=['BBCH_0', 'BBCH_51'],
        filter_on_years=list(range(1980, 2025)),
        datetime_observations=True,
    )
    obs = Observations(dfs['data'], dfs['locations'], species_names=dfs.get('species_names'))
    obs = obs.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])
    # No grid aggregation applied for this dataset
    return obs


# ---------------------------------------------------------------------------
# Composite fruit tree collections (use lazy loading to avoid redundant I/O)
# ---------------------------------------------------------------------------

def build_pep725_fruit_trees(**kwargs) -> Observations:
    keys = [
        'PEP725_Apple',
        'PEP725_Pear',
        'PEP725_Peach',
        'PEP725_Almond',
        'PEP725_Hazel',
        'PEP725_Cherry',
        'PEP725_Apricot',
        'PEP725_Plum',
        'PEP725_Blackthorn',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_pep725_fruit_trees_2(**kwargs) -> Observations:
    # Subset represented in Germany (Apricot excluded)
    keys = [
        'PEP725_Apple',
        'PEP725_Pear',
        'PEP725_Peach',
        'PEP725_Almond',
        'PEP725_Hazel',
        'PEP725_Cherry',
        # 'PEP725_Apricot',
        'PEP725_Plum',
        'PEP725_Blackthorn',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_pep725_fruit_trees_3(**kwargs) -> Observations:
    # Subset with cultivar information
    keys = [
        'PEP725_Apple',
        'PEP725_Pear',
        # 'PEP725_Peach',
        'PEP725_Plum',
        'PEP725_Cherry',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_pep725_fruit_trees_4(**kwargs) -> Observations:
    # Subset without Hazel
    keys = [
        'PEP725_Apple',
        'PEP725_Pear',
        'PEP725_Peach',
        'PEP725_Almond',
        # 'PEP725_Hazel',
        'PEP725_Cherry',
        'PEP725_Apricot',
        'PEP725_Plum',
        'PEP725_Blackthorn',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_pep725_fruit_trees_5(**kwargs) -> Observations:
    # Core subset: Apple, Pear, Peach, Cherry, Plum, Blackthorn
    keys = [
        'PEP725_Apple',
        'PEP725_Pear',
        'PEP725_Peach',
        # 'PEP725_Almond',
        # 'PEP725_Hazel',
        'PEP725_Cherry',
        # 'PEP725_Apricot',
        'PEP725_Plum',
        'PEP725_Blackthorn',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


# ---------------------------------------------------------------------------
# Calendar configuration: per-dataset season window defaults
# ---------------------------------------------------------------------------

def _pep725_configure_calendar_winter_wheat(calendar):
    calendar.set_season(_SRC, species_id=333, start_date='09-01', length=365, subgroup_id=300)

def _pep725_configure_calendar_winter_barley(calendar):
    calendar.set_season(_SRC, species_id=330, start_date='09-01', length=365, subgroup_id=300)

def _pep725_configure_calendar_winter_rye(calendar):
    calendar.set_season(_SRC, species_id=332, start_date='09-01', length=365, subgroup_id=300)


CALENDAR_CONFIGS = {
    'test_dataset':             _pep725_configure_calendar_winter_wheat,
    'CPF_PEP725_winter_wheat':  _pep725_configure_calendar_winter_wheat,
    'CPF_PEP725_winter_barley': _pep725_configure_calendar_winter_barley,
    'CPF_PEP725_winter_rye':    _pep725_configure_calendar_winter_rye,
}


# ---------------------------------------------------------------------------
# Registry mapping for PEP725 datasets
# ---------------------------------------------------------------------------

DATASETS = {
    # Test / smoke-test
    'test_dataset': build_test_dataset,

    # CPF winter crops
    'CPF_PEP725_winter_wheat':  build_cpf_winter_wheat,
    'CPF_PEP725_winter_barley': build_cpf_winter_barley,
    'CPF_PEP725_winter_rye':    build_cpf_winter_rye,

    # PEP725 individual fruit trees
    'PEP725_Apple':      build_pep725_apple,
    'PEP725_Pear':       build_pep725_pear,
    'PEP725_Peach':      build_pep725_peach,
    'PEP725_Almond':     build_pep725_almond,
    'PEP725_Hazel':      build_pep725_hazel,
    'PEP725_Cherry':        build_pep725_cherry,
    'PEP725_Cherry_NoGMU':  build_pep725_cherry_no_gmu,
    'PEP725_Apricot':    build_pep725_apricot,
    'PEP725_Plum':       build_pep725_plum,
    'PEP725_Blackthorn': build_pep725_blackthorn,
    'PEP725_Oak':        build_pep725_oak,

    # PEP725 composite fruit tree collections
    'PEP725_fruit_trees':   build_pep725_fruit_trees,
    'PEP725_fruit_trees_2': build_pep725_fruit_trees_2,
    'PEP725_fruit_trees_3': build_pep725_fruit_trees_3,
    'PEP725_fruit_trees_4': build_pep725_fruit_trees_4,
    'PEP725_fruit_trees_5': build_pep725_fruit_trees_5,

    # CFM crop
    'CFM_zea_mays': build_cfm_zea_mays,
}
