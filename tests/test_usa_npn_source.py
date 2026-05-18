"""
Tests for USANPNSource.

Network access is mocked — only the in-memory build pipeline is exercised
(catalogue filtering → phenometrics shaping → ObservationData assembly).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from pysephone.constants import KEYS_INDEX, KEY_OBS_TYPE, KEY_OBSERVATIONS
from pysephone.data.source import ObservationData
from pysephone.data.usa_npn.source import USANPNSource


_MODULE = 'pysephone.data.usa_npn.source'


# A tiny species catalogue stub: two genera the user might filter on.
_STUB_SPECIES = pd.DataFrame({
    'species_id':  [80, 102, 999],
    'genus':       ['Juglans', 'Quercus', 'Pinus'],
    'species':     ['nigra', 'rubra', 'strobus'],
    'common_name': ['black walnut', 'northern red oak', 'eastern white pine'],
    'family_name': ['Juglandaceae', 'Fagaceae', 'Pinaceae'],
    'kingdom':     ['Plantae', 'Plantae', 'Plantae'],
})


def _stub_phenometrics_frame(
    species_id: int, individual_id: int, site_id: int,
    phenophase_id: int, phenophase_description: str,
    year: int, month: int, day: int,
    lat: float, lon: float, alt: float, state: str,
    genus: str, species: str, common_name: str,
):
    return {
        'site_id':              site_id,
        'latitude':             lat,
        'longitude':            lon,
        'elevation_in_meters':  alt,
        'state':                state,
        'species_id':           species_id,
        'genus':                genus,
        'species':              species,
        'common_name':          common_name,
        'kingdom':              'Plantae',
        'individual_id':        individual_id,
        'phenophase_id':        phenophase_id,
        'phenophase_description': phenophase_description,
        'first_yes_year':       year,
        'first_yes_month':      month,
        'first_yes_day':        day,
    }


# Two phenophases, two sites, three individuals; one row uses a -9999 sentinel
# month and must be dropped.
_STUB_FLOWERS = pd.DataFrame([
    _stub_phenometrics_frame(80, 449, 720, 501, 'Open flowers',
                              2021, 4, 19, 38.53, -77.23, 12, 'MD',
                              'Juglans', 'nigra', 'black walnut'),
    _stub_phenometrics_frame(80, 450, 720, 501, 'Open flowers',
                              2022, 4, 25, 38.53, -77.23, 12, 'MD',
                              'Juglans', 'nigra', 'black walnut'),
    _stub_phenometrics_frame(80, 451, 999, 501, 'Open flowers',
                              2023, -9999, -9999, 40.0, -75.0, -9999, 'PA',
                              'Juglans', 'nigra', 'black walnut'),
])

_STUB_LEAFBUDS = pd.DataFrame([
    _stub_phenometrics_frame(80, 449, 720, 371, 'Breaking leaf buds',
                              2021, 3, 30, 38.53, -77.23, 12, 'MD',
                              'Juglans', 'nigra', 'black walnut'),
])


def _no_network_fetch_species(*args, **kwargs):
    return _STUB_SPECIES.copy()


def _no_network_fetch_all(*args, **kwargs):
    return {501: _STUB_FLOWERS.copy(), 371: _STUB_LEAFBUDS.copy()}


@pytest.fixture
def source():
    return USANPNSource()


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Smoke tests: ObservationData shape
# ---------------------------------------------------------------------------

@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_get_data_returns_observation_data(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    assert isinstance(result, ObservationData)


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_observations_index_matches_schema(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    assert result.observations.index.names == [*KEYS_INDEX, KEY_OBS_TYPE]
    assert KEY_OBSERVATIONS in result.observations.columns
    # 3 input flower rows -> 1 dropped (sentinel) -> 2; plus 1 leafbud row.
    assert len(result.observations) == 3


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_obs_type_uses_npn_phenophase_naming(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    obs_types = set(result.observations.index.get_level_values('obs_type'))
    assert obs_types == {'NPN_501', 'NPN_371'}


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_events_table_indexed_by_src_event(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    assert result.events.index.names == ['src', 'event']
    assert set(result.events.index.get_level_values('event')) == {'NPN_501', 'NPN_371'}
    assert 'description' in result.events.columns


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_locations_table_shape(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    assert result.locations.index.names == ['src', 'loc_id']
    assert {'lat', 'lon'}.issubset(result.locations.columns)
    # 720 (good) + 999 (sentinel-elev kept; sentinel in month was dropped)
    assert set(result.locations.index.get_level_values('loc_id')) == {720, 999}
    # NPN_MISSING elevation should be NaN, not -9999
    alts = result.locations['alt'].dropna().tolist()
    assert -9999 not in alts


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_subgroups_link_to_species(mock_species, mock_pheno, source, tmp_root):
    result = source.get_data({'genera': ['Juglans']}, root=tmp_root)
    assert result.subgroups.index.names == ['src', 'subgroup_id']
    assert (result.subgroups['species_id'] == 80).all()


# ---------------------------------------------------------------------------
# cfg semantics
# ---------------------------------------------------------------------------

@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_genera_filter_passed_to_phenometrics_call(mock_species, mock_pheno, source, tmp_root):
    source.get_data({'genera': ['Juglans']}, root=tmp_root)
    species_ids = mock_pheno.call_args.kwargs['species_ids']
    assert species_ids == [80]


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_explicit_species_ids_override_genera(mock_species, mock_pheno, source, tmp_root):
    source.get_data({'genera': ['Juglans'], 'species_ids': [102]}, root=tmp_root)
    species_ids = mock_pheno.call_args.kwargs['species_ids']
    assert species_ids == [102]


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_default_phenophases_are_501_and_371(mock_species, mock_pheno, source, tmp_root):
    source.get_data({'genera': ['Juglans']}, root=tmp_root)
    phenophases = mock_pheno.call_args.kwargs['phenophase_ids']
    assert tuple(phenophases) == (501, 371)


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_force_download_propagates(mock_species, mock_pheno, source, tmp_root):
    source.get_data({'genera': ['Juglans'], 'force_download': True}, root=tmp_root)
    assert mock_species.call_args.kwargs['force_download'] is True
    assert mock_pheno.call_args.kwargs['force_download']    is True


@patch(f'{_MODULE}.fetch_all_phenometrics', side_effect=_no_network_fetch_all)
@patch(f'{_MODULE}.fetch_species_table',    side_effect=_no_network_fetch_species)
def test_empty_filter_raises(mock_species, mock_pheno, source, tmp_root):
    with pytest.raises(ValueError):
        source.get_data({'genera': ['Doesnotexist']}, root=tmp_root)


# ---------------------------------------------------------------------------
# Cache path stability — paranoia check that the hash matches the notebook's.
# ---------------------------------------------------------------------------

def test_cache_path_hash_matches_notebook_convention(tmp_root):
    import hashlib
    from pysephone.data.usa_npn.download import path_phenometrics_cache

    species = [80, 102]
    phenophase = 501
    start, end = '2009-01-01', '2024-12-31'

    expected_sha = hashlib.sha1(
        f'{sorted(species)}|{phenophase}|{start}|{end}'.encode()
    ).hexdigest()[:10]

    p = path_phenometrics_cache(tmp_root, species, phenophase, start, end)
    assert p.name == (
        f'individual_phenometrics_{start}_{end}_p{phenophase}_{expected_sha}.csv'
    )
