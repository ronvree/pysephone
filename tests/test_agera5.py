"""
Tests for AgEra5Features.

All tests are self-contained — no network access, no HDF5 files, no real data
required.  AgEra5Features is always used in debug_mode=True.
"""

import numpy as np
import pandas as pd
import pytest

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_LAT,
    KEY_LOC_ID,
    KEY_LOC_NAME,
    KEY_LON,
    KEY_OBS_TYPE,
    KEY_OBSERVATIONS,
    KEY_SPECIES_ID,
    KEY_SUBGROUP_ID,
    KEY_YEAR,
    KEYS_INDEX,
)
from pysephone.data.agera5.download import (
    AgEra5Entry,
    _bbox_for_bundle,
    _bundle_entries,
    _tile_id,
)
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.agera5 import AgEra5Features
from pysephone.dataset.util.calendar import Calendar


_SRC = 'pep725'


def _make_df_y(rows: list) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        names=list(KEYS_INDEX) + [KEY_OBS_TYPE],
    )
    dates = [np.datetime64(r[6]) for r in rows]
    return pd.DataFrame({KEY_OBSERVATIONS: dates}, index=index)


def _make_df_y_loc(rows: list) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows],
        names=[KEY_DATA_SOURCE, KEY_LOC_ID],
    )
    return pd.DataFrame(
        {
            KEY_LAT: [r[2] for r in rows],
            KEY_LON: [r[3] for r in rows],
            KEY_LOC_NAME: [r[4] for r in rows],
        },
        index=index,
    )


@pytest.fixture
def df_y():
    return _make_df_y([
        (_SRC, 'loc_A', 2020, 333, 300, 'BBCH_0',  '2019-10-15'),
        (_SRC, 'loc_A', 2020, 333, 300, 'BBCH_51', '2020-05-10'),
        (_SRC, 'loc_B', 2020, 333, 300, 'BBCH_0',  '2019-10-20'),
        (_SRC, 'loc_B', 2020, 333, 300, 'BBCH_51', '2020-05-20'),
        (_SRC, 'loc_A', 2021, 333, 300, 'BBCH_0',  '2020-10-16'),
        (_SRC, 'loc_A', 2021, 333, 300, 'BBCH_51', '2021-05-11'),
    ])


@pytest.fixture
def df_y_loc():
    return _make_df_y_loc([
        (_SRC, 'loc_A', 51.0, 10.0, 'Station A'),
        (_SRC, 'loc_B', 52.0, 11.5, 'Station B'),
    ])


@pytest.fixture
def obs(df_y, df_y_loc):
    return Observations(df_y, df_y_loc)


@pytest.fixture
def calendar():
    cal = Calendar()
    cal.set_season(_SRC, species_id=333, subgroup_id=300,
                   start_date='10-01', length=365)
    return cal


@pytest.fixture
def features(calendar):
    return AgEra5Features(
        calendar=calendar,
        data_keys=['Temperature_Air_2m_Mean_24h', 'Solar_Radiation_Flux'],
        debug_mode=True,
    )


class TestAgEra5Features:

    def test_get_data_returns_dict(self, features, obs):
        ix = obs.index[0]
        result = features.get_data(ix)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'Temperature_Air_2m_Mean_24h', 'Solar_Radiation_Flux'}

    def test_get_data_arrays_are_ndarray(self, features, obs):
        ix = obs.index[0]
        result = features.get_data(ix)
        for arr in result.values():
            assert isinstance(arr, np.ndarray)

    def test_debug_arrays_are_zeros(self, features, obs):
        ix = obs.index[0]
        result = features.get_data(ix)
        for arr in result.values():
            assert np.all(arr == 0)

    def test_array_length_matches_season(self, calendar, obs):
        feats = AgEra5Features(
            calendar=calendar,
            data_keys=['Temperature_Air_2m_Mean_24h'],
            debug_mode=True,
        )
        ix = obs.index[0]
        src, loc_id, year, species_id, subgroup_id = ix
        info = calendar.get_season_info(year=year, src=src,
                                        species_id=species_id, subgroup_id=subgroup_id)
        expected_len = int(
            (info['season_end'] - info['season_start']) / np.timedelta64(1, 'D') - 1
        )
        arr = feats.get_variable('Temperature_Air_2m_Mean_24h', ix)
        assert abs(len(arr) - expected_len) <= 1

    def test_cache_size_increases(self, features, obs):
        assert features.cache_size() == 0
        ix = obs.index[0]
        features.get_data(ix)
        assert features.cache_size() > 0

    def test_clear_cache(self, features, obs):
        ix = obs.index[0]
        features.get_data(ix)
        features.clear_cache()
        assert features.cache_size() == 0

    def test_step_property(self, features):
        assert features.step == 'daily'

    def test_data_keys_property(self, features):
        assert 'Temperature_Air_2m_Mean_24h' in features.data_keys

    def test_default_data_keys(self, calendar):
        feats = AgEra5Features(calendar=calendar, debug_mode=True)
        assert 'Temperature_Air_2m_Mean_24h' in feats.data_keys
        assert 'Solar_Radiation_Flux' in feats.data_keys

    def test_context_manager(self, calendar):
        with AgEra5Features(calendar=calendar, debug_mode=True) as f:
            assert f.step == 'daily'
        assert f.cache_size() == 0


class TestAgEra5Entry:

    def test_loc_name_optional(self):
        # Sources without a loc_name column should not block entry creation.
        e = AgEra5Entry(
            data_key='Temperature_Air_2m_Mean_24h',
            src_key='gmu_cherry', loc_id=1,
            lat=35.0, lon=139.0, year=2020,
        )
        assert e.loc_name is None

    def test_entries_hash_ignores_loc_name(self):
        e_named   = AgEra5Entry(
            data_key='Temperature_Air_2m_Mean_24h',
            src_key='pep725', loc_id=1, lat=51.0, lon=10.0, year=2020,
            loc_name='Station A',
        )
        e_unnamed = AgEra5Entry(
            data_key='Temperature_Air_2m_Mean_24h',
            src_key='pep725', loc_id=1, lat=51.0, lon=10.0, year=2020,
            loc_name=None,
        )
        assert hash(e_named) == hash(e_unnamed)


def _entry(data_key='Temperature_Air_2m_Mean_24h', src='pep725', loc=1,
           lat=51.0, lon=10.0, year=2020):
    return AgEra5Entry(
        data_key=data_key, src_key=src, loc_id=loc,
        loc_name=f'station_{loc}', lat=lat, lon=lon, year=year,
    )


class TestTileBundling:

    def test_tile_id_close_points_share_tile(self):
        # Same 3° tile.
        assert _tile_id(51.0, 10.0, 3.0) == _tile_id(52.0, 11.5, 3.0)

    def test_tile_id_distant_points_differ(self):
        assert _tile_id(51.0, 10.0, 3.0) != _tile_id(36.0, 140.0, 3.0)

    def test_tile_id_southern_hemisphere(self):
        # floor() must keep negative latitudes in their own tiles
        assert _tile_id(-1.0, 0.0, 3.0) != _tile_id(1.0, 0.0, 3.0)

    def test_bundle_groups_by_year_and_tile(self):
        e1 = _entry(loc=1, lat=51.0, lon=10.0, year=2020)
        e2 = _entry(loc=2, lat=51.2, lon=10.1, year=2020)   # same tile + year
        e3 = _entry(loc=3, lat=51.0, lon=10.0, year=2021)   # same tile, diff year
        e4 = _entry(loc=4, lat=36.0, lon=140.0, year=2020)  # diff tile

        bundles = _bundle_entries({e1, e2, e3, e4}, tile_deg=3.0)
        # 3 distinct bundles: (var, 2020, EU-tile), (var, 2021, EU-tile), (var, 2020, JP-tile)
        assert len(bundles) == 3

        # The two co-located 2020 entries land in the same bundle
        eu_2020 = next(b for k, b in bundles.items() if k[1] == 2020 and len(b) == 2)
        assert {e.loc_id for e in eu_2020} == {1, 2}

    def test_bundle_merges_across_sources(self):
        # Two entries at the same location but different src_keys (e.g. PEP725_Apple
        # and PEP725_Pear) should land in the same bundle — that's the cross-source
        # dedup win.
        e_apple = _entry(src='pep725_apple', loc=1, lat=51.0, lon=10.0, year=2020)
        e_pear  = _entry(src='pep725_pear',  loc=1, lat=51.0, lon=10.0, year=2020)

        bundles = _bundle_entries({e_apple, e_pear}, tile_deg=3.0)
        assert len(bundles) == 1
        only = next(iter(bundles.values()))
        assert {e.src_key for e in only} == {'pep725_apple', 'pep725_pear'}

    def test_bundle_separates_by_variable(self):
        e_t = _entry(data_key='Temperature_Air_2m_Mean_24h', loc=1, lat=51.0, lon=10.0)
        e_r = _entry(data_key='Solar_Radiation_Flux',         loc=1, lat=51.0, lon=10.0)
        bundles = _bundle_entries({e_t, e_r}, tile_deg=3.0)
        assert len(bundles) == 2

    def test_bbox_padding(self):
        e1 = _entry(loc=1, lat=51.0, lon=10.0)
        e2 = _entry(loc=2, lat=52.0, lon=11.0)
        north, west, south, east = _bbox_for_bundle([e1, e2], pad_deg=0.2)
        assert north == pytest.approx(52.2)
        assert south == pytest.approx(50.8)
        assert east  == pytest.approx(11.2)
        assert west  == pytest.approx(9.8)

    def test_tile_deg_smaller_gives_more_bundles(self):
        e1 = _entry(loc=1, lat=51.0, lon=10.0)
        e2 = _entry(loc=2, lat=52.0, lon=11.0)
        # 3° tile: same bundle
        assert len(_bundle_entries({e1, e2}, tile_deg=3.0)) == 1
        # 0.5° tile: separate bundles
        assert len(_bundle_entries({e1, e2}, tile_deg=0.5)) == 2
