"""
Tests for DaylengthFeatures and the underlying compute_daylength helpers.

Self-contained: no network, no files, no real data required.
"""

import datetime as dt

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
    KEYS_INDEX,
)
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.daylength import (
    DaylengthFeatures,
    compute_daylength,
    compute_daylength_from_doy,
)


_SRC = 'pep725'

# Day-of-year for solstices in a non-leap year
_DOY_JUN_21 = 172
_DOY_DEC_21 = 355


# ----------------------------------------------------------------------
# Fixtures (pattern borrowed from tests/test_agera5.py)
# ----------------------------------------------------------------------


def _make_df_y(rows):
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        names=list(KEYS_INDEX) + [KEY_OBS_TYPE],
    )
    dates = [np.datetime64(r[6]) for r in rows]
    return pd.DataFrame({KEY_OBSERVATIONS: dates}, index=index)


def _make_df_y_loc(rows):
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
    ])


@pytest.fixture
def df_y_loc():
    return _make_df_y_loc([
        (_SRC, 'loc_A', 52.0, 10.0, 'Station A'),
        (_SRC, 'loc_B',  0.0, 11.5, 'Station B'),  # equator for sanity checks
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
def features(calendar, obs):
    f = DaylengthFeatures(calendar=calendar)
    f.preload(obs, verbose=False)
    return f


# ----------------------------------------------------------------------
# compute_daylength_from_doy: numerical correctness
# ----------------------------------------------------------------------


class TestComputeDaylengthFromDoy:

    def test_equator_is_about_twelve_hours(self):
        all_doy = np.arange(1, 366)
        d = compute_daylength_from_doy(0.0, all_doy)
        # Refraction adds ~7 min; values should be ~12.1 h year-round.
        assert np.all(np.abs(d - 12.1) < 0.2)

    def test_berlin_summer_solstice(self):
        # Lat 52° on Jun 21 → ~16.7 h
        d = float(compute_daylength_from_doy(52.0, _DOY_JUN_21))
        assert abs(d - 16.7) < 0.3

    def test_berlin_winter_solstice(self):
        # Lat 52° on Dec 21 → ~7.7 h
        d = float(compute_daylength_from_doy(52.0, _DOY_DEC_21))
        assert abs(d - 7.7) < 0.3

    def test_polar_day(self):
        # Lat 80°N on Jun 21 → 24 h, no NaN
        d = float(compute_daylength_from_doy(80.0, _DOY_JUN_21))
        assert d == pytest.approx(24.0, abs=1e-3)
        assert not np.isnan(d)

    def test_polar_night(self):
        # Lat 80°N on Dec 21 → 0 h, no NaN
        d = float(compute_daylength_from_doy(80.0, _DOY_DEC_21))
        assert d == pytest.approx(0.0, abs=1e-3)
        assert not np.isnan(d)

    def test_hemisphere_symmetry_at_summer_solstice(self):
        # Northern summer at +52° pairs with southern winter at -52°.
        # The pair sums to slightly more than 24 h because the 0.8333°
        # refraction coefficient adds ~7 min to both sunrise and sunset on
        # both sides (≈ +0.5 h total).
        d_north = float(compute_daylength_from_doy(+52.0, _DOY_JUN_21))
        d_south = float(compute_daylength_from_doy(-52.0, _DOY_JUN_21))
        assert 24.0 < (d_north + d_south) < 24.7

    def test_returns_float32(self):
        d = compute_daylength_from_doy(45.0, np.arange(1, 366))
        assert d.dtype == np.float32

    def test_scalar_doy_returns_scalar(self):
        # np.asarray + .astype on a Python int yields a numpy scalar
        # (a 0-d array or np.floating instance — both acceptable).
        d = compute_daylength_from_doy(45.0, 100)
        assert isinstance(d, (np.ndarray, np.floating))
        assert np.asarray(d).shape == ()

    def test_array_shape_preserved(self):
        doy = np.arange(1, 366)
        d = compute_daylength_from_doy(45.0, doy)
        assert d.shape == doy.shape

    def test_no_nans_across_full_year_and_latitudes(self):
        doy = np.arange(1, 367)  # include leap day
        for lat in (-89.0, -66.0, -45.0, 0.0, 45.0, 66.0, 89.0):
            d = compute_daylength_from_doy(lat, doy)
            assert not np.any(np.isnan(d)), f"NaN at lat={lat}"
            assert np.all(d >= 0.0)
            assert np.all(d <= 24.0)


# ----------------------------------------------------------------------
# compute_daylength: input-type dispatch
# ----------------------------------------------------------------------


class TestComputeDaylengthDispatch:

    EXPECTED_BERLIN_JUN_21 = 16.7  # hours, ±0.3 tolerance

    def test_python_date(self):
        d = compute_daylength(52.0, dt.date(2024, 6, 21))
        assert isinstance(d, float)
        assert abs(d - self.EXPECTED_BERLIN_JUN_21) < 0.3

    def test_python_datetime(self):
        d = compute_daylength(52.0, dt.datetime(2024, 6, 21, 12, 0))
        assert isinstance(d, float)
        assert abs(d - self.EXPECTED_BERLIN_JUN_21) < 0.3

    def test_numpy_datetime64_scalar(self):
        d = compute_daylength(52.0, np.datetime64('2024-06-21'))
        assert isinstance(d, float)
        assert abs(d - self.EXPECTED_BERLIN_JUN_21) < 0.3

    def test_numpy_datetime64_array(self):
        dates = np.array(['2024-06-21', '2024-12-21'], dtype='datetime64[D]')
        d = compute_daylength(52.0, dates)
        assert isinstance(d, np.ndarray)
        assert d.shape == (2,)
        assert abs(d[0] - 16.7) < 0.3
        assert abs(d[1] - 7.7) < 0.3

    def test_pandas_datetime_index(self):
        idx = pd.date_range('2024-06-21', periods=3, freq='D')
        d = compute_daylength(52.0, idx)
        assert isinstance(d, np.ndarray)
        assert d.shape == (3,)

    def test_int_doy(self):
        d = compute_daylength(52.0, _DOY_JUN_21)
        assert isinstance(d, float)
        assert abs(d - self.EXPECTED_BERLIN_JUN_21) < 0.3

    def test_int_array_doy(self):
        d = compute_daylength(52.0, np.array([_DOY_JUN_21, _DOY_DEC_21]))
        assert isinstance(d, np.ndarray)
        assert d.shape == (2,)

    def test_list_of_dates(self):
        d = compute_daylength(52.0, ['2024-06-21', '2024-12-21'])
        assert isinstance(d, np.ndarray)
        assert d.shape == (2,)
        assert abs(d[0] - 16.7) < 0.3

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            compute_daylength(52.0, 3.14)

    def test_date_and_doy_agree(self):
        # Same instant via three paths → same value
        from_date = compute_daylength(45.0, dt.date(2023, 6, 21))
        from_dt64 = compute_daylength(45.0, np.datetime64('2023-06-21'))
        from_doy = compute_daylength(45.0, _DOY_JUN_21)
        assert abs(from_date - from_dt64) < 1e-3
        assert abs(from_date - from_doy) < 1e-3


# ----------------------------------------------------------------------
# DaylengthFeatures provider
# ----------------------------------------------------------------------


class TestDaylengthFeatures:

    def test_get_data_returns_dict(self, features, obs):
        ix = obs.index[0]
        result = features.get_data(ix)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'daylength'}

    def test_get_data_returns_float32_ndarray(self, features, obs):
        ix = obs.index[0]
        arr = features.get_data(ix)['daylength']
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_array_length_matches_season_length(self, features, obs, calendar):
        ix = obs.index[0]
        src, _loc, year, sp, sg = ix
        info = calendar.get_season_info(year=year, src=src,
                                        species_id=sp, subgroup_id=sg)
        expected_len = int(info['season_length'] / np.timedelta64(1, 'D'))
        arr = features.get_data(ix)['daylength']
        assert len(arr) == expected_len

    def test_equator_values_are_about_twelve(self, features, obs):
        # loc_B is at the equator in the fixture
        ix = next(ix for ix in obs.iter_index() if ix[1] == 'loc_B')
        arr = features.get_data(ix)['daylength']
        assert np.all(np.abs(arr - 12.1) < 0.2)

    def test_get_data_before_preload_raises(self, calendar, obs):
        f = DaylengthFeatures(calendar=calendar)  # no preload
        ix = obs.index[0]
        with pytest.raises(RuntimeError, match='no latitude loaded'):
            f.get_data(ix)

    def test_custom_key(self, calendar, obs):
        f = DaylengthFeatures(calendar=calendar, key='photoperiod_h')
        f.preload(obs, verbose=False)
        assert set(f.get_data(obs.index[0]).keys()) == {'photoperiod_h'}

    def test_step_property(self, features):
        assert features.step == 'daily'

    def test_context_manager(self, calendar, obs):
        with DaylengthFeatures(calendar=calendar) as f:
            f.preload(obs, verbose=False)
            assert f.get_data(obs.index[0])['daylength'].size > 0

    def test_no_download_method(self, features):
        # Provider has no download() — Dataset.download_features must tolerate this.
        assert not hasattr(features, 'download')
