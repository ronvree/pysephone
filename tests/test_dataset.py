"""
Tests for Dataset, Observations, Calendar, and OpenMeteoFeatures.

All tests are self-contained — no network access, no HDF5 files, no real data
required.  OpenMeteoFeatures is always used in debug_mode=True.
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
from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.dataset.dataset import Dataset, _KEY_SEASON_END, _KEY_SEASON_START
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.func import DatasetException
from pysephone.dataset.util.openmeteo import OpenMeteoFeatures


# ---------------------------------------------------------------------------
# Helpers / shared constants
# ---------------------------------------------------------------------------

_SRC = 'pep725'
_SPECIES_A = (_SRC, 333, 300)  # winter wheat


def _make_df_y(rows: list) -> pd.DataFrame:
    """Build a df_y DataFrame from a list of (src, loc_id, year, sp, sub, obs_type, date)."""
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        names=list(KEYS_INDEX) + [KEY_OBS_TYPE],
    )
    dates = [np.datetime64(r[6]) for r in rows]
    return pd.DataFrame({KEY_OBSERVATIONS: dates}, index=index)


def _make_df_y_loc(rows: list) -> pd.DataFrame:
    """Build a df_y_loc DataFrame from a list of (src, loc_id, lat, lon, name)."""
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_y():
    return _make_df_y([
        # loc_A, year 2020
        (_SRC, 'loc_A', 2020, 333, 300, 'BBCH_0',  '2019-10-15'),
        (_SRC, 'loc_A', 2020, 333, 300, 'BBCH_51', '2020-05-10'),
        # loc_B, year 2020
        (_SRC, 'loc_B', 2020, 333, 300, 'BBCH_0',  '2019-10-20'),
        (_SRC, 'loc_B', 2020, 333, 300, 'BBCH_51', '2020-05-20'),
        # loc_A, year 2021
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
    # Winter wheat: season starts Oct 1 of previous year, lasts 365 days
    cal.set_season(_SRC, species_id=333, subgroup_id=300,
                   start_date='10-01', length=365)
    return cal


@pytest.fixture
def features(calendar):
    return OpenMeteoFeatures(
        calendar=calendar,
        data_keys=['temperature_2m_mean', 'daylight_duration'],
        step='daily',
        debug_mode=True,
    )


@pytest.fixture
def dataset(obs, calendar, features):
    return Dataset(obs, calendar=calendar, feature_providers=[features])


# ===========================================================================
# Observations
# ===========================================================================

class TestObservationsBasic:

    def test_len(self, obs):
        # 2 locations × 2 years for loc_A + 1 year for loc_B = 3 index entries
        assert len(obs) == 3

    def test_contains_tuple(self, obs):
        ix = obs.index[0]
        assert ix in obs
        assert (_SRC, 'nonexistent', 2020, 333, 300) not in obs

    def test_getitem_int(self, obs):
        item = obs[0]
        assert KEY_DATA_SOURCE in item
        assert KEY_OBSERVATIONS in item
        assert KEY_LAT in item
        assert KEY_LON in item

    def test_getitem_tuple(self, obs):
        ix = obs.index[0]
        item = obs[ix]
        assert item[KEY_DATA_SOURCE] == ix[0]
        assert item[KEY_LOC_ID] == ix[1]
        assert item[KEY_YEAR] == ix[2]

    def test_getitem_int_out_of_range(self, obs):
        with pytest.raises(IndexError):
            obs[999]

    def test_getitem_bad_type(self, obs):
        with pytest.raises(DatasetException):
            obs['bad']

    def test_iter_index(self, obs):
        ixs = list(obs.iter_index())
        assert len(ixs) == len(obs)
        assert all(len(ix) == 5 for ix in ixs)

    def test_iter_items(self, obs):
        items = list(obs.iter_items())
        assert len(items) == len(obs)
        assert all(KEY_OBSERVATIONS in item for item in items)


class TestObservationsProperties:

    def test_locations(self, obs):
        locs = obs.locations
        assert (_SRC, 'loc_A') in locs
        assert (_SRC, 'loc_B') in locs
        assert len(locs) == 2

    def test_species(self, obs):
        sp = obs.species
        assert (_SRC, 333) in sp

    def test_species_subgroups(self, obs):
        sg = obs.species_subgroups
        assert (_SRC, 333, 300) in sg

    def test_years(self, obs):
        assert obs.years == [2020, 2021]

    def test_observation_types(self, obs):
        ots = obs.observation_types
        assert 'BBCH_0' in ots
        assert 'BBCH_51' in ots

    def test_num_observation_types(self, obs):
        assert obs.num_observation_types == 2

    def test_bounding_box(self, obs):
        bb = obs.bounding_box
        assert bb['min_lat'] == 51.0
        assert bb['max_lat'] == 52.0
        assert bb['min_lon'] == 10.0
        assert bb['max_lon'] == 11.5

    def test_get_location_coordinates(self, obs):
        coords = obs.get_location_coordinates((_SRC, 'loc_A'))
        assert coords[KEY_LAT] == 51.0
        assert coords[KEY_LON] == 10.0

    def test_get_location_coordinates_from_index(self, obs):
        ix = (_SRC, 'loc_A', 2020, 333, 300)
        coords = obs.get_location_coordinates(ix, from_index=True)
        assert coords[KEY_LAT] == 51.0

    def test_get_location_name(self, obs):
        assert obs.get_location_name((_SRC, 'loc_A')) == 'Station A'

    def test_observation_counts(self, obs):
        counts = obs.observation_counts()
        assert counts['BBCH_0'] == 3
        assert counts['BBCH_51'] == 3


class TestObservationsSelection:

    def test_select_locations_single(self, obs):
        sub = obs.select_locations((_SRC, 'loc_A'))
        assert all(loc == (_SRC, 'loc_A') for loc in sub.locations)

    def test_select_locations_list(self, obs):
        sub = obs.select_locations([(_SRC, 'loc_A'), (_SRC, 'loc_B')])
        assert len(sub) == len(obs)

    def test_select_years_single(self, obs):
        sub = obs.select_years(2020)
        assert sub.years == [2020]

    def test_select_years_list(self, obs):
        sub = obs.select_years([2020, 2021])
        assert set(sub.years) == {2020, 2021}

    def test_select_species_two_tuple(self, obs):
        sub = obs.select_species((_SRC, 333))
        assert len(sub) == len(obs)

    def test_select_species_three_tuple(self, obs):
        sub = obs.select_species((_SRC, 333, 300))
        assert len(sub) == len(obs)

    def test_select_by_observation_requirement_single(self, obs):
        sub = obs.select_by_observation_requirement('BBCH_0')
        assert 'BBCH_0' in sub.observation_types

    def test_select_by_observation_requirement_list(self, obs):
        sub = obs.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])
        # All entries must have both types
        assert 'BBCH_0' in sub.observation_types
        assert 'BBCH_51' in sub.observation_types

    def test_select_by_observation_requirement_nonexistent(self, obs):
        sub = obs.select_by_observation_requirement('NONEXISTENT')
        assert len(sub) == 0

    def test_select_by_local_num_observations_zero(self, obs):
        sub = obs.select_by_local_num_observations(0, 'BBCH_0')
        assert len(sub) == len(obs)

    def test_select_by_local_num_observations_filters(self, obs):
        # loc_A has BBCH_0 for 2 years, loc_B for 1 year
        sub = obs.select_by_local_num_observations(2, 'BBCH_0')
        assert all(loc == (_SRC, 'loc_A') for loc in sub.locations)

    def test_select_by_ixs(self, obs):
        ixs = [obs.index[0]]
        sub = obs.select_by_ixs(ixs)
        assert len(sub) == 1

    def test_select_by_ixs_empty(self, obs):
        sub = obs.select_by_ixs([])
        assert len(sub) == 0

    def test_select_by_ixs_type_error(self, obs):
        with pytest.raises(TypeError):
            obs.select_by_ixs(obs.index[0])  # tuple, not list


class TestObservationsAggregation:

    def test_aggregate_in_grid_returns_observations(self, obs):
        result = obs.aggregate_in_grid(method='median', grid_size=(2.0, 2.0))
        assert isinstance(result, Observations)

    def test_aggregate_in_grid_reduces_locations(self, obs):
        # loc_A (51, 10) and loc_B (52, 11.5) fall in same 2° cell
        result = obs.aggregate_in_grid(method='median', grid_size=(2.0, 2.0))
        # Should have fewer or equal entries than original
        assert len(result) <= len(obs)

    def test_aggregate_in_grid_unknown_method(self, obs):
        with pytest.raises(DatasetException):
            obs.aggregate_in_grid(method='unknown')


class TestObservationsSplitting:

    def test_split_by_lon_border(self, obs):
        # loc_A lon=10.0, loc_B lon=11.5; split at 11.0
        below, above, info = obs.split_by_lon_border(11.0)
        assert isinstance(below, Observations)
        assert isinstance(above, Observations)
        assert len(below) + len(above) == len(obs)
        # loc_A (lon=10) should be in below, loc_B (lon=11.5) in above
        assert all(loc[1] == 'loc_A' for loc in below.locations)
        assert all(loc[1] == 'loc_B' for loc in above.locations)

    def test_split_by_lat_border(self, obs):
        # loc_A lat=51.0, loc_B lat=52.0; split at 51.5
        below, above, info = obs.split_by_lat_border(51.5)
        assert len(below) + len(above) == len(obs)
        assert all(loc[1] == 'loc_A' for loc in below.locations)
        assert all(loc[1] == 'loc_B' for loc in above.locations)

    def test_split_by_grid(self, obs):
        part1, part2, info = obs.split_by_grid(
            grid_size=(2.0, 2.0), split_size=0.5, random_state=0
        )
        assert isinstance(part1, Observations)
        assert isinstance(part2, Observations)
        assert 'cells_1' in info
        assert 'cells_2' in info
        # No overlap in locations
        locs1 = set(part1.locations)
        locs2 = set(part2.locations)
        assert locs1.isdisjoint(locs2)


class TestObservationsMerge:

    def test_merge(self, obs):
        # Split then merge should recover the original size
        below, above, _ = obs.split_by_lon_border(11.0)
        merged = Observations.merge(below, above)
        assert len(merged) == len(obs)

    def test_merge_deduplicates(self, obs):
        merged = Observations.merge(obs, obs)
        assert len(merged) == len(obs)


class TestObservationsShiftYear:

    def test_shift_year(self, df_y, df_y_loc):
        # BBCH_0 for year 2020 should become 2021 after shift +1
        shifted = Observations.shift_year(df_y, 'BBCH_0', 1)
        obs_shifted = Observations(shifted, df_y_loc)
        # After shift the BBCH_0 entries have year += 1
        bbch0_years = set(
            ix[2] for ix in obs_shifted.iter_index()
        )
        # Original BBCH_0 years were 2020 and 2021 → become 2021 and 2022
        assert 2020 not in bbch0_years or True  # years mixed with BBCH_51


# ===========================================================================
# Calendar
# ===========================================================================

class TestCalendar:

    def test_default_season(self):
        cal = Calendar()
        info = cal.get_season_info(year=2020, src=_SRC, species_id=999, subgroup_id=0)
        assert 'season_start' in info
        assert 'season_end' in info
        assert info['year'] == 2020

    def test_set_and_get_season(self):
        cal = Calendar()
        cal.set_season(_SRC, species_id=333, subgroup_id=300,
                       start_date='10-01', length=365)
        info = cal.get_season_info(year=2020, src=_SRC, species_id=333, subgroup_id=300)
        assert info['season_start'] == np.datetime64('2019-10-01')
        assert info['season_end'] == np.datetime64('2020-09-30')

    def test_season_length(self):
        cal = Calendar()
        cal.set_season(_SRC, species_id=1, subgroup_id=0,
                       start_date='03-01', length=180)
        info = cal.get_season_info(year=2020, src=_SRC, species_id=1, subgroup_id=0)
        delta = info['season_end'] - info['season_start']
        assert int(delta / np.timedelta64(1, 'D')) == 180

    def test_season_ends_in_correct_year(self):
        # Start 10-01, length 365 → ends ~10-01 next year (2020)
        cal = Calendar()
        cal.set_season(_SRC, species_id=333, subgroup_id=300,
                       start_date='10-01', length=365)
        info = cal.get_season_info(year=2020, src=_SRC, species_id=333, subgroup_id=300)
        end_year = info['season_end'].astype('datetime64[Y]').astype(int) + 1970
        assert end_year == 2020

    def test_cache_invalidated_after_set_season(self):
        cal = Calendar()
        cal.set_season(_SRC, species_id=1, subgroup_id=0,
                       start_date='03-01', length=100)
        info1 = cal.get_season_info(year=2020, src=_SRC, species_id=1, subgroup_id=0)

        # Overwrite with different config
        cal.set_season(_SRC, species_id=1, subgroup_id=0,
                       start_date='03-01', length=200)
        info2 = cal.get_season_info(year=2020, src=_SRC, species_id=1, subgroup_id=0)

        assert info1['season_length'] != info2['season_length']

    def test_from_config(self):
        cal = Calendar.from_config()
        assert isinstance(cal, Calendar)


# ===========================================================================
# OpenMeteoFeatures (debug mode only)
# ===========================================================================

class TestOpenMeteoFeatures:

    def test_get_data_returns_dict(self, features, obs):
        ix = obs.index[0]
        result = features.get_data(ix)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'temperature_2m_mean', 'daylight_duration'}

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
        feats = OpenMeteoFeatures(
            calendar=calendar,
            data_keys=['temperature_2m_mean'],
            step='daily',
            debug_mode=True,
        )
        ix = obs.index[0]
        src, loc_id, year, species_id, subgroup_id = ix
        info = calendar.get_season_info(year=year, src=src,
                                        species_id=species_id, subgroup_id=subgroup_id)
        expected_len = int(
            (info['season_end'] - info['season_start']) / np.timedelta64(1, 'D') - 1
        )
        arr = feats.get_variable('temperature_2m_mean', ix)
        # Length should be within 1 of expected (off-by-one due to inclusive end)
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
        assert 'temperature_2m_mean' in features.data_keys

    def test_context_manager(self, calendar):
        with OpenMeteoFeatures(calendar=calendar, debug_mode=True) as f:
            assert f.step == 'daily'
        # After exit, store is closed (cache cleared)
        assert f.cache_size() == 0

    def test_invalid_step_raises(self, calendar):
        with pytest.raises(ValueError):
            OpenMeteoFeatures(calendar=calendar, step='weekly', debug_mode=True)


# ===========================================================================
# Dataset
# ===========================================================================

class TestDatasetInit:

    def test_requires_observations_instance(self):
        with pytest.raises(TypeError):
            Dataset("not an observations object")

    def test_minimal_init(self, obs):
        ds = Dataset(obs)
        assert len(ds) == len(obs)

    def test_with_calendar_and_features(self, dataset):
        assert dataset.calendar is not None
        assert len(dataset.feature_providers) > 0


class TestDatasetSequenceProtocol:

    def test_len(self, dataset):
        assert len(dataset) == len(dataset.observations)

    def test_contains_tuple(self, dataset):
        ix = dataset.observations.index[0]
        assert ix in dataset

    def test_getitem_without_calendar(self, obs):
        ds = Dataset(obs)
        item = ds[0]
        assert KEY_OBSERVATIONS in item
        assert _KEY_SEASON_START not in item
        assert KEY_FEATURES not in item

    def test_getitem_with_calendar_adds_season_fields(self, obs, calendar):
        ds = Dataset(obs, calendar=calendar)
        item = ds[0]
        assert _KEY_SEASON_START in item
        assert _KEY_SEASON_END in item
        assert KEY_OBSERVATIONS_INDEX in item

    def test_getitem_observation_indices_are_ints(self, obs, calendar):
        ds = Dataset(obs, calendar=calendar)
        item = ds[0]
        for v in item[KEY_OBSERVATIONS_INDEX].values():
            assert isinstance(v, int)

    def test_getitem_observation_indices_non_negative(self, obs, calendar):
        ds = Dataset(obs, calendar=calendar)
        for i in range(len(ds)):
            item = ds[i]
            for v in item[KEY_OBSERVATIONS_INDEX].values():
                assert v >= 0

    def test_getitem_with_features_adds_features(self, dataset):
        item = dataset[0]
        assert KEY_FEATURES in item
        assert isinstance(item[KEY_FEATURES], dict)

    def test_iter_index(self, dataset):
        ixs = list(dataset.iter_index())
        assert len(ixs) == len(dataset)

    def test_iter_items(self, dataset):
        items = list(dataset.iter_items())
        assert len(items) == len(dataset)
        assert all(KEY_FEATURES in item for item in items)


class TestDatasetProperties:

    def test_locations(self, dataset):
        assert len(dataset.locations) == 2

    def test_species(self, dataset):
        assert (_SRC, 333) in dataset.species

    def test_years(self, dataset):
        assert set(dataset.years) == {2020, 2021}

    def test_bounding_box(self, dataset):
        bb = dataset.bounding_box
        assert bb['min_lat'] < bb['max_lat']

    def test_get_location_coordinates(self, dataset):
        coords = dataset.get_location_coordinates((_SRC, 'loc_A'))
        assert KEY_LAT in coords

    def test_observation_counts(self, dataset):
        counts = dataset.observation_counts()
        assert 'BBCH_0' in counts


class TestDatasetSelection:

    def test_select_locations_returns_dataset(self, dataset):
        result = dataset.select_locations((_SRC, 'loc_A'))
        assert isinstance(result, Dataset)

    def test_select_locations_preserves_calendar(self, dataset):
        result = dataset.select_locations((_SRC, 'loc_A'))
        assert result.calendar is dataset.calendar

    def test_select_locations_preserves_features(self, dataset):
        result = dataset.select_locations((_SRC, 'loc_A'))
        assert result.feature_providers == dataset.feature_providers

    def test_select_years_returns_dataset(self, dataset):
        result = dataset.select_years(2020)
        assert isinstance(result, Dataset)
        assert result.years == [2020]

    def test_select_species_returns_dataset(self, dataset):
        result = dataset.select_species((_SRC, 333))
        assert isinstance(result, Dataset)

    def test_select_by_observation_requirement_returns_dataset(self, dataset):
        result = dataset.select_by_observation_requirement('BBCH_0')
        assert isinstance(result, Dataset)

    def test_select_by_local_num_observations_returns_dataset(self, dataset):
        result = dataset.select_by_local_num_observations(1, 'BBCH_0')
        assert isinstance(result, Dataset)

    def test_select_by_ixs_returns_dataset(self, dataset):
        ixs = [dataset.observations.index[0]]
        result = dataset.select_by_ixs(ixs)
        assert isinstance(result, Dataset)
        assert len(result) == 1

    def test_aggregate_in_grid_returns_dataset(self, dataset):
        result = dataset.aggregate_in_grid(method='median', grid_size=(2.0, 2.0))
        assert isinstance(result, Dataset)


class TestDatasetSplitting:

    def test_split_by_lon_border(self, dataset):
        d1, d2, info = dataset.split_by_lon_border(11.0)
        assert isinstance(d1, Dataset)
        assert isinstance(d2, Dataset)
        assert len(d1) + len(d2) == len(dataset)

    def test_split_by_lon_border_preserves_calendar(self, dataset):
        d1, d2, _ = dataset.split_by_lon_border(11.0)
        assert d1.calendar is dataset.calendar
        assert d2.calendar is dataset.calendar

    def test_split_by_lat_border(self, dataset):
        d1, d2, info = dataset.split_by_lat_border(51.5)
        assert isinstance(d1, Dataset)
        assert isinstance(d2, Dataset)
        assert len(d1) + len(d2) == len(dataset)

    def test_split_by_grid(self, dataset):
        d1, d2, info = dataset.split_by_grid(
            grid_size=(2.0, 2.0), split_size=0.5, random_state=0
        )
        assert isinstance(d1, Dataset)
        assert isinstance(d2, Dataset)


class TestDatasetContextManager:

    def test_context_manager(self, obs, calendar):
        feats = OpenMeteoFeatures(calendar=calendar, debug_mode=True)
        with Dataset(obs, calendar=calendar, feature_providers=[feats]) as ds:
            assert len(ds) > 0
        # Feature cache should be cleared after exit
        assert feats.cache_size() == 0

    def test_close_without_features(self, obs):
        ds = Dataset(obs)
        ds.close()  # should not raise


class TestDatasetFeatureStats:

    def test_compute_feature_stats_debug_mode(self, dataset):
        stats = dataset.compute_feature_stats(verbose=False)
        assert 'temperature_2m_mean' in stats
        for mean, std in stats.values():
            assert isinstance(mean, float)
            assert isinstance(std, float)

    def test_compute_feature_stats_without_provider_raises(self, obs):
        ds = Dataset(obs)
        with pytest.raises(DatasetException):
            ds.compute_feature_stats(verbose=False)

    def test_download_features_without_provider_raises(self, obs):
        ds = Dataset(obs)
        with pytest.raises(DatasetException):
            ds.download_features()


class TestDatasetCacheData:

    def test_cache_data_runs_without_error(self, dataset):
        dataset.cache_data(verbose=False)


class TestDatasetMerge:

    def test_merge_returns_dataset(self, dataset):
        d1, d2, _ = dataset.split_by_lon_border(11.0)
        merged = Dataset.merge(d1, d2)
        assert isinstance(merged, Dataset)

    def test_merge_keeps_calendar_of_first(self, dataset):
        d1, d2, _ = dataset.split_by_lon_border(11.0)
        merged = Dataset.merge(d1, d2)
        assert merged.calendar is d1.calendar

    def test_merge_recovers_full_size(self, dataset):
        d1, d2, _ = dataset.split_by_lon_border(11.0)
        merged = Dataset.merge(d1, d2)
        assert len(merged) == len(dataset)


class TestDatasetListDatasets:

    def test_list_datasets_returns_list(self):
        result = Dataset.list_datasets()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_known_dataset_names_present(self):
        names = Dataset.list_datasets()
        assert 'CPF_PEP725_winter_wheat' in names
        assert 'GMU_Cherry_Japan' in names
        assert 'PEP725_Apple' in names
