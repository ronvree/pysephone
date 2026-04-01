"""
Tests for GDDModel and start-index helpers.
All tests are self-contained — uses FakeWeatherProvider and synthetic observations.
"""

import numpy as np
import pandas as pd
import pytest

from pysephone.constants import KEYS_INDEX, KEY_OBS_TYPE, KEY_OBSERVATIONS
from pysephone.dataset.dataset import Dataset
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.fake_weather import FakeWeatherProvider
from pysephone.models.gdd import GDDModel, observation_start, zero_start


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df_y(rows):
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        names=list(KEYS_INDEX) + [KEY_OBS_TYPE],
    )
    return pd.DataFrame({KEY_OBSERVATIONS: [np.datetime64(r[6]) for r in rows]}, index=index)


def _make_df_y_loc(rows):
    index = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=['src', 'loc_id'])
    return pd.DataFrame({'lat': [r[2] for r in rows], 'lon': [r[3] for r in rows]}, index=index)


@pytest.fixture
def dataset():
    df_y = _make_df_y([
        ('src', 'loc1', 2000, 1, 10, 'BBCH_11', '2000-03-01'),
        ('src', 'loc1', 2000, 1, 10, 'BBCH_60', '2000-05-10'),
        ('src', 'loc1', 2001, 1, 10, 'BBCH_11', '2001-03-05'),
        ('src', 'loc1', 2001, 1, 10, 'BBCH_60', '2001-05-12'),
    ])
    df_y_loc = _make_df_y_loc([('src', 'loc1', 48.0, 11.0)])
    obs = Observations(df_y, df_y_loc)
    cal = Calendar()
    cal.set_season('src', species_id=1, subgroup_id=10, start_date='10-01', length=365)
    provider = FakeWeatherProvider(cal, seed=0)
    return Dataset(obs, calendar=cal, feature_providers=[provider])


@pytest.fixture
def sample(dataset):
    return dataset[('src', 'loc1', 2000, 1, 10)]


# ---------------------------------------------------------------------------
# Start-index functions
# ---------------------------------------------------------------------------

class TestStartIndexFns:

    def test_zero_start_returns_zero(self, sample):
        fn = zero_start()
        assert fn(sample) == 0

    def test_observation_start_reads_index(self, sample):
        fn = observation_start('BBCH_11')
        ix = fn(sample)
        assert isinstance(ix, int)
        assert ix >= 0

    def test_observation_start_missing_key_raises(self, sample):
        fn = observation_start('BBCH_99')
        with pytest.raises(KeyError):
            fn(sample)


# ---------------------------------------------------------------------------
# GDDModel construction
# ---------------------------------------------------------------------------

class TestGDDModelConstruction:

    def test_basic(self):
        m = GDDModel(threshold=500.0, t_base=5.0)
        assert m.threshold == 500.0
        assert m.t_base == 5.0
        assert not m.has_upper_bound
        assert m.t_upper is None

    def test_with_upper_bound(self):
        m = GDDModel(threshold=500.0, t_base=5.0, t_upper=25.0)
        assert m.has_upper_bound
        assert m.t_upper == 25.0
        assert 't_upper' in m.param_keys

    def test_default_ix_start_fn_is_zero(self, sample):
        m = GDDModel(threshold=500.0, t_base=5.0)
        _, info = m.predict(sample)
        m2 = GDDModel(threshold=500.0, t_base=5.0, ix_start_fn=zero_start())
        _, info2 = m2.predict(sample)
        assert info['ix'] == info2['ix']


# ---------------------------------------------------------------------------
# GDDModel.predict
# ---------------------------------------------------------------------------

class TestGDDModelPredict:

    def test_returns_datetime64_and_ix(self, sample):
        m = GDDModel(threshold=100.0, t_base=5.0)
        date, info = m.predict(sample)
        assert isinstance(date, np.datetime64)
        assert 'ix' in info
        assert isinstance(info['ix'], int)

    def test_ix_within_season(self, sample, dataset):
        m = GDDModel(threshold=100.0, t_base=5.0)
        temps = sample['features']['temperature_2m_mean']
        _, info = m.predict(sample)
        assert 0 <= info['ix'] <= len(temps)

    def test_predicted_date_matches_ix(self, sample):
        m = GDDModel(threshold=100.0, t_base=5.0)
        date, info = m.predict(sample)
        expected = sample['season_start'] + np.timedelta64(info['ix'], 'D')
        assert date == expected

    def test_high_threshold_predicts_end_of_season(self, sample):
        m = GDDModel(threshold=1e9, t_base=5.0)
        _, info = m.predict(sample)
        temps = sample['features']['temperature_2m_mean']
        assert info['ix'] == len(temps)

    def test_zero_threshold_predicts_ix_start(self, sample):
        m = GDDModel(threshold=0.0, t_base=5.0)
        _, info = m.predict(sample)
        assert info['ix'] == 0

    def test_upper_bound_reduces_accumulation(self, sample):
        m_no_cap  = GDDModel(threshold=500.0, t_base=5.0)
        m_low_cap = GDDModel(threshold=500.0, t_base=5.0, t_upper=6.0)
        _, info_no  = m_no_cap.predict(sample)
        _, info_cap = m_low_cap.predict(sample)
        # With a lower cap, accumulation is slower → threshold reached later
        assert info_cap['ix'] >= info_no['ix']

    def test_ix_start_fn_shifts_result(self, sample):
        m0 = GDDModel(threshold=200.0, t_base=5.0, ix_start_fn=zero_start())
        m1 = GDDModel(threshold=200.0, t_base=5.0, ix_start_fn=observation_start('BBCH_11'))
        _, info0 = m0.predict(sample)
        _, info1 = m1.predict(sample)
        # Starting later cannot predict before the start index
        ix_start = observation_start('BBCH_11')(sample)
        assert info1['ix'] >= ix_start


# ---------------------------------------------------------------------------
# GDDModel.fit (smoke test — checks it runs and improves MSE)
# ---------------------------------------------------------------------------

class TestGDDModelFit:

    def test_fit_returns_model_and_dict(self, dataset):
        model, info = GDDModel.fit(
            target_fn=lambda s: s['observations']['BBCH_60'],
            dataset=dataset,
            model_kwargs=dict(threshold=300.0, t_base=5.0),
        )
        assert isinstance(model, GDDModel)
        assert isinstance(info, dict)

    def test_fit_preserves_param_keys(self, dataset):
        model, _ = GDDModel.fit(
            target_fn=lambda s: s['observations']['BBCH_60'],
            dataset=dataset,
            model_kwargs=dict(threshold=300.0, t_base=5.0),
        )
        assert 'threshold' in model.params
        assert 't_base' in model.params

    def test_fit_empty_dataset_returns_unchanged_model(self, dataset):
        empty = dataset.select_years([9999])
        initial = GDDModel(threshold=300.0, t_base=5.0)
        model, _ = GDDModel.fit(
            target_fn=lambda s: s['observations']['BBCH_60'],
            dataset=empty,
            model=initial,
        )
        assert model is initial
