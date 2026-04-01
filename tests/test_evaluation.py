"""
Tests for SingleTargetRegression evaluation.

All tests are self-contained — no real model or dataset required.
Uses NullModel and FakeWeatherProvider on synthetic observations.
"""

import numpy as np
import pandas as pd
import pytest

from pysephone.constants import KEYS_INDEX, KEY_OBS_TYPE, KEY_OBSERVATIONS
from pysephone.dataset.dataset import Dataset
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.fake_weather import FakeWeatherProvider
from pysephone.evaluation.regression import EvaluationException, SingleTargetRegression
from pysephone.models.base import NullModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        names=['src', 'loc_id'],
    )
    return pd.DataFrame(
        {'lat': [r[2] for r in rows], 'lon': [r[3] for r in rows]},
        index=index,
    )


@pytest.fixture
def simple_dataset():
    df_y = _make_df_y([
        ('src', 'loc1', 2000, 1, 10, 'BBCH_60', '2000-05-10'),
        ('src', 'loc1', 2001, 1, 10, 'BBCH_60', '2001-05-12'),
        ('src', 'loc2', 2000, 1, 10, 'BBCH_60', '2000-05-08'),
    ])
    df_y_loc = _make_df_y_loc([
        ('src', 'loc1', 48.0, 11.0),
        ('src', 'loc2', 49.0, 12.0),
    ])
    obs = Observations(df_y, df_y_loc)

    cal = Calendar()
    cal.set_season('src', species_id=1, subgroup_id=10,
                   start_date='10-01', length=365)
    provider = FakeWeatherProvider(cal, seed=42)
    return Dataset(obs, calendar=cal, feature_providers=[provider])


def _target_fn(sample):
    return sample['observations']['BBCH_60']


# ---------------------------------------------------------------------------
# SingleTargetRegression.run
# ---------------------------------------------------------------------------

class TestSingleTargetRegressionRun:

    def test_run_train_only(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        assert isinstance(ev, SingleTargetRegression)
        assert len(ev.df_train) == len(simple_dataset)
        assert ev.df_test.empty

    def test_run_with_test(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            dataset_test=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        assert len(ev.df_train) == len(simple_dataset)
        assert len(ev.df_test) == len(simple_dataset)

    def test_df_columns(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        expected_cols = {'year', 'predicted_doy', 'observed_doy', 'error'}
        assert expected_cols.issubset(set(ev.df_train.columns))

    def test_error_is_pred_minus_obs(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        diff = (ev.df_train['predicted_doy'] - ev.df_train['observed_doy'] - ev.df_train['error']).abs()
        assert diff.max() < 1e-6

    def test_metadata_contains_model_class(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        assert ev.metadata['model_class'] == 'NullModel'
        assert ev.metadata['n_train'] == len(simple_dataset)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_metric_keys(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        metrics = ev.compute_metrics()
        assert 'train' in metrics
        assert 'test' in metrics
        for key in ('mae', 'rmse', 'r2', 'bias', 'n'):
            assert key in metrics['train']

    def test_test_metrics_empty_when_no_test(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        assert ev.compute_metrics()['test'] == {}

    def test_rmse_nonnegative(self, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_run',
        )
        assert ev.compute_metrics()['train']['rmse'] >= 0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_and_load(self, tmp_path, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            dataset_test=simple_dataset,
            target_fn=_target_fn,
            run_name='test_save',
        )
        ev.save(root=tmp_path)
        loaded = SingleTargetRegression.load('test_save', root=tmp_path)
        assert len(loaded.df_train) == len(ev.df_train)
        assert len(loaded.df_test) == len(ev.df_test)
        assert loaded.metadata['model_class'] == 'NullModel'

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(EvaluationException):
            SingleTargetRegression.load('does_not_exist', root=tmp_path)

    def test_roundtrip_values(self, tmp_path, simple_dataset):
        model = NullModel()
        ev = SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            target_fn=_target_fn,
            run_name='test_rt',
        )
        ev.save(root=tmp_path)
        loaded = SingleTargetRegression.load('test_rt', root=tmp_path)
        pd.testing.assert_frame_equal(
            ev.df_train.reset_index(drop=True),
            loaded.df_train.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

class TestPlots:

    @pytest.fixture
    def eval_obj(self, simple_dataset):
        model = NullModel()
        return SingleTargetRegression.run(
            model=model,
            dataset_train=simple_dataset,
            dataset_test=simple_dataset,
            target_fn=_target_fn,
            run_name='plot_test',
        )

    def test_plot_scatter_returns_figure(self, eval_obj):
        import matplotlib.pyplot as plt
        fig = eval_obj.plot_scatter()
        assert hasattr(fig, 'savefig')
        plt.close('all')

    def test_plot_residuals_returns_figure(self, eval_obj):
        import matplotlib.pyplot as plt
        fig = eval_obj.plot_residuals_over_time()
        assert hasattr(fig, 'savefig')
        plt.close('all')

    def test_plot_error_dist_returns_figure(self, eval_obj):
        import matplotlib.pyplot as plt
        fig = eval_obj.plot_error_distribution()
        assert hasattr(fig, 'savefig')
        plt.close('all')

    def test_plot_annual_mean_doy_returns_figure(self, eval_obj):
        import matplotlib.pyplot as plt
        fig = eval_obj.plot_annual_mean_doy()
        assert hasattr(fig, 'savefig')
        plt.close('all')
