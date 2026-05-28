"""
Tests for the BloomBench benchmark module.

Self-contained: synthetic observations, FakeWeatherProvider, NullModel.
No network, no real AgERA5 data, no torch fits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysephone.constants import KEY_OBS_TYPE, KEY_OBSERVATIONS, KEYS_INDEX
from pysephone.dataset.dataset import Dataset
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.fake_weather import FakeWeatherProvider

from pysephone.benchmarks.bloombench import (
    MODELS,
    compare,
    config,
    fit as bb_fit,
    is_torch_model,
    is_tree_model,
)
from pysephone.benchmarks.bloombench.cli import build_parser, main as cli_main
from pysephone.benchmarks.bloombench.datasets import temporal_split
from pysephone.benchmarks.bloombench.runner import run_benchmark


# ---------------------------------------------------------------------------
# Synthetic-dataset fixtures
# ---------------------------------------------------------------------------

_SRC = 'src'
_TARGET = 'BBCH_60'
_FAKE_FEATURE_KEY = FakeWeatherProvider.KEY_TEMPERATURE  # 'temperature_2m_mean'


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
def synthetic_dataset():
    """A 4-year × 3-location synthetic dataset with FakeWeatherProvider."""
    rows = []
    for year in (2000, 2001, 2002, 2003):
        # Bloom day drifts a bit by year so Linear / Mean differ a little.
        bloom_doy = 130 + (year - 2000) * 2
        bloom_date = np.datetime64(f'{year}-01-01') + np.timedelta64(bloom_doy - 1, 'D')
        for loc in ('loc1', 'loc2', 'loc3'):
            rows.append((_SRC, loc, year, 1, 10, _TARGET, str(bloom_date)))

    df_y = _make_df_y(rows)
    df_y_loc = _make_df_y_loc([
        (_SRC, 'loc1', 48.0, 11.0),
        (_SRC, 'loc2', 49.0, 12.0),
        (_SRC, 'loc3', 50.0, 13.0),
    ])
    obs = Observations(df_y, df_y_loc)

    cal = Calendar(default_start='10-01', default_length=365)
    cal.set_season(_SRC, species_id=1, subgroup_id=10,
                   start_date='10-01', length=365)
    provider = FakeWeatherProvider(cal, seed=42)
    return Dataset(obs, calendar=cal, feature_providers=[provider])


@pytest.fixture
def datasets_dict(synthetic_dataset):
    """A datasets_dict with one entry — the synthetic dataset split 75/25."""
    ds_train, ds_test = temporal_split(synthetic_dataset, train_fraction=0.75)
    assert len(ds_train) > 0 and len(ds_test) > 0
    return {'fake_bloom': (ds_train, ds_test, _TARGET)}


@pytest.fixture
def datasets_dict_two(synthetic_dataset):
    """Two synthetic datasets — required by compare (needs >= 2 datasets)."""
    ds_train, ds_test = temporal_split(synthetic_dataset, train_fraction=0.75)
    return {
        'fake_bloom_a': (ds_train, ds_test, _TARGET),
        'fake_bloom_b': (ds_train, ds_test, _TARGET),
    }


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:

    def test_datasets_requested_count(self):
        assert len(config.DATASETS_REQUESTED) == 18

    def test_each_dataset_pair_is_2tuple(self):
        for entry in config.DATASETS_REQUESTED:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_run_name_roundtrip(self):
        rn = config.run_name('PEP725_Apple', 'LSTM', 0)
        assert rn == 'bb_ext_PEP725_Apple_LSTM_seed0'

    def test_run_name_custom_prefix(self):
        rn = config.run_name('PEP725_Apple', 'LSTM', 1, run_prefix='myrun')
        assert rn == 'myrun_PEP725_Apple_LSTM_seed1'

    def test_hp_cache_dir_namespaced_by_provider(self, tmp_path):
        path = config.hp_cache_dir(tmp_path)
        # No mkdir on call
        assert not path.exists()
        assert path.parts[-3:] == ('bloombench', 'hyperparams', 'agera5')

    def test_runs_dir_namespaced(self, tmp_path):
        path = config.runs_dir(tmp_path)
        assert path.parts[-2:] == ('bloombench', 'runs')

    def test_torch_train_kwargs_has_device(self):
        kwargs = config.torch_train_kwargs()
        assert 'device' in kwargs
        assert kwargs['device'] in ('cpu', 'cuda')

    def test_torch_train_kwargs_has_budget(self):
        kwargs = config.torch_train_kwargs()
        assert kwargs['num_epochs'] == 200
        assert kwargs['batch_size'] == 32
        assert kwargs['early_stopping'] is True


# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------

class TestTemporalSplit:

    def test_split_75_25(self, synthetic_dataset):
        ds_train, ds_test = temporal_split(synthetic_dataset, train_fraction=0.75)
        train_years = set(ds_train.years)
        test_years = set(ds_test.years)
        # Years are disjoint and train years are all earlier than test years.
        assert train_years.isdisjoint(test_years)
        assert max(train_years) < min(test_years)

    def test_split_non_empty(self, synthetic_dataset):
        ds_train, ds_test = temporal_split(synthetic_dataset)
        assert len(ds_train) > 0
        assert len(ds_test) > 0


# ---------------------------------------------------------------------------
# fit.py
# ---------------------------------------------------------------------------

class TestFitRegistry:

    def test_seven_models(self):
        assert set(MODELS) == {'Mean', 'Linear', 'RandomForest', 'XGBoost',
                               'CNN1D', 'LSTM', 'Transformer'}

    def test_is_torch_model_classification(self):
        assert is_torch_model('LSTM')
        assert is_torch_model('CNN1D')
        assert is_torch_model('Transformer')
        assert not is_torch_model('Mean')
        assert not is_torch_model('Linear')
        assert not is_torch_model('RandomForest')

    def test_is_tree_model_classification(self):
        assert is_tree_model('RandomForest')
        assert is_tree_model('XGBoost')
        assert not is_tree_model('Mean')
        assert not is_tree_model('LSTM')

    def test_fit_one_unknown_key_raises(self, synthetic_dataset):
        with pytest.raises(KeyError):
            bb_fit.fit_one(
                'NoSuchModel',
                lambda s: s[KEY_OBSERVATIONS][_TARGET],
                synthetic_dataset,
                seed=0, dataset_name='fake',
            )

    def test_fit_one_mean(self, synthetic_dataset):
        target_fn = bb_fit._make_target_fn(_TARGET)
        model = bb_fit.fit_one('Mean', target_fn, synthetic_dataset,
                               seed=0, dataset_name='fake')
        assert model is not None
        date, _ = model.predict(synthetic_dataset[0])
        assert np.datetime64(date).dtype.kind == 'M'


class TestHpCacheRoundtrip:

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        bb_fit.save_hp_cache('fake_ds', 'LSTM',
                             {'hidden_size': 64, 'learning_rate': 1e-3})
        loaded = bb_fit.load_hp_cache('fake_ds', 'LSTM')
        assert loaded == {'hidden_size': 64, 'learning_rate': 1e-3}

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        assert bb_fit.load_hp_cache('not_there', 'LSTM') is None

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        # Pre-condition: dir does not exist
        assert not config.hp_cache_dir().exists()
        bb_fit.save_hp_cache('ds', 'LSTM', {'lr': 1e-3})
        # mkdir only on write
        assert config.hp_cache_dir().exists()

    def test_atomic_write_leaves_no_tempfile(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        bb_fit.save_hp_cache('ds', 'LSTM', {'lr': 1e-3})
        leftover = [p for p in config.hp_cache_dir().iterdir()
                    if p.name != 'ds_LSTM.json']
        assert leftover == []


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------

class TestRunBenchmark:

    def test_run_mean_and_linear(self, tmp_path, monkeypatch, datasets_dict):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        df = run_benchmark(
            seeds=[0],
            models=['Mean', 'Linear'],
            datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False,
            verbose=False,
        )
        assert len(df) == 2  # 1 dataset x 2 models x 1 seed
        assert set(df['model']) == {'Mean', 'Linear'}
        assert (df['status'] == 'ok').all()
        # Sidecar files are present.
        out_dir = config.runs_dir() / config.RUN_PREFIX
        assert (out_dir / 'results.csv').exists()
        assert (out_dir / 'datasets.json').exists()
        with open(out_dir / 'datasets.json') as f:
            assert json.load(f) == ['fake_bloom']

    def test_results_columns(self, tmp_path, monkeypatch, datasets_dict):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        df = run_benchmark(
            seeds=[0], models=['Mean'], datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        expected = {'seed', 'dataset', 'model', 'source', 'status',
                    'n_train', 'n_test', 'mae_train', 'mae_test',
                    'rmse_test', 'r2_test', 'seconds', 'error'}
        assert expected.issubset(df.columns)

    def test_cache_hit_on_second_run(self, tmp_path, monkeypatch, datasets_dict):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        df1 = run_benchmark(
            seeds=[0], models=['Mean'], datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        assert (df1['source'] == 'fit').all()

        df2 = run_benchmark(
            seeds=[0], models=['Mean'], datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        assert (df2['source'] == 'cache').all()

    def test_force_retrain_bypasses_cache(self, tmp_path, monkeypatch, datasets_dict):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        run_benchmark(
            seeds=[0], models=['Mean'], datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        df = run_benchmark(
            seeds=[0], models=['Mean'], datasets_dict=datasets_dict,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, force_retrain=True, verbose=False,
        )
        assert (df['source'] == 'fit').all()


# ---------------------------------------------------------------------------
# compare.py
# ---------------------------------------------------------------------------

try:
    import scikit_posthocs  # noqa: F401
    _HAS_POSTHOCS = True
except ImportError:
    _HAS_POSTHOCS = False


@pytest.mark.skipif(not _HAS_POSTHOCS,
                    reason='scikit-posthocs not installed (install pysephone[stats])')
class TestRunComparison:

    def test_compare_with_two_models_two_seeds(self, tmp_path, monkeypatch, datasets_dict_two):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        run_benchmark(
            seeds=[0, 1], models=['Mean', 'Linear'], datasets_dict=datasets_dict_two,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        report = compare.run_comparison(
            seeds=[0, 1], models=['Mean', 'Linear'],
            save=False, verbose=False,
        )
        assert set(report.scores.columns) == {'Mean', 'Linear'}
        # Both synthetic datasets should survive into the scores table.
        assert set(report.scores.index) == {'fake_bloom_a', 'fake_bloom_b'}

    def test_compare_picks_up_sidecar_datasets(self, tmp_path, monkeypatch, datasets_dict_two):
        monkeypatch.setenv('PYSEPHONE_DATA_ROOT', str(tmp_path))
        run_benchmark(
            seeds=[0], models=['Mean', 'Linear'], datasets_dict=datasets_dict_two,
            feature_keys=[_FAKE_FEATURE_KEY],
            compute_feature_stats=False, verbose=False,
        )
        report = compare.run_comparison(
            seeds=[0], models=['Mean', 'Linear'],
            save=False, verbose=False,
        )
        # Even though we did not pass `datasets=...`, the sidecar should
        # have restricted the comparison to the datasets we actually ran.
        assert set(report.scores.index) == {'fake_bloom_a', 'fake_bloom_b'}


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------

class TestCli:

    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_run_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(['run', '--help'])
        assert exc.value.code == 0

    def test_hpo_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(['hpo', '--help'])
        assert exc.value.code == 0

    def test_compare_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(['compare', '--help'])
        assert exc.value.code == 0

    def test_no_subcommand_errors(self):
        with pytest.raises(SystemExit):
            cli_main([])
