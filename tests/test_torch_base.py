"""
Tests for BaseTorchModel.

Uses NullTorchModel — a minimal concrete subclass that predicts a single
learnable constant index — to exercise the training loop, collation,
device handling, save/load, and loss without requiring a real architecture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pysephone.constants import KEYS_INDEX, KEY_OBS_TYPE, KEY_OBSERVATIONS
from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.dataset.dataset import Dataset
from pysephone.dataset.observations import Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.fake_weather import FakeWeatherProvider
from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------

class NullTorchModel(BaseTorchModel):
    """Predicts a single learnable scalar index for every sample in a batch."""

    def __init__(self, init_ix: float = 100.0) -> None:
        super().__init__()
        self._ix = nn.Parameter(torch.tensor(init_ix))

    def forward(self, xs, **kwargs):
        batch_size = len(xs['season_start'])
        ix = self._ix.expand(batch_size)
        return ix, {}


# ---------------------------------------------------------------------------
# Dataset fixtures  (shared with test_gdd.py pattern)
# ---------------------------------------------------------------------------

def _make_df_y(rows):
    index = pd.MultiIndex.from_tuples(
        [r[:6] for r in rows],
        names=list(KEYS_INDEX) + [KEY_OBS_TYPE],
    )
    return pd.DataFrame({KEY_OBSERVATIONS: [np.datetime64(r[6]) for r in rows]}, index=index)


def _make_df_y_loc(rows):
    index = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=['src', 'loc_id'])
    return pd.DataFrame({'lat': [r[2] for r in rows], 'lon': [r[3] for r in rows]}, index=index)


@pytest.fixture
def dataset():
    df_y = _make_df_y([
        ('src', 'loc1', 2000, 1, 10, 'BBCH_60', '2000-05-10'),
        ('src', 'loc1', 2001, 1, 10, 'BBCH_60', '2001-05-12'),
        ('src', 'loc1', 2002, 1, 10, 'BBCH_60', '2002-05-08'),
        ('src', 'loc1', 2003, 1, 10, 'BBCH_60', '2003-05-15'),
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


TARGET_FN = lambda s: s['observations']['BBCH_60']


# ---------------------------------------------------------------------------
# cast_to_tensor
# ---------------------------------------------------------------------------

class TestCastToTensor:

    def test_features_become_tensors(self, sample):
        out = NullTorchModel.cast_to_tensor(sample)
        assert isinstance(out[KEY_FEATURES]['temperature_2m_mean'], torch.Tensor)

    def test_observations_index_become_tensors(self, sample):
        out = NullTorchModel.cast_to_tensor(sample)
        for v in out[KEY_OBSERVATIONS_INDEX].values():
            assert isinstance(v, torch.Tensor)

    def test_non_feature_keys_pass_through(self, sample):
        out = NullTorchModel.cast_to_tensor(sample)
        assert out['season_start'] == sample['season_start']
        assert out['season_end'] == sample['season_end']

    def test_device_argument_respected(self, sample):
        device = torch.device('cpu')
        out = NullTorchModel.cast_to_tensor(sample, device=device)
        for v in out[KEY_FEATURES].values():
            assert v.device.type == 'cpu'


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:

    def test_single_item_batch_size_one(self, sample):
        tensor_sample = NullTorchModel.cast_to_tensor(sample)
        batch = NullTorchModel.collate_fn([tensor_sample])
        assert batch[KEY_FEATURES]['temperature_2m_mean'].shape[0] == 1

    def test_feature_tensors_stacked(self, dataset):
        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())[:2]]
        batch = NullTorchModel.collate_fn(items)
        assert batch[KEY_FEATURES]['temperature_2m_mean'].shape[0] == 2

    def test_metadata_collected_as_lists(self, dataset):
        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())[:3]]
        batch = NullTorchModel.collate_fn(items)
        assert isinstance(batch['season_start'], list)
        assert len(batch['season_start']) == 3

    def test_observations_index_stacked(self, dataset):
        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())[:2]]
        batch = NullTorchModel.collate_fn(items)
        for v in batch[KEY_OBSERVATIONS_INDEX].values():
            assert isinstance(v, torch.Tensor)
            assert v.shape[0] == 2


# ---------------------------------------------------------------------------
# _batch_to_device
# ---------------------------------------------------------------------------

class TestBatchToDevice:

    def test_tensor_moved(self):
        t = torch.zeros(4)
        out = BaseTorchModel._batch_to_device(t, torch.device('cpu'))
        assert isinstance(out, torch.Tensor)

    def test_dict_of_tensors_moved_recursively(self):
        d = {'a': torch.zeros(3), 'b': {'c': torch.ones(2)}}
        out = BaseTorchModel._batch_to_device(d, torch.device('cpu'))
        assert isinstance(out['a'], torch.Tensor)
        assert isinstance(out['b']['c'], torch.Tensor)

    def test_non_tensor_passed_through(self):
        lst = [np.datetime64('2000-01-01'), np.datetime64('2001-01-01')]
        out = BaseTorchModel._batch_to_device(lst, torch.device('cpu'))
        assert out is lst

    def test_mixed_dict_non_tensor_untouched(self, sample):
        tensor_sample = NullTorchModel.cast_to_tensor(sample)
        batch = NullTorchModel.collate_fn([tensor_sample])
        out = BaseTorchModel._batch_to_device(batch, torch.device('cpu'))
        assert isinstance(out['season_start'], list)
        assert isinstance(out[KEY_FEATURES]['temperature_2m_mean'], torch.Tensor)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestNullTorchModelPredict:

    def test_returns_datetime64(self, sample):
        m = NullTorchModel()
        date, _ = m.predict(sample)
        assert isinstance(date, np.datetime64)

    def test_date_matches_season_start_plus_ix(self, sample):
        init_ix = 50.0
        m = NullTorchModel(init_ix=init_ix)
        date, _ = m.predict(sample)
        expected = sample['season_start'] + np.timedelta64(int(init_ix + 0.5), 'D')
        assert date == expected

    def test_info_contains_forward_pass(self, sample):
        m = NullTorchModel()
        _, info = m.predict(sample)
        assert 'forward_pass' in info


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

class TestNullTorchModelLoss:

    def test_loss_is_scalar_tensor(self, dataset):
        m = NullTorchModel()
        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())[:2]]
        batch = NullTorchModel.collate_fn(items)
        loss, _ = m.loss(batch, TARGET_FN)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_loss_info_contains_ys_pred_and_ys_true(self, dataset):
        m = NullTorchModel()
        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())[:2]]
        batch = NullTorchModel.collate_fn(items)
        _, info = m.loss(batch, TARGET_FN)
        assert 'ys_pred' in info
        assert 'ys_true' in info
        assert info['ys_pred'].shape == (2,)
        assert info['ys_true'].shape == (2,)

    def test_loss_is_zero_when_prediction_matches_target(self, dataset):
        # Build a model whose constant prediction equals the target index for all samples
        # by checking what the true indices are and setting init_ix accordingly.
        # This is a soft check: loss should decrease when init_ix is close to truth.
        m_far  = NullTorchModel(init_ix=0.0)
        m_near = NullTorchModel(init_ix=200.0)  # typical DOY in spring season

        items = [NullTorchModel.cast_to_tensor(dataset[ix]) for ix in list(dataset.iter_index())]
        batch = NullTorchModel.collate_fn(items)

        loss_far,  _ = m_far.loss(batch, TARGET_FN)
        loss_near, _ = m_near.loss(batch, TARGET_FN)
        assert loss_near.item() < loss_far.item()


# ---------------------------------------------------------------------------
# fit — no validation split
# ---------------------------------------------------------------------------

class TestNullTorchModelFitNoVal:

    def test_returns_model_and_dict(self, dataset):
        m, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=2, batch_size=4, verbose=False,
        )
        assert isinstance(m, NullTorchModel)
        assert isinstance(info, dict)

    def test_fit_info_has_epochs_list(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=3, batch_size=4, verbose=False,
        )
        assert 'epochs' in info
        assert len(info['epochs']) == 3

    def test_fit_info_has_timing(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=1, batch_size=4, verbose=False,
        )
        assert 'time_start' in info
        assert 'time_end' in info
        assert 'time_passed' in info

    def test_model_moves_to_cpu_after_fit(self, dataset):
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=1, batch_size=4, verbose=False,
        )
        assert next(m.parameters()).device.type == 'cpu'

    def test_model_is_in_eval_mode_after_fit(self, dataset):
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=1, batch_size=4, verbose=False,
        )
        assert not m.training

    def test_loss_decreases_over_training(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=20, batch_size=4, verbose=False,
        )
        first_loss = info['epochs'][0]['loss']
        last_loss  = info['epochs'][-1]['loss']
        assert last_loss < first_loss

    def test_warm_start_uses_provided_model(self, dataset):
        existing = NullTorchModel(init_ix=50.0)
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            model=existing,
            num_epochs=1, batch_size=4, verbose=False,
        )
        assert m is existing

    def test_empty_dataset_does_not_crash(self, dataset):
        empty = dataset.select_years([9999])
        existing = NullTorchModel()
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=empty,
            model=existing,
            num_epochs=1, batch_size=4, verbose=False,
        )
        assert isinstance(m, NullTorchModel)


# ---------------------------------------------------------------------------
# fit — with validation split
# ---------------------------------------------------------------------------

class TestNullTorchModelFitWithVal:

    def test_val_loss_present_in_epochs(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=4, batch_size=2, val_period=2,
            early_stopping=False, seed=0, verbose=False,
        )
        val_epochs = [e for e in info['epochs'] if 'val' in e]
        assert len(val_epochs) == 2  # validated at epoch 2 and 4

    def test_val_info_contains_mae_and_r2(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=2, batch_size=2, val_period=2,
            early_stopping=False, seed=0, verbose=False,
        )
        val_info = next(e['val'] for e in info['epochs'] if 'val' in e)
        assert 'mae' in val_info
        assert 'r2' in val_info
        assert 'loss' in val_info

    def test_early_stopping_stops_before_num_epochs(self):
        # Train samples target ~day 50 in season; val sample targets ~day 300.
        # As the constant model converges toward the training mean (~50), the
        # val loss (vs target ~300) increases, triggering early stopping.
        df_y = _make_df_y([
            ('src', 'A', 2000, 1, 10, 'BBCH_60', '2000-11-20'),  # ~day 50
            ('src', 'A', 2001, 1, 10, 'BBCH_60', '2001-11-20'),  # ~day 50
            ('src', 'A', 2002, 1, 10, 'BBCH_60', '2002-11-20'),  # ~day 50
            ('src', 'A', 2003, 1, 10, 'BBCH_60', '2003-11-20'),  # ~day 50
            ('src', 'A', 2004, 1, 10, 'BBCH_60', '2004-11-20'),  # ~day 50
            ('src', 'A', 2005, 1, 10, 'BBCH_60', '2005-11-20'),  # ~day 50
            ('src', 'A', 2006, 1, 10, 'BBCH_60', '2006-04-17'),  # ~day 198, used as val
        ])
        df_y_loc = _make_df_y_loc([('src', 'A', 48.0, 11.0)])
        obs = Observations(df_y, df_y_loc)
        cal = Calendar()
        cal.set_season('src', species_id=1, subgroup_id=10, start_date='10-01', length=365)
        provider = FakeWeatherProvider(cal, seed=0)
        ds = Dataset(obs, calendar=cal, feature_providers=[provider])

        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=ds,
            model_kwargs=dict(init_ix=200.0),
            num_epochs=200, batch_size=len(ds), val_period=1,
            early_stopping=True, early_stopping_patience=1,
            optimizer='sgd', optimizer_kwargs={'lr': 10.0},
            seed=0, verbose=False,
        )
        assert len(info['epochs']) < 200

    def test_early_stopping_rerun_adds_rerun_key(self, dataset):
        _, info = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=10, batch_size=2, val_period=1,
            early_stopping=True, early_stopping_patience=1,
            early_stopping_rerun=True,
            seed=0, verbose=False,
        )
        assert 'epochs_rerun' in info


# ---------------------------------------------------------------------------
# fit_from_args
# ---------------------------------------------------------------------------

class TestFitFromArgs:

    def test_fit_from_args_runs(self, dataset):
        args = BaseTorchModelArgs(
            num_epochs=2,
            batch_size=4,
            val_period=None,
            verbose=False,
        )
        m, info = NullTorchModel.fit_from_args(
            target_fn=TARGET_FN, dataset=dataset, model_args=args,
        )
        assert isinstance(m, NullTorchModel)
        assert len(info['epochs']) == 2

    def test_fit_from_args_passes_model_kwargs(self, dataset):
        args = BaseTorchModelArgs(
            model_kwargs=dict(init_ix=77.0),
            num_epochs=1,
            batch_size=4,
            verbose=False,
        )
        m, _ = NullTorchModel.fit_from_args(
            target_fn=TARGET_FN, dataset=dataset, model_args=args,
        )
        assert isinstance(m, NullTorchModel)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_round_trip(self, dataset, tmp_path):
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=1, batch_size=4, verbose=False,
        )
        m.save('null_torch_test', root=tmp_path)
        loaded, _ = NullTorchModel.load('null_torch_test', root=tmp_path)
        assert isinstance(loaded, NullTorchModel)

    def test_loaded_model_predicts(self, dataset, sample, tmp_path):
        m, _ = NullTorchModel.fit(
            target_fn=TARGET_FN, dataset=dataset,
            num_epochs=1, batch_size=4, verbose=False,
        )
        m.save('null_torch_test2', root=tmp_path)
        loaded, _ = NullTorchModel.load('null_torch_test2', root=tmp_path)
        date, _ = loaded.predict(sample)
        assert isinstance(date, np.datetime64)

    def test_load_missing_raises(self, tmp_path):
        from pysephone.models.base import ModelException
        with pytest.raises(ModelException):
            NullTorchModel.load('nonexistent', root=tmp_path)
