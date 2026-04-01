"""
Abstract base class for PyTorch-based phenology models.

Subclass :class:`BaseTorchModel`, implement :meth:`forward`, and training,
early stopping, and evaluation are inherited for free.

Example::

    from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs
    import torch.nn as nn

    class MyNet(BaseTorchModel):
        def __init__(self, hidden: int = 64):
            super().__init__()
            self.fc = nn.Linear(365, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(self, xs):
            t = xs['features']['temperature_2m_mean']   # (B, T)
            h = torch.relu(self.fc(t))
            ix = self.head(h).squeeze(-1)               # (B,)
            return ix, {}

    model, info = MyNet.fit(
        target_fn=lambda s: s['observations']['BBCH_60'],
        dataset=ds_train,
        model_kwargs=dict(hidden=128),
        num_epochs=50,
        batch_size=32,
        val_period=5,
    )
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pysephone.constants import KEY_FEATURES, KEY_OBSERVATIONS_INDEX
from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel, ModelArgs, ModelException
from pysephone.models.util.dataset_torch import TorchDataset
from pysephone.models.util.early_stopping import EarlyStopper
from pysephone.utils.func_torch import batch_tensors


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class BaseTorchModelArgs(ModelArgs):
    """Fitting-procedure arguments for :class:`BaseTorchModel`.

    Constructor kwargs (architecture hyperparameters) belong in
    ``model_kwargs``; these fields control the training loop.
    """
    num_epochs: int = 1
    batch_size: Optional[int] = None
    val_period: Optional[int] = None          # None → no validation split
    plot_period: Optional[int] = None         # None → never plot losses
    scheduler_step_size: Optional[int] = None # None → no LR decay
    scheduler_decay: float = 0.5
    clip_gradient: Optional[float] = None
    optimizer: str = 'adam'
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    early_stopping: bool = True
    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0
    early_stopping_rerun: bool = False
    device: str = 'cpu'                       # passed to torch.device()
    num_workers: int = 0
    pin_memory: bool = False
    seed: Optional[int] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

_OPTIMIZERS: Dict[str, type] = {
    'adam': torch.optim.Adam,
    'sgd':  torch.optim.SGD,
}


class BaseTorchModel(BaseModel, nn.Module):
    """Abstract base for PyTorch phenology models.

    Subclasses implement :meth:`forward`, which receives a collated batch dict
    and returns ``(ix_tensor, info_dict)`` where *ix_tensor* has shape
    ``(batch_size,)`` and contains predicted within-season day indices.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, xs: Dict[str, Any], **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            xs: Collated batch dict (output of :meth:`collate_fn`).

        Returns:
            ``(ix, info)`` where *ix* is a ``(batch_size,)`` float tensor of
            predicted within-season day indices, and *info* is a dict of
            auxiliary outputs.
        """

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        sample: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        **kwargs,
    ) -> Tuple[np.datetime64, Dict[str, Any]]:
        self.eval()
        with torch.no_grad():
            tensor_sample = self.__class__.cast_to_tensor(sample, device=device)
            batch = self.__class__.collate_fn([tensor_sample])
            ix_tensor, info = self(batch)
            [ix] = self._ixs_to_int(ix_tensor)
        date = sample['season_start'] + np.timedelta64(ix, 'D')
        return date, {'forward_pass': info}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_name: Optional[str] = None,
        model: Optional['BaseTorchModel'] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        val_period: Optional[int] = None,
        plot_period: Optional[int] = None,
        scheduler_step_size: Optional[int] = None,
        scheduler_decay: float = 0.5,
        clip_gradient: Optional[float] = None,
        optimizer: str = 'adam',
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 1,
        early_stopping_min_delta: float = 0.0,
        early_stopping_rerun: bool = False,
        device: torch.device = torch.device('cpu'),
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple['BaseTorchModel', Dict[str, Any]]:
        """Fit the model.

        Args:
            target_fn:    Callable extracting the ground-truth date from a sample.
            dataset:      Dataset to train on.
            model_name:   Optional name; defaults to the class name.
            model:        Optional warm-start instance.
            model_kwargs: Forwarded to ``cls(**model_kwargs)`` when *model* is None.
            num_epochs:   Training epochs.
            batch_size:   Mini-batch size (``None`` → full dataset per step).
            val_period:   Validate every *n* epochs.  ``None`` → train on full
                          dataset, no held-out validation.
            plot_period:  Save loss plots every *n* epochs.  ``None`` → never.
            scheduler_step_size: LR decay period in epochs.  ``None`` → no decay.
            scheduler_decay:     LR decay factor (default 0.5).
            clip_gradient:       Gradient clip value.  ``None`` → disabled.
            optimizer:    ``'adam'`` or ``'sgd'``.
            optimizer_kwargs: Extra kwargs forwarded to the optimizer constructor.
            early_stopping:          Enable early stopping on validation loss.
            early_stopping_patience: Patience in epochs.
            early_stopping_min_delta: Minimum loss improvement.
            early_stopping_rerun:    Retrain on full data for the same number of
                                     epochs after early stopping triggers.
            device:      Training device.
            num_workers: Number of worker processes for the DataLoader.
            pin_memory:  If ``True``, DataLoader copies tensors to pinned memory
                         before returning them (speeds up CPU→GPU transfers).
            seed:     Random seed for train/val split.
            verbose:  Show tqdm progress bars and metric output.

        Returns:
            ``(fitted_model, fit_info)``
        """
        super().fit(target_fn, dataset, model_name=model_name,
                    model=model, model_kwargs=model_kwargs)

        assert num_epochs > 0, "num_epochs must be > 0"

        if len(dataset) == 0:
            if model_kwargs is None:
                model_kwargs = {}
            return (model or cls(**model_kwargs)), {}

        if model_kwargs is None:
            model_kwargs = {}
        if optimizer_kwargs is None:
            optimizer_kwargs = cls._default_optimizer_kwargs(optimizer)

        batch_size = batch_size or len(dataset)
        scheduler_step_size = scheduler_step_size or num_epochs

        if val_period is None:
            dataset_trn, dataset_val = dataset, None
        else:
            dataset_trn, dataset_val = cls._split_dataset(dataset, seed=seed)

        model_init = model is None
        model = (model or cls(**model_kwargs)).to(device)
        model._batch_size = batch_size

        optimizer_inst = cls._make_optimizer(model, optimizer, optimizer_kwargs)
        scheduler = StepLR(optimizer_inst, step_size=scheduler_step_size, gamma=scheduler_decay)
        stopper = EarlyStopper(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        dl_trn = cls._make_dataloader(dataset_trn, batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory)
        dl_val = (cls._make_dataloader(dataset_val, batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
                  if dataset_val is not None else None)

        fit_info: Dict[str, Any] = {'epochs': []}
        time_start = datetime.now()
        final_epoch = 0

        for epoch in range(1, num_epochs + 1):
            epoch_info = model._run_train_epoch(
                target_fn=target_fn,
                dataloader=dl_trn,
                optimizer=optimizer_inst,
                scheduler=scheduler,
                epoch_nr=epoch,
                num_epochs=num_epochs,
                clip_gradient=clip_gradient,
                device=device,
                model_name=model_name or cls.__name__,
                verbose=verbose,
            )
            fit_info['epochs'].append(epoch_info)
            final_epoch += 1

            if dl_val is not None and (epoch % val_period) == 0:
                val_info = model._run_eval_epoch(
                    target_fn=target_fn,
                    dataloader=dl_val,
                    epoch_nr=epoch,
                    device=device,
                    model_name=model_name or cls.__name__,
                    verbose=verbose,
                )
                epoch_info['val'] = val_info

                if early_stopping and stopper.early_stop(val_info['loss'], epoch=epoch):
                    break

            if plot_period is not None and (epoch % plot_period) == 0:
                model._plot_losses(fit_info)

        if early_stopping and early_stopping_rerun and model_init and dataset_val is not None:
            model, rerun_info = cls._rerun_on_full_dataset(
                target_fn=target_fn,
                dataset=dataset,
                model_kwargs=model_kwargs,
                optimizer_name=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_step_size=scheduler_step_size,
                scheduler_decay=scheduler_decay,
                batch_size=batch_size,
                final_epoch=final_epoch,
                clip_gradient=clip_gradient,
                device=device,
                num_workers=num_workers,
                pin_memory=pin_memory,
                model_name=model_name or cls.__name__,
                verbose=verbose,
            )
            fit_info['epochs_rerun'] = rerun_info

        model.to(torch.device('cpu'))
        model.eval()

        time_end = datetime.now()
        fit_info.update(time_start=time_start, time_end=time_end,
                        time_passed=time_end - time_start)
        return model, fit_info

    @classmethod
    def fit_from_args(
        cls,
        target_fn: Callable[[Dict[str, Any]], Any],
        dataset: Dataset,
        model_args: BaseTorchModelArgs,
        model: Optional['BaseTorchModel'] = None,
        **kwargs,
    ) -> Tuple['BaseTorchModel', Dict[str, Any]]:
        """Fit from a :class:`BaseTorchModelArgs` instance."""
        return cls.fit(
            target_fn=target_fn,
            dataset=dataset,
            model_name=model_args.model_name or cls.__name__,
            model=model,
            model_kwargs=model_args.model_kwargs,
            num_epochs=model_args.num_epochs,
            batch_size=model_args.batch_size,
            val_period=model_args.val_period,
            plot_period=model_args.plot_period,
            scheduler_step_size=model_args.scheduler_step_size,
            scheduler_decay=model_args.scheduler_decay,
            clip_gradient=model_args.clip_gradient,
            optimizer=model_args.optimizer,
            optimizer_kwargs=model_args.optimizer_kwargs,
            early_stopping=model_args.early_stopping,
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_min_delta=model_args.early_stopping_min_delta,
            early_stopping_rerun=model_args.early_stopping_rerun,
            device=torch.device(model_args.device),
            num_workers=model_args.num_workers,
            pin_memory=model_args.pin_memory,
            seed=model_args.seed,
            verbose=model_args.verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _run_train_epoch(
        self,
        target_fn: Callable,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: StepLR,
        epoch_nr: int,
        num_epochs: int,
        clip_gradient: Optional[float],
        device: torch.device,
        model_name: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        self.train()
        time_start = datetime.now()
        losses = []
        it = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader

        for xs in it:
            xs = self._batch_to_device(xs, device)
            optimizer.zero_grad()
            loss, _ = self.loss(xs, target_fn)
            loss.backward()
            if clip_gradient is not None:
                nn.utils.clip_grad_value_(self.parameters(), clip_value=clip_gradient)
            optimizer.step()
            losses.append(loss.item())
            if verbose:
                lr = scheduler.get_last_lr()[0]
                it.set_description(
                    f'{model_name} epoch [{epoch_nr:5d}/{num_epochs}] '
                    f'lr={lr:.2e} loss={sum(losses)/len(losses):.5f}'
                )

        scheduler.step()
        time_end = datetime.now()
        return dict(epoch=epoch_nr, loss=sum(losses)/len(losses),
                    time_start=time_start, time_end=time_end,
                    time_passed=time_end - time_start)

    def _run_eval_epoch(
        self,
        target_fn: Callable,
        dataloader: DataLoader,
        epoch_nr: int,
        device: torch.device,
        model_name: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        self.eval()
        time_start = datetime.now()
        losses, ys_pred_all, ys_true_all = [], [], []
        it = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader

        with torch.no_grad():
            for xs in it:
                xs = self._batch_to_device(xs, device)
                loss, info = self.loss(xs, target_fn)
                losses.append(loss.item())
                ys_pred_all.append(info['ys_pred'].detach().cpu().numpy())
                ys_true_all.append(info['ys_true'].detach().cpu().numpy())
                if verbose:
                    it.set_description(
                        f'{model_name} val loss={sum(losses)/len(losses):.5f}'
                    )

        ys_pred = np.concatenate(ys_pred_all)
        ys_true = np.concatenate(ys_true_all)
        mae  = float(np.mean(np.abs(ys_pred - ys_true)))
        mse  = float(np.mean((ys_pred - ys_true) ** 2))
        ss_res = float(np.sum((ys_true - ys_pred) ** 2))
        ss_tot = float(np.sum((ys_true - ys_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        time_end = datetime.now()
        return dict(epoch=epoch_nr, loss=sum(losses)/len(losses),
                    mae=mae, mse=mse, r2=r2,
                    time_start=time_start, time_end=time_end,
                    time_passed=time_end - time_start)

    @classmethod
    def _rerun_on_full_dataset(
        cls,
        target_fn: Callable,
        dataset: Dataset,
        model_kwargs: Dict[str, Any],
        optimizer_name: str,
        optimizer_kwargs: Dict[str, Any],
        scheduler_step_size: int,
        scheduler_decay: float,
        batch_size: int,
        final_epoch: int,
        clip_gradient: Optional[float],
        device: torch.device,
        num_workers: int,
        pin_memory: bool,
        model_name: str,
        verbose: bool,
    ) -> Tuple['BaseTorchModel', List[Dict[str, Any]]]:
        new_model = cls(**model_kwargs).to(device)
        optimizer = cls._make_optimizer(new_model, optimizer_name, optimizer_kwargs)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_decay)
        dl = cls._make_dataloader(dataset, batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

        epoch_infos = []
        for epoch in range(1, final_epoch + 1):
            info = new_model._run_train_epoch(
                target_fn=target_fn,
                dataloader=dl,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_nr=epoch,
                num_epochs=final_epoch,
                clip_gradient=clip_gradient,
                device=device,
                model_name=model_name,
                verbose=verbose,
            )
            epoch_infos.append(info)

        return new_model, epoch_infos

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        xs: Dict[str, Any],
        target_fn: Callable[[Dict[str, Any]], Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """MSE loss between predicted and ground-truth within-season indices.

        Ground-truth is derived by applying *target_fn* to each sample and
        computing the offset in days from ``season_start``.
        """
        ys_pred, info = self(xs)

        # Build ground-truth index tensor from target_fn applied per sample
        season_starts = xs['season_start']  # list of np.datetime64
        ys_true_list = []
        for i, season_start in enumerate(season_starts):
            # Reconstruct a minimal sample dict for target_fn
            sample = {k: v[i] if isinstance(v, (list, torch.Tensor)) else v
                      for k, v in xs.items()}
            target_dt = np.datetime64(target_fn(sample), 'D')
            start_dt  = np.datetime64(season_start, 'D')
            ix = int((target_dt - start_dt) / np.timedelta64(1, 'D'))
            ys_true_list.append(float(ix))

        ys_true = torch.tensor(ys_true_list, dtype=ys_pred.dtype, device=ys_pred.device)
        loss = F.mse_loss(ys_pred, ys_true)
        return loss, {'forward_pass': info, 'ys_pred': ys_pred, 'ys_true': ys_true}

    # ------------------------------------------------------------------
    # Tensor / batch utilities
    # ------------------------------------------------------------------

    @classmethod
    def cast_to_tensor(
        cls,
        item: Dict[str, Any],
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, Any]:
        """Convert feature and observation arrays in *item* to tensors."""
        result = {k: v for k, v in item.items()
                  if k not in (KEY_FEATURES, KEY_OBSERVATIONS_INDEX)}

        if KEY_FEATURES in item:
            result[KEY_FEATURES] = {
                k: torch.tensor(v, dtype=dtype, device=device)
                for k, v in item[KEY_FEATURES].items()
            }
        if KEY_OBSERVATIONS_INDEX in item:
            result[KEY_OBSERVATIONS_INDEX] = {
                k: torch.tensor(v, dtype=dtype, device=device)
                for k, v in item[KEY_OBSERVATIONS_INDEX].items()
            }
        return result

    @classmethod
    def collate_fn(cls, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of tensor samples into a batched dict."""
        assert len(items) > 0

        batch: Dict[str, Any] = defaultdict(list)
        for item in items:
            for k, v in item.items():
                if k not in (KEY_FEATURES, KEY_OBSERVATIONS_INDEX):
                    batch[k].append(v)
        batch = dict(batch)

        if KEY_FEATURES in items[0]:
            batch[KEY_FEATURES] = {
                k: batch_tensors(*[item[KEY_FEATURES][k] for item in items])
                for k in items[0][KEY_FEATURES]
            }
        if KEY_OBSERVATIONS_INDEX in items[0]:
            batch[KEY_OBSERVATIONS_INDEX] = {
                k: batch_tensors(*[item[KEY_OBSERVATIONS_INDEX][k] for item in items])
                for k in items[0][KEY_OBSERVATIONS_INDEX]
            }
        return batch

    # ------------------------------------------------------------------
    # Save / load  (torch-specific, overrides pickle-based BaseModel)
    # ------------------------------------------------------------------

    def save(self, model_name: str, root=None) -> None:
        """Save the full model with ``torch.save``."""
        from pysephone.paths import get_data_root, get_model_dir
        model_dir = get_model_dir(root or get_data_root(), model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self, model_dir / f'{model_name}.pt')

    @classmethod
    def load(cls, model_name: str, root=None):
        """Load a model saved with :meth:`save`."""
        from pysephone.paths import get_data_root, get_model_dir
        from pysephone.models.base import ModelException
        path = get_model_dir(root or get_data_root(), model_name) / f'{model_name}.pt'
        if not path.exists():
            raise ModelException(f"Model file not found: {path}")
        model = torch.load(path, weights_only=False)
        return model, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _make_dataloader(
        cls,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        items = [cls.cast_to_tensor(item) for item in dataset.iter_items()]
        return DataLoader(TorchDataset(items), batch_size=batch_size,
                          collate_fn=cls.collate_fn, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    @classmethod
    def _split_dataset(cls, dataset: Dataset, seed: Optional[int]) -> Tuple[Dataset, Dataset]:
        yrs_trn, yrs_val = train_test_split(dataset.years, random_state=seed, shuffle=True)
        return dataset.select_years(yrs_trn), dataset.select_years(yrs_val)

    @classmethod
    def _make_optimizer(
        cls, model: 'BaseTorchModel', name: str, kwargs: Dict[str, Any]
    ) -> optim.Optimizer:
        if name not in _OPTIMIZERS:
            raise ModelException(f"Unknown optimizer {name!r}. Choose from: {list(_OPTIMIZERS)}")
        return _OPTIMIZERS[name](model.parameters(), **kwargs)

    @classmethod
    def _default_optimizer_kwargs(cls, optimizer: str) -> Dict[str, Any]:
        if optimizer == 'adam':
            return {'lr': 1e-3, 'weight_decay': 1e-4}
        return {'lr': 1e-3}

    @staticmethod
    def _ixs_to_int(ixs: torch.Tensor) -> List[int]:
        return [int(ix.item() + 0.5) for ix in ixs]

    @staticmethod
    def _batch_to_device(batch: Any, device: torch.device) -> Any:
        """Recursively move all tensors in *batch* to *device*."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: BaseTorchModel._batch_to_device(v, device) for k, v in batch.items()}
        return batch

    def _plot_losses(self, fit_info: Dict[str, Any]) -> None:
        """Return a Figure of train/val loss curves (does not save to disk)."""
        from matplotlib import pyplot as plt
        epochs    = [e['epoch'] for e in fit_info['epochs']]
        trn_loss  = [e['loss']  for e in fit_info['epochs']]
        val_epochs = [e['epoch'] for e in fit_info['epochs'] if 'val' in e]
        val_loss   = [e['val']['loss'] for e in fit_info['epochs'] if 'val' in e]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, trn_loss, label='Train')
        if val_loss:
            ax.plot(val_epochs, val_loss, label='Val')
            ax.axhline(min(val_loss), color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.tight_layout()
        return fig
