"""
Single-target regression evaluation for phenology models.

Evaluates a :class:`~pysephone.models.base.BaseModel` against train and test
:class:`~pysephone.dataset.dataset.Dataset` instances, stores predictions to
disk so metrics can be recomputed without re-running inference, and produces
diagnostic plots.

Example::

    from pysephone.evaluation.regression import SingleTargetRegression

    eval_obj = SingleTargetRegression.run(
        model=model,
        dataset_train=ds_train,
        dataset_test=ds_test,
        target_fn=lambda s: s['observations']['BBCH_11'],
        run_name='my_experiment',
    )
    metrics = eval_obj.compute_metrics()
    fig = eval_obj.plot_scatter()
    eval_obj.save()

    # Later, reload without re-running inference:
    eval_obj = SingleTargetRegression.load('my_experiment')
    metrics = eval_obj.compute_metrics()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pysephone.dataset.dataset import Dataset
from pysephone.models.base import BaseModel
from pysephone.paths import get_data_root, get_evaluations_dir


# ---------------------------------------------------------------------------
# Style (mirrors visualize/dataset.py)
# ---------------------------------------------------------------------------

_STYLE = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
}

_COLOUR_TRAIN = '#4878d0'
_COLOUR_TEST  = '#ee854a'


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _datetime64_to_doy(dt: np.datetime64) -> float:
    """Convert np.datetime64 to day-of-year (1-indexed)."""
    day = dt.astype('datetime64[D]')
    year_start = day.astype('datetime64[Y]').astype('datetime64[D]')
    return float((day - year_start) / np.timedelta64(1, 'D')) + 1.0


def _collect_predictions(
    model: BaseModel,
    dataset: Dataset,
    target_fn: Callable[[Dict[str, Any]], np.datetime64],
) -> pd.DataFrame:
    """Run model.predict over every sample and return a DataFrame.

    Columns: year, predicted_doy, observed_doy, error (pred - obs).
    """
    rows = []
    for sample in dataset.iter_items():
        observed_dt = target_fn(sample)
        predicted_dt, _ = model.predict(sample)
        obs_doy  = _datetime64_to_doy(np.datetime64(observed_dt,  'D'))
        pred_doy = _datetime64_to_doy(np.datetime64(predicted_dt, 'D'))
        rows.append({
            'year':          int(sample['year']),
            'predicted_doy': pred_doy,
            'observed_doy':  obs_doy,
            'error':         pred_doy - obs_doy,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EvaluationException(Exception):
    pass


class SingleTargetRegression:
    """Stores and analyses single-target regression predictions.

    Attributes:
        run_name:  Identifier used for saving/loading.
        df_train:  Predictions on the training set (columns: year,
                   predicted_doy, observed_doy, error).
        df_test:   Predictions on the test set (same columns; may be empty).
        metadata:  Arbitrary dict of extra info (model class, dataset sizes …).
    """

    _TRAIN_CSV = 'train_predictions.csv'
    _TEST_CSV  = 'test_predictions.csv'
    _META_JSON = 'metadata.json'

    def __init__(
        self,
        run_name: str,
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name = run_name
        self.df_train = df_train.reset_index(drop=True)
        self.df_test  = (df_test.reset_index(drop=True)
                         if df_test is not None and not df_test.empty
                         else pd.DataFrame(columns=['year', 'predicted_doy', 'observed_doy', 'error']))
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def run(
        cls,
        model: BaseModel,
        dataset_train: Dataset,
        target_fn: Callable[[Dict[str, Any]], np.datetime64],
        run_name: str,
        dataset_test: Optional[Dataset] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> 'SingleTargetRegression':
        """Run inference and return a :class:`SingleTargetRegression` object.

        Args:
            model:          Fitted model implementing :class:`~pysephone.models.base.BaseModel`.
            dataset_train:  Training dataset.
            target_fn:      Callable ``(sample) -> np.datetime64`` extracting ground truth.
            run_name:       Identifier for this evaluation run.
            dataset_test:   Optional test dataset.
            extra_metadata: Extra key/value pairs stored alongside results.

        Returns:
            A :class:`SingleTargetRegression` instance with predictions populated.
        """
        df_train = _collect_predictions(model, dataset_train, target_fn)
        df_test  = _collect_predictions(model, dataset_test, target_fn) if dataset_test is not None else None

        metadata: Dict[str, Any] = {
            'model_class':      type(model).__name__,
            'n_train':          len(df_train),
            'n_test':           len(df_test) if df_test is not None else 0,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return cls(run_name=run_name, df_train=df_train, df_test=df_test, metadata=metadata)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _eval_dir(self, root: Optional[Path]) -> Path:
        base = get_evaluations_dir(root or get_data_root())
        return base / self.run_name

    def save(self, root: Optional[Path] = None) -> None:
        """Save predictions (CSV) and metadata (JSON) to disk.

        Args:
            root: Data root path.  Defaults to :func:`~pysephone.paths.get_data_root`.
        """
        d = self._eval_dir(root)
        d.mkdir(parents=True, exist_ok=True)
        self.df_train.to_csv(d / self._TRAIN_CSV, index=False)
        self.df_test.to_csv(d / self._TEST_CSV, index=False)
        with open(d / self._META_JSON, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, run_name: str, root: Optional[Path] = None) -> 'SingleTargetRegression':
        """Load a previously saved evaluation from disk.

        Args:
            run_name: Identifier used when :meth:`save` was called.
            root:     Data root path.  Defaults to :func:`~pysephone.paths.get_data_root`.

        Raises:
            EvaluationException: If the expected files are not found.
        """
        d = get_evaluations_dir(root or get_data_root()) / run_name
        train_path = d / cls._TRAIN_CSV
        test_path  = d / cls._TEST_CSV
        meta_path  = d / cls._META_JSON
        if not train_path.exists():
            raise EvaluationException(f"Training predictions not found: {train_path}")
        df_train = pd.read_csv(train_path)
        df_test: Optional[pd.DataFrame] = None
        if test_path.exists():
            try:
                candidate = pd.read_csv(test_path)
                if not candidate.empty:
                    df_test = candidate
            except Exception:
                pass
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        return cls(run_name=run_name, df_train=df_train, df_test=df_test, metadata=metadata)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_for(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        errors = df['error'].values
        obs    = df['observed_doy'].values
        pred   = df['predicted_doy'].values
        n      = len(errors)
        mae    = float(np.mean(np.abs(errors)))
        mse    = float(np.mean(errors ** 2))
        rmse   = float(np.sqrt(mse))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        corr   = float(np.corrcoef(obs, pred)[0, 1]) if n > 1 else float('nan')
        # Kendall tau via rank correlation (scipy not required)
        try:
            from scipy.stats import kendalltau
            tau = float(kendalltau(obs, pred).statistic)
        except Exception:
            tau = float('nan')
        bias = float(np.mean(errors))
        return dict(n=n, mae=mae, mse=mse, rmse=rmse, r2=r2, pearson_r=corr,
                    kendall_tau=tau, bias=bias)

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute regression metrics for train and (if available) test splits.

        Returns:
            ``{'train': {...}, 'test': {...}}`` — each inner dict contains:
            ``n``, ``mae``, ``mse``, ``rmse``, ``r2``, ``pearson_r``,
            ``kendall_tau``, ``bias`` (all in days, where applicable).
        """
        return {
            'train': self._metrics_for(self.df_train),
            'test':  self._metrics_for(self.df_test),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_scatter(self, title: Optional[str] = None) -> Figure:
        """Predicted vs. observed DOY scatter (train and test on the same axes).

        Returns:
            :class:`matplotlib.figure.Figure`
        """
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(5, 5))

            all_doys = pd.concat(
                [d[['predicted_doy', 'observed_doy']]
                 for d in [self.df_train, self.df_test] if not d.empty]
            )
            lo = all_doys.min().min()
            hi = all_doys.max().max()
            pad = (hi - lo) * 0.05
            lim = (lo - pad, hi + pad)

            ax.plot(lim, lim, color='#888888', linewidth=1, linestyle='--', zorder=0)

            if not self.df_train.empty:
                m = self._metrics_for(self.df_train)
                ax.scatter(
                    self.df_train['observed_doy'],
                    self.df_train['predicted_doy'],
                    s=18, alpha=0.55, color=_COLOUR_TRAIN,
                    label=f"Train  RMSE={m['rmse']:.1f} d  R²={m['r2']:.3f}",
                    zorder=2,
                )
            if not self.df_test.empty:
                m = self._metrics_for(self.df_test)
                ax.scatter(
                    self.df_test['observed_doy'],
                    self.df_test['predicted_doy'],
                    s=18, alpha=0.55, color=_COLOUR_TEST,
                    label=f"Test   RMSE={m['rmse']:.1f} d  R²={m['r2']:.3f}",
                    zorder=3,
                )

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel('Observed DOY')
            ax.set_ylabel('Predicted DOY')
            ax.set_title(title or f'{self.run_name} — predicted vs. observed')
            ax.legend(fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
        return fig

    def plot_residuals_over_time(self, title: Optional[str] = None) -> Figure:
        """Residuals (predicted − observed, in days) plotted against year.

        A horizontal zero line and ±1 MAE band are drawn for reference.

        Returns:
            :class:`matplotlib.figure.Figure`
        """
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axhline(0, color='#888888', linewidth=1, linestyle='--', zorder=0)

            if not self.df_train.empty:
                m = self._metrics_for(self.df_train)
                ax.scatter(
                    self.df_train['year'], self.df_train['error'],
                    s=16, alpha=0.5, color=_COLOUR_TRAIN,
                    label=f'Train (bias={m["bias"]:+.1f} d)', zorder=2,
                )
            if not self.df_test.empty:
                m = self._metrics_for(self.df_test)
                ax.scatter(
                    self.df_test['year'], self.df_test['error'],
                    s=16, alpha=0.5, color=_COLOUR_TEST,
                    label=f'Test (bias={m["bias"]:+.1f} d)', zorder=3,
                )

            ax.set_xlabel('Year')
            ax.set_ylabel('Error (pred − obs, days)')
            ax.set_title(title or f'{self.run_name} — residuals over time')
            ax.legend(fontsize=9)
            fig.tight_layout()
        return fig

    def plot_error_distribution(self, bins: int = 30, title: Optional[str] = None) -> Figure:
        """Histogram of prediction errors for train and test splits.

        Returns:
            :class:`matplotlib.figure.Figure`
        """
        has_test = not self.df_test.empty
        with plt.rc_context(_STYLE):
            ncols = 2 if has_test else 1
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=False)
            if ncols == 1:
                axes = [axes]

            for ax, (df, label, colour) in zip(
                axes,
                [(self.df_train, 'Train', _COLOUR_TRAIN)]
                + ([(self.df_test, 'Test', _COLOUR_TEST)] if has_test else []),
            ):
                if df.empty:
                    continue
                m = self._metrics_for(df)
                ax.hist(df['error'], bins=bins, color=colour, alpha=0.75, edgecolor='white')
                ax.axvline(0,            color='#444444', linewidth=1.2, linestyle='--')
                ax.axvline(m['bias'],    color=colour,    linewidth=1.5, linestyle='-',
                           label=f'Bias {m["bias"]:+.1f} d')
                ax.set_xlabel('Error (pred − obs, days)')
                ax.set_ylabel('Count')
                ax.set_title(f'{label}  MAE={m["mae"]:.1f} d  RMSE={m["rmse"]:.1f} d')
                ax.legend(fontsize=9)

            fig.suptitle(title or f'{self.run_name} — error distribution', y=1.01)
            fig.tight_layout()
        return fig

    def plot_annual_mean_doy(self, title: Optional[str] = None) -> Figure:
        """Year-grouped mean observed vs. predicted DOY — useful for detecting
        systematic temporal drift.

        Returns:
            :class:`matplotlib.figure.Figure`
        """
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))

            for df, label, colour in [
                (self.df_train, 'Train', _COLOUR_TRAIN),
                (self.df_test,  'Test',  _COLOUR_TEST),
            ]:
                if df.empty:
                    continue
                grp = df.groupby('year')[['observed_doy', 'predicted_doy']].mean()
                ax.plot(grp.index, grp['observed_doy'],  color=colour, linewidth=1.4,
                        linestyle='--', alpha=0.7, label=f'{label} observed')
                ax.plot(grp.index, grp['predicted_doy'], color=colour, linewidth=1.8,
                        linestyle='-',  label=f'{label} predicted')

            ax.set_xlabel('Year')
            ax.set_ylabel('Mean DOY')
            ax.set_title(title or f'{self.run_name} — annual mean DOY')
            ax.legend(fontsize=9)
            fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_tr = len(self.df_train)
        n_te = len(self.df_test)
        return (
            f"SingleTargetRegression(run_name={self.run_name!r}, "
            f"n_train={n_tr}, n_test={n_te})"
        )
