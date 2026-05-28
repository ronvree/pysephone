"""
Model fitting and hyperparameter optimisation for BloomBench.

Holds:

* :data:`MODELS` — mapping ``model_key -> (model_class, fit_dispatcher)``.
* :func:`fit_one` — single public entry that picks the right dispatcher.
* :func:`run_hpo` — top-level HPO entrypoint used by the CLI.
* Internal HP cache I/O, tree HPO dispatcher, torch HPO random search and
  the per-model search spaces.

Tree models (``RandomForest``, ``XGBoost``) reuse the in-model
``hyperparameter_search=True`` plumbing via
:func:`sklearn.model_selection.RandomizedSearchCV` with year-aware
:class:`~sklearn.model_selection.GroupKFold`.  Torch models
(``CNN1D``, ``LSTM``, ``Transformer``) use a plain random search where each
trial fits on ``ds_train[:80%]`` and is scored on ``ds_train[80%:]`` by MAE.
Best params are cached as JSON under :func:`config.hp_cache_dir`.

Feature statistics for torch models are computed per-dataset in
:func:`compute_feature_stats` and injected into ``model_kwargs`` so the
benchmark works against AgERA5 (Kelvin / different keys) without touching
the model layer.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

from pysephone.constants import KEY_YEAR
from pysephone.dataset.dataset import Dataset
from pysephone.evaluation.regression import SingleTargetRegression
from pysephone.models.base import BaseModel
from pysephone.models.cnn_1d import CNN1DModel
from pysephone.models.linear_trend import LinearTrendModel
from pysephone.models.lstm import LSTMModel
from pysephone.models.mean import MeanModel
from pysephone.models.random_forest import RandomForestModel
from pysephone.models.transformer import TransformerModel
from pysephone.models.xgb import XGBoostModel

from pysephone.benchmarks.bloombench import config as _cfg


# ---------------------------------------------------------------------------
# sklearn 1.8 noise suppression (same as the original notebook config)
# ---------------------------------------------------------------------------

_SKLEARN_DELAYED_RE = r'`sklearn\.utils\.parallel\.delayed`'
warnings.filterwarnings('ignore', message=_SKLEARN_DELAYED_RE, category=UserWarning)
os.environ.setdefault('PYTHONWARNINGS', f'ignore:{_SKLEARN_DELAYED_RE}:UserWarning')


# ---------------------------------------------------------------------------
# HP cache I/O (atomic writes via tempfile + os.replace)
# ---------------------------------------------------------------------------

def _hp_cache_path(dataset_name: str, model_key: str, *, root: Optional[Path] = None) -> Path:
    return _cfg.hp_cache_dir(root) / f'{dataset_name}_{model_key}.json'


def _hp_trials_path(dataset_name: str, model_key: str, *, root: Optional[Path] = None) -> Path:
    return _cfg.hp_cache_dir(root) / f'{dataset_name}_{model_key}_trials.json'


def _json_safe(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=2, default=_json_safe)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_hp_cache(
    dataset_name: str,
    model_key: str,
    *,
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached best-params dict for (*dataset_name*, *model_key*) or ``None``."""
    p = _hp_cache_path(dataset_name, model_key, root=root)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - cache corruption is non-fatal
        print(f'    HP cache read failed for {p.name}: {type(exc).__name__}: {exc}')
        return None


def save_hp_cache(
    dataset_name: str,
    model_key: str,
    params: Dict[str, Any],
    *,
    root: Optional[Path] = None,
) -> None:
    """Atomically persist *params* as the best-params for (*dataset_name*, *model_key*)."""
    _atomic_write_json(_hp_cache_path(dataset_name, model_key, root=root), params)


def load_hp_trials(
    dataset_name: str,
    model_key: str,
    *,
    root: Optional[Path] = None,
) -> Optional[list]:
    """Return the per-trial log for a torch HPO run, or ``None`` if absent."""
    p = _hp_trials_path(dataset_name, model_key, root=root)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save_hp_trials(
    dataset_name: str,
    model_key: str,
    trials: list,
    *,
    root: Optional[Path] = None,
) -> None:
    """Atomically persist a torch HPO trial log."""
    _atomic_write_json(_hp_trials_path(dataset_name, model_key, root=root), trials)


# ---------------------------------------------------------------------------
# Tree HPO (RandomForest / XGBoost)
# ---------------------------------------------------------------------------

def _fit_tree(
    model_cls,
    target_fn,
    ds_train,
    *,
    seed: int,
    dataset_name: str,
    model_key: str,
    feature_keys: List[str],
    run_hpo: bool,
    force_retune: bool,
    n_iter_search: int,
    cv_folds: int,
) -> BaseModel:
    cached = None if force_retune else load_hp_cache(dataset_name, model_key)

    if cached is not None:
        model_kwargs = dict(data_keys=feature_keys, random_state=seed, **cached)
        model, _ = model_cls.fit(
            target_fn=target_fn, dataset=ds_train,
            model_kwargs=model_kwargs,
            hyperparameter_search=False, random_state=seed,
        )
        return model

    if not run_hpo:
        model_kwargs = dict(data_keys=feature_keys, random_state=seed)
        model, _ = model_cls.fit(
            target_fn=target_fn, dataset=ds_train,
            model_kwargs=model_kwargs,
            hyperparameter_search=False, random_state=seed,
        )
        return model

    # Fresh HPO with year-aware CV
    model_kwargs = dict(data_keys=feature_keys, random_state=seed)
    model, info = model_cls.fit(
        target_fn=target_fn, dataset=ds_train,
        model_kwargs=model_kwargs,
        hyperparameter_search=True,
        n_iter_search=n_iter_search,
        cv=GroupKFold(n_splits=cv_folds),
        cv_group_by=KEY_YEAR,
        random_state=seed,
    )
    if 'best_params' in info:
        save_hp_cache(dataset_name, model_key, info['best_params'])
    return model


# ---------------------------------------------------------------------------
# Torch HPO (CNN1D / LSTM / Transformer)
# ---------------------------------------------------------------------------

def _split_train_for_hpo(ds_train: Dataset, val_fraction: float) -> Tuple[Dataset, Dataset]:
    """Year-cutoff split of *ds_train* for HPO validation.

    The val fold takes the last ``val_fraction`` of years so the HPO
    train/val split shares the temporal-future-as-validation structure of
    the benchmark's outer train/test split.
    """
    years_sorted = sorted(set(ds_train.years))
    cutoff_ix = int(len(years_sorted) * (1.0 - val_fraction))
    cutoff_ix = max(1, min(cutoff_ix, len(years_sorted) - 1))
    cutoff = years_sorted[cutoff_ix]
    years_trn = [y for y in years_sorted if y < cutoff]
    years_val = [y for y in years_sorted if y >= cutoff]
    return ds_train.select_years(years_trn), ds_train.select_years(years_val)


def _lstm_search_space(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        'hidden_size':   int(rng.choice([32, 64, 128, 256])),
        'num_layers':    int(rng.choice([1, 2, 3])),
        'learning_rate': float(10 ** rng.uniform(-4, -2)),
    }


def _transformer_search_space(rng: np.random.Generator) -> Dict[str, Any]:
    hidden_size = int(rng.choice([64, 128, 256]))
    valid_nheads = [n for n in [2, 4, 8] if hidden_size % n == 0]
    return {
        'hidden_size':     hidden_size,
        'num_layers':      int(rng.choice([1, 2, 3])),
        'nhead':           int(rng.choice(valid_nheads)),
        'dim_feedforward': 2 * hidden_size,
        'learning_rate':   float(10 ** rng.uniform(-4, -2)),
    }


def _cnn_search_space(rng: np.random.Generator) -> Dict[str, Any]:
    # Constrain (kernel_size, num_layers) to receptive field >= season length
    # so the CNN can see a full season.
    candidates = []
    for k in (3, 5):
        for L in range(4, 11):
            rf = 1 + (k - 1) * (2 ** L - 1)
            if rf >= 365:
                candidates.append((k, L))
    k, L = candidates[int(rng.integers(len(candidates)))]
    return {
        'hidden_size':   int(rng.choice([32, 64, 128])),
        'num_layers':    L,
        'kernel_size':   k,
        'dilation_base': 2,
        'learning_rate': float(10 ** rng.uniform(-4, -2)),
    }


def _torch_dispatch(
    trial_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Split a trial config into (arch_kwargs, opt_kwargs)."""
    arch = {k: v for k, v in trial_config.items() if k != 'learning_rate'}
    opt_kwargs = (
        dict(lr=float(trial_config['learning_rate']), weight_decay=1e-4)
        if 'learning_rate' in trial_config else None
    )
    return arch, opt_kwargs


def _torch_hpo(
    model_cls,
    target_fn,
    ds_train,
    *,
    search_space_fn,
    n_trials: int,
    seed: int,
    feature_keys: List[str],
    feature_stats: Optional[Dict[str, Tuple[float, float]]],
    val_fraction: float,
    train_kwargs: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    ds_trn_prime, ds_val_cutoff = _split_train_for_hpo(ds_train, val_fraction)
    if len(ds_val_cutoff) == 0 or len(ds_trn_prime) == 0:
        return None, []

    rng = np.random.default_rng(seed)
    trials: List[Dict[str, Any]] = []

    for i in range(n_trials):
        trial_config = search_space_fn(rng)
        arch, opt_kwargs = _torch_dispatch(trial_config)
        t0 = time.time()
        try:
            extra: Dict[str, Any] = dict(data_keys=feature_keys)
            if feature_stats is not None:
                extra['feature_statistics'] = feature_stats
            model, _info = model_cls.fit(
                target_fn=target_fn, dataset=ds_trn_prime,
                model_kwargs=dict(extra, **arch),
                seed=seed,
                optimizer_kwargs=opt_kwargs,
                verbose=False,
                **train_kwargs,
            )
            res = SingleTargetRegression.run(
                model=model,
                dataset_train=ds_trn_prime,
                dataset_test=ds_val_cutoff,
                target_fn=target_fn,
                run_name='_hpo_trial_tmp',
            )
            val_mae = float(res.compute_metrics()['test'].get('mae', float('inf')))
            err: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 - per-trial isolation
            val_mae = float('inf')
            err = f'{type(exc).__name__}: {exc}'
        trials.append(dict(
            trial=i, config=trial_config, val_mae=val_mae,
            seconds=round(time.time() - t0, 1), error=err,
        ))
        if verbose:
            marker = '   '
            ok_trials = [t for t in trials if t['error'] is None]
            if err is None and ok_trials and val_mae == min(t['val_mae'] for t in ok_trials):
                marker = '* '
            print(
                f'      trial {i:2d}{marker}MAE_val={val_mae:.3f}  '
                f'{trial_config}  ({trials[-1]["seconds"]}s)'
            )

    finite = [t for t in trials if t['error'] is None and np.isfinite(t['val_mae'])]
    if not finite:
        return None, trials
    best = min(finite, key=lambda t: t['val_mae'])
    return best['config'], trials


def _fit_torch(
    model_cls,
    target_fn,
    ds_train,
    *,
    seed: int,
    dataset_name: str,
    model_key: str,
    feature_keys: List[str],
    feature_stats: Optional[Dict[str, Tuple[float, float]]],
    search_space_fn,
    default_arch: Dict[str, Any],
    run_hpo: bool,
    force_retune: bool,
    n_trials_torch: int,
    val_fraction: float,
    train_kwargs: Dict[str, Any],
    verbose: bool = True,
) -> BaseModel:
    cached = None if force_retune else load_hp_cache(dataset_name, model_key)

    if cached is None and run_hpo:
        if verbose:
            print(f'    [{model_key}] running {n_trials_torch} random-search trials...')
        best_config, trials = _torch_hpo(
            model_cls, target_fn, ds_train,
            search_space_fn=search_space_fn,
            n_trials=n_trials_torch,
            seed=seed,
            feature_keys=feature_keys,
            feature_stats=feature_stats,
            val_fraction=val_fraction,
            train_kwargs=train_kwargs,
            verbose=verbose,
        )
        save_hp_trials(dataset_name, model_key, trials)
        if best_config is not None:
            save_hp_cache(dataset_name, model_key, best_config)
            cached = best_config

    arch_kwargs = dict(default_arch)
    if cached is not None:
        arch, opt_kwargs = _torch_dispatch(cached)
        arch_kwargs.update(arch)
    else:
        opt_kwargs = None

    model_kwargs: Dict[str, Any] = dict(data_keys=feature_keys, **arch_kwargs)
    if feature_stats is not None:
        model_kwargs['feature_statistics'] = feature_stats

    model, _ = model_cls.fit(
        target_fn=target_fn, dataset=ds_train,
        model_kwargs=model_kwargs,
        seed=seed,
        optimizer_kwargs=opt_kwargs,
        **train_kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# Per-model dispatchers (signature: see fit_one below)
# ---------------------------------------------------------------------------

FitDispatcher = Callable[..., BaseModel]


def _fit_mean(target_fn, ds_train, *, seed, dataset_name, model_key, **_) -> BaseModel:
    model, _ = MeanModel.fit(target_fn=target_fn, dataset=ds_train)
    return model


def _fit_linear(target_fn, ds_train, *, seed, dataset_name, model_key, **_) -> BaseModel:
    model, _ = LinearTrendModel.fit(target_fn=target_fn, dataset=ds_train)
    return model


def _fit_rf(target_fn, ds_train, *, seed, dataset_name, model_key, **opts) -> BaseModel:
    return _fit_tree(
        RandomForestModel, target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=opts['feature_keys'],
        run_hpo=opts['run_hpo_trees'],
        force_retune=opts['force_retune'],
        n_iter_search=opts['n_iter_trees'],
        cv_folds=opts['cv_folds'],
    )


def _fit_xgb(target_fn, ds_train, *, seed, dataset_name, model_key, **opts) -> BaseModel:
    return _fit_tree(
        XGBoostModel, target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=opts['feature_keys'],
        run_hpo=opts['run_hpo_trees'],
        force_retune=opts['force_retune'],
        n_iter_search=opts['n_iter_trees'],
        cv_folds=opts['cv_folds'],
    )


def _fit_cnn(target_fn, ds_train, *, seed, dataset_name, model_key, **opts) -> BaseModel:
    return _fit_torch(
        CNN1DModel, target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=opts['feature_keys'],
        feature_stats=opts.get('feature_stats'),
        search_space_fn=_cnn_search_space,
        default_arch=dict(),  # rely on model defaults
        run_hpo=opts['run_hpo_torch'],
        force_retune=opts['force_retune'],
        n_trials_torch=opts['n_trials_torch'],
        val_fraction=opts['val_fraction'],
        train_kwargs=opts['train_kwargs'],
        verbose=opts.get('verbose', True),
    )


def _fit_lstm(target_fn, ds_train, *, seed, dataset_name, model_key, **opts) -> BaseModel:
    return _fit_torch(
        LSTMModel, target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=opts['feature_keys'],
        feature_stats=opts.get('feature_stats'),
        search_space_fn=_lstm_search_space,
        default_arch=dict(hidden_size=64, num_layers=2),
        run_hpo=opts['run_hpo_torch'],
        force_retune=opts['force_retune'],
        n_trials_torch=opts['n_trials_torch'],
        val_fraction=opts['val_fraction'],
        train_kwargs=opts['train_kwargs'],
        verbose=opts.get('verbose', True),
    )


def _fit_transformer(target_fn, ds_train, *, seed, dataset_name, model_key, **opts) -> BaseModel:
    return _fit_torch(
        TransformerModel, target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=opts['feature_keys'],
        feature_stats=opts.get('feature_stats'),
        search_space_fn=_transformer_search_space,
        default_arch=dict(hidden_size=64, num_layers=2, nhead=4, dim_feedforward=128),
        run_hpo=opts['run_hpo_torch'],
        force_retune=opts['force_retune'],
        n_trials_torch=opts['n_trials_torch'],
        val_fraction=opts['val_fraction'],
        train_kwargs=opts['train_kwargs'],
        verbose=opts.get('verbose', True),
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

#: Maps ``model_key -> (model_class, fit_dispatcher)``.  Order is the canonical
#: column order for the results table.
MODELS: Dict[str, Tuple[type, FitDispatcher]] = {
    'Mean':         (MeanModel,         _fit_mean),
    'Linear':       (LinearTrendModel,  _fit_linear),
    'RandomForest': (RandomForestModel, _fit_rf),
    'XGBoost':      (XGBoostModel,      _fit_xgb),
    'CNN1D':        (CNN1DModel,        _fit_cnn),
    'LSTM':         (LSTMModel,         _fit_lstm),
    'Transformer':  (TransformerModel,  _fit_transformer),
}


_TORCH_MODEL_KEYS = {'CNN1D', 'LSTM', 'Transformer'}
_TREE_MODEL_KEYS = {'RandomForest', 'XGBoost'}


def is_torch_model(model_key: str) -> bool:
    """Return ``True`` if *model_key* names a PyTorch-based BloomBench model."""
    return model_key in _TORCH_MODEL_KEYS


def is_tree_model(model_key: str) -> bool:
    """Return ``True`` if *model_key* names a tree-based BloomBench model."""
    return model_key in _TREE_MODEL_KEYS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_one(
    model_key: str,
    target_fn,
    ds_train: Dataset,
    *,
    seed: int,
    dataset_name: str,
    feature_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_keys: List[str] = list(_cfg.FEATURE_KEYS),
    run_hpo_trees: bool = False,
    run_hpo_torch: bool = False,
    force_retune: bool = False,
    n_iter_trees: int = _cfg.HPO_N_ITER_TREES,
    n_trials_torch: int = _cfg.HPO_N_TRIALS_TORCH,
    cv_folds: int = _cfg.HPO_CV_FOLDS,
    val_fraction: float = _cfg.HPO_VAL_FRACTION,
    train_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> BaseModel:
    """Fit one BloomBench model on *ds_train* and return it.

    Args:
        model_key:       Key in :data:`MODELS`.
        target_fn:       Callable extracting the ground-truth date from a sample.
        ds_train:        Training dataset.
        seed:            Random seed for model init, sampling, and HPO RNG.
        dataset_name:    Used as the cache key for HP best-params lookup.
        feature_stats:   Per-dataset ``{key: (mean, std)}`` for torch input
                         normalisation.  Ignored by tree models.  If ``None``
                         torch models fall back to
                         :meth:`BaseTorchModel.get_default_norm_params`
                         (only valid for OpenMeteo keys).
        feature_keys:    Climate feature keys passed via ``data_keys=...``.
        run_hpo_trees:   Run RandomizedSearchCV when no cached HP exists for
                         a tree model.
        run_hpo_torch:   Run a random search when no cached HP exists for a
                         torch model.
        force_retune:    Ignore any cached HP and re-run search.
        n_iter_trees:    Sklearn ``n_iter`` for tree HPO.
        n_trials_torch:  Trial budget for torch HPO.
        cv_folds:        ``GroupKFold(n_splits=...)`` for tree HPO.
        val_fraction:    Fraction of training years held out as val during
                         torch HPO.
        train_kwargs:    Override for torch training kwargs.  Defaults to
                         :func:`config.torch_train_kwargs`.
        verbose:         Print per-trial HPO progress when enabled.

    Returns:
        The fitted model.
    """
    if model_key not in MODELS:
        raise KeyError(f'Unknown BloomBench model key {model_key!r}. Choose from: {list(MODELS)}')

    if train_kwargs is None:
        train_kwargs = _cfg.torch_train_kwargs()

    _, dispatcher = MODELS[model_key]
    return dispatcher(
        target_fn, ds_train,
        seed=seed, dataset_name=dataset_name, model_key=model_key,
        feature_keys=list(feature_keys),
        feature_stats=feature_stats,
        run_hpo_trees=run_hpo_trees,
        run_hpo_torch=run_hpo_torch,
        force_retune=force_retune,
        n_iter_trees=n_iter_trees,
        n_trials_torch=n_trials_torch,
        cv_folds=cv_folds,
        val_fraction=val_fraction,
        train_kwargs=train_kwargs,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# HPO-only orchestrator (called by the CLI's `hpo` subcommand)
# ---------------------------------------------------------------------------

def run_hpo(
    datasets_dict,
    *,
    models: Optional[List[str]] = None,
    seed: int = 0,
    force_retune: bool = False,
    n_iter_trees: int = _cfg.HPO_N_ITER_TREES,
    n_trials_torch: int = _cfg.HPO_N_TRIALS_TORCH,
    cv_folds: int = _cfg.HPO_CV_FOLDS,
    val_fraction: float = _cfg.HPO_VAL_FRACTION,
    feature_keys: List[str] = list(_cfg.FEATURE_KEYS),
    train_kwargs: Optional[Dict[str, Any]] = None,
    compute_feature_stats: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run HPO for every (dataset, model) pair, caching best params.

    Models that have neither tree HPO nor torch HPO (``Mean``, ``Linear``)
    are skipped silently.  Iterates the full grid; failures are caught
    per-pair and logged into the returned summary rows.

    Args:
        datasets_dict:        Output of
                              :func:`pysephone.benchmarks.bloombench.datasets.load_bloombench_datasets`.
        models:               Subset of :data:`MODELS` keys to tune.  Defaults
                              to all tunable models.
        seed:                 RNG seed for HPO sampling and per-trial fits.
        force_retune:         Re-run HPO even if cached params exist.
        n_iter_trees:         Tree HPO budget.
        n_trials_torch:       Torch HPO budget.
        cv_folds:             Tree HPO CV folds.
        val_fraction:         Torch HPO val fraction.
        feature_keys:         Forwarded to :func:`fit_one`.
        train_kwargs:         Forwarded to :func:`fit_one`.
        compute_feature_stats: When ``True`` (default), compute per-dataset
                              feature statistics from the train fold and
                              pass them to torch models.
        verbose:              Print per-pair progress.

    Returns:
        List of summary rows (one per (dataset, model) attempted).  Each row
        is a dict with keys ``dataset``, ``model``, ``status``, ``seconds``,
        ``error``.
    """
    if train_kwargs is None:
        train_kwargs = _cfg.torch_train_kwargs()

    if models is None:
        models = [m for m in MODELS if is_tree_model(m) or is_torch_model(m)]
    else:
        models = [m for m in models if is_tree_model(m) or is_torch_model(m)]

    rows: List[Dict[str, Any]] = []
    for ds_name, (ds_train, _ds_test, target) in datasets_dict.items():
        target_fn = _make_target_fn(target)

        stats: Optional[Dict[str, Tuple[float, float]]] = None
        if compute_feature_stats and any(is_torch_model(m) for m in models):
            stats = ds_train.compute_feature_stats(verbose=False)

        for model_key in models:
            t0 = time.time()
            status = 'ok'
            error: Optional[str] = None
            try:
                fit_one(
                    model_key, target_fn, ds_train,
                    seed=seed, dataset_name=ds_name,
                    feature_stats=stats,
                    feature_keys=feature_keys,
                    run_hpo_trees=is_tree_model(model_key),
                    run_hpo_torch=is_torch_model(model_key),
                    force_retune=force_retune,
                    n_iter_trees=n_iter_trees,
                    n_trials_torch=n_trials_torch,
                    cv_folds=cv_folds,
                    val_fraction=val_fraction,
                    train_kwargs=train_kwargs,
                    verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-pair failures
                status = 'error'
                error = f'{type(exc).__name__}: {exc}'

            elapsed = round(time.time() - t0, 1)
            rows.append(dict(
                dataset=ds_name, model=model_key,
                status=status, seconds=elapsed, error=error,
            ))
            if verbose:
                tag = 'OK ' if status == 'ok' else 'ERR'
                msg = '' if error is None else f' — {error}'
                print(f'  [{ds_name:28s}] [{model_key:12s}] {tag}  {elapsed}s{msg}')

    return rows


# ---------------------------------------------------------------------------
# Target-fn helper
# ---------------------------------------------------------------------------

def _make_target_fn(target_key: str):
    """Return ``lambda sample: sample['observations'][target_key]``."""
    from pysephone.constants import KEY_OBSERVATIONS

    def target_fn(sample):
        return sample[KEY_OBSERVATIONS][target_key]

    return target_fn
