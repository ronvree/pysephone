"""
Main BloomBench evaluation loop.

Iterates ``(seed, dataset, model)`` triples.  For each cell:

1. Try to load a cached fitted model under
   :func:`config.run_name(dataset, model_key, seed)`.
2. If the cache is empty or corrupt, fit via :func:`fit.fit_one`, then
   :meth:`model.save(run_name)`.
3. Run :meth:`SingleTargetRegression.run` on (train, test), persist the
   evaluation under the same ``run_name``, and accumulate a metrics row.

Returns the tidy results :class:`pandas.DataFrame` and writes a sidecar
``results.csv`` + ``datasets.json`` (the post-NPN-filter dataset list)
under ``<data_root>/outputs/bloombench/runs/<run_prefix>/``.  The compare
step reads ``datasets.json`` so it stays in sync with the actual runs.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pysephone.dataset.dataset import Dataset
from pysephone.evaluation.regression import SingleTargetRegression
from pysephone.models.base import BaseModel, ModelException

from pysephone.benchmarks.bloombench import config as _cfg
from pysephone.benchmarks.bloombench import datasets as _datasets
from pysephone.benchmarks.bloombench import fit as _fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOAD_EXC: Tuple[type, ...] = (
    ModelException,
    ModuleNotFoundError,
    EOFError,
    pickle.UnpicklingError,
    FileNotFoundError,
    AttributeError,
)


def _try_load_model(model_cls, run_name: str, root: Optional[Path]) -> Optional[BaseModel]:
    """Try to load a cached fitted model; return ``None`` on any expected miss.

    Catches the (broader-than-strictly-necessary) set of exceptions the
    original notebook used, so a refactor of a model class doesn't poison
    every cached run — we just fall through to a fresh fit.
    """
    try:
        model, _ = model_cls.load(run_name, root=root)
        return model
    except _LOAD_EXC:
        return None


def _results_dir(run_prefix: str, root: Optional[Path]) -> Path:
    return _cfg.runs_dir(root) / run_prefix


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_benchmark(
    seeds: Sequence[int] = (0,),
    *,
    datasets: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    datasets_dict: Optional[_datasets.DatasetDict] = None,
    run_prefix: str = _cfg.RUN_PREFIX,
    force_retrain: bool = False,
    force_retrain_models: Sequence[str] = (),
    feature_keys: Sequence[str] = _cfg.FEATURE_KEYS,
    compute_feature_stats: bool = True,
    train_kwargs: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the BloomBench evaluation loop and return a tidy results DataFrame.

    Args:
        seeds:                  Random seeds to fit each model under.
        datasets:               Subset of dataset names.  ``None`` uses the
                                full :data:`config.DATASETS_REQUESTED`.
                                Ignored when *datasets_dict* is provided.
        models:                 Subset of :data:`fit.MODELS` keys to run.
                                ``None`` runs every entry.
        datasets_dict:          Pre-loaded dict from
                                :func:`datasets.load_bloombench_datasets`.
                                If ``None``, loaded here.
        run_prefix:             Prefix used to build the run-name for each
                                ``(dataset, model, seed)`` triple.
        force_retrain:          Ignore the model cache for every model and
                                refit from scratch.
        force_retrain_models:   Ignore the cache only for the listed model
                                keys (overrides cache hits for those).
        feature_keys:           Forwarded to :func:`fit.fit_one`.
        compute_feature_stats:  When ``True`` (default), compute per-dataset
                                feature stats from the train fold and pass
                                them to torch models for input normalisation.
        train_kwargs:           Forwarded to :func:`fit.fit_one`.
        root:                   Override data root (used by tests).
        verbose:                Print per-cell progress.

    Returns:
        Tidy :class:`pandas.DataFrame` with one row per ``(seed, dataset,
        model)`` cell.  Columns: ``seed, dataset, model, source, status,
        n_train, n_test, mae_train, mae_test, rmse_test, r2_test, seconds,
        error``.

        Also writes the same DataFrame to
        ``<root>/outputs/bloombench/runs/<run_prefix>/results.csv`` and the
        kept-dataset list to ``datasets.json`` in the same directory.
    """
    if datasets_dict is None:
        datasets_dict, _summary = _datasets.load_bloombench_datasets(
            datasets=datasets,
            feature_keys=feature_keys,
            verbose=verbose,
        )

    if models is None:
        models = list(_fit.MODELS)

    force_set = set(force_retrain_models)

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for ds_name, (ds_train, ds_test, target) in datasets_dict.items():
            target_fn = _fit._make_target_fn(target)

            stats = None
            if compute_feature_stats and any(_fit.is_torch_model(m) for m in models):
                stats = ds_train.compute_feature_stats(verbose=False)

            for model_key in models:
                if model_key not in _fit.MODELS:
                    if verbose:
                        print(f'  skipping unknown model {model_key!r}')
                    continue

                model_cls, _ = _fit.MODELS[model_key]
                rn = _cfg.run_name(ds_name, model_key, seed, run_prefix=run_prefix)
                t0 = time.time()
                source = 'cache'
                status = 'ok'
                error: Optional[str] = None

                skip_cache = force_retrain or model_key in force_set
                model = None if skip_cache else _try_load_model(model_cls, rn, root=root)

                if model is None:
                    source = 'fit'
                    try:
                        model = _fit.fit_one(
                            model_key, target_fn, ds_train,
                            seed=seed, dataset_name=ds_name,
                            feature_stats=stats,
                            feature_keys=list(feature_keys),
                            run_hpo_trees=False,
                            run_hpo_torch=False,
                            force_retune=False,
                            train_kwargs=train_kwargs,
                            verbose=verbose,
                        )
                        model.save(rn, root=root)
                    except Exception as exc:  # noqa: BLE001 - isolate per-cell failures
                        status = 'error'
                        error = f'{type(exc).__name__}: {exc}'

                metrics = {'mae': None, 'rmse': None, 'r2': None}
                metrics_train = {'mae': None}
                n_train = len(ds_train)
                n_test = len(ds_test)

                if status == 'ok':
                    try:
                        evaluation = SingleTargetRegression.run(
                            model=model,
                            dataset_train=ds_train,
                            dataset_test=ds_test,
                            target_fn=target_fn,
                            run_name=rn,
                        )
                        evaluation.save(root=root)
                        m = evaluation.compute_metrics()
                        metrics_train = m.get('train', metrics_train)
                        metrics = m.get('test', metrics)
                    except Exception as exc:  # noqa: BLE001 - eval-failure isolation
                        status = 'error'
                        error = f'{type(exc).__name__}: {exc}'

                elapsed = round(time.time() - t0, 1)
                rows.append(dict(
                    seed=int(seed),
                    dataset=ds_name,
                    model=model_key,
                    source=source,
                    status=status,
                    n_train=n_train,
                    n_test=n_test,
                    mae_train=metrics_train.get('mae'),
                    mae_test=metrics.get('mae'),
                    rmse_test=metrics.get('rmse'),
                    r2_test=metrics.get('r2'),
                    seconds=elapsed,
                    error=error,
                ))

                if verbose:
                    mae = metrics.get('mae')
                    mae_txt = 'n/a' if mae is None else f'{mae:6.3f}'
                    tag = 'OK ' if status == 'ok' else 'ERR'
                    print(
                        f'  seed={seed} [{ds_name:28s}] [{model_key:12s}] '
                        f'{tag} src={source:5s} MAE_test={mae_txt}  {elapsed}s'
                        + ('' if error is None else f'  — {error}')
                    )

    df = pd.DataFrame(rows)

    out_dir = _results_dir(run_prefix, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'results.csv', index=False)
    with open(out_dir / 'datasets.json', 'w') as f:
        json.dump(list(datasets_dict.keys()), f, indent=2)

    return df
