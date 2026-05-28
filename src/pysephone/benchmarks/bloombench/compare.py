"""
Statistical comparison of BloomBench runs.

Thin wrapper around :func:`pysephone.evaluation.model_comparison.compare_models`
that:

* discovers the (dataset, model, seed) runs produced by :func:`runner.run_benchmark`
  by re-building the run-names from :func:`config.run_name`,
* loads each :class:`SingleTargetRegression` from disk and wraps it as an
  :class:`EvaluationRun`,
* invokes ``compare_models`` with BloomBench defaults (``metric='mae'``,
  ``split='test'``, ascending order), and
* persists the resulting :class:`ComparisonReport` under
  ``<root>/outputs/comparisons/<comparison_id>/``.

If no explicit dataset list is given, the kept-dataset list written by
``runner.run_benchmark`` (``runs/<run_prefix>/datasets.json``) is consumed
so the comparison stays in lockstep with what was actually evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from pysephone.evaluation.model_comparison import (
    ComparisonReport,
    EvaluationRun,
    compare_models,
)
from pysephone.evaluation.regression import SingleTargetRegression

from pysephone.benchmarks.bloombench import config as _cfg
from pysephone.benchmarks.bloombench import fit as _fit


def _resolve_datasets(
    datasets: Optional[Sequence[str]],
    run_prefix: str,
    root: Optional[Path],
) -> List[str]:
    if datasets is not None:
        return list(datasets)
    sidecar = _cfg.runs_dir(root) / run_prefix / 'datasets.json'
    if sidecar.exists():
        with open(sidecar) as f:
            kept = json.load(f)
        return list(kept)
    return [name for name, _ in _cfg.DATASETS_REQUESTED]


def run_comparison(
    seeds: Sequence[int] = (0,),
    *,
    datasets: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    run_prefix: str = _cfg.RUN_PREFIX,
    metric: str = 'mae',
    split: str = 'test',
    alpha: float = 0.05,
    order: str = 'ascending',
    missing_policy: str = 'skip_seed',
    comparison_id: Optional[str] = None,
    save: bool = True,
    save_plots: bool = True,
    root: Optional[Path] = None,
    verbose: bool = True,
) -> ComparisonReport:
    """Load existing BloomBench evaluations and run Friedman + Nemenyi.

    Args:
        seeds:          Seeds that were used by :func:`run_benchmark`.
        datasets:       Dataset subset.  ``None`` reuses the kept list
                        produced by the most recent ``run_benchmark`` (via
                        the ``datasets.json`` sidecar) and falls back to
                        :data:`config.DATASETS_REQUESTED`.
        models:         Model subset.  ``None`` runs across all models in
                        :data:`fit.MODELS`.
        run_prefix:     Same prefix used by :func:`run_benchmark`.
        metric:         Forwarded to ``compare_models``.
        split:          Forwarded to ``compare_models``.
        alpha:          Significance threshold for Nemenyi.
        order:          ``'ascending'`` (lower is better) or
                        ``'descending'``.
        missing_policy: How ``compare_models`` handles missing (seed,
                        dataset, model) combinations.
        comparison_id:  Identifier for the saved report directory.
                        Defaults to ``'{run_prefix}_{metric}_{split}'``.
        save:           Persist the report to disk.
        save_plots:     When *save* is True, render plot PNGs alongside.
        root:           Override data root (used by tests).
        verbose:        Print per-run load status.

    Returns:
        The :class:`ComparisonReport`.
    """
    dataset_keys = _resolve_datasets(datasets, run_prefix, root)
    model_keys = list(models) if models is not None else list(_fit.MODELS)

    runs: List[EvaluationRun] = []
    n_missing = 0
    for seed in seeds:
        for ds_name in dataset_keys:
            for model_key in model_keys:
                rn = _cfg.run_name(ds_name, model_key, seed, run_prefix=run_prefix)
                try:
                    sr = SingleTargetRegression.load(rn, root=root)
                except Exception as exc:  # noqa: BLE001 - skip missing eval
                    n_missing += 1
                    if verbose:
                        print(f'  miss [{ds_name:28s}] [{model_key:12s}] seed={seed}  {type(exc).__name__}')
                    continue
                runs.append(EvaluationRun(
                    eval_result=sr,
                    model_key=model_key,
                    dataset_key=ds_name,
                    seed=int(seed),
                ))

    if verbose:
        print(f'  loaded {len(runs)} runs ({n_missing} missing)')

    report = compare_models(
        runs,
        model_keys=model_keys,
        dataset_keys=dataset_keys,
        seeds=list(seeds),
        metric=metric,
        split=split,
        alpha=alpha,
        order=order,
        missing_policy=missing_policy,
    )

    if save:
        cid = comparison_id or f'{run_prefix}_{metric}_{split}'
        report.save(cid, save_plots=save_plots, root=root)

    return report
