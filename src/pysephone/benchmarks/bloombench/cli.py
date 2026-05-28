"""
Thin CLI over the BloomBench Python API.

Subcommands:

* ``run``     — fit & evaluate every (seed, dataset, model) triple.
* ``hpo``     — run hyperparameter search and cache best params.
* ``compare`` — Friedman + Nemenyi + ranks across cached evaluations.

Invocation::

    python -m pysephone.benchmarks.bloombench run --seeds 0 1 2
    python -m pysephone.benchmarks.bloombench hpo --models RandomForest LSTM
    python -m pysephone.benchmarks.bloombench compare --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from pysephone.benchmarks.bloombench import config as _cfg
from pysephone.benchmarks.bloombench import compare as _compare
from pysephone.benchmarks.bloombench import datasets as _datasets
from pysephone.benchmarks.bloombench import fit as _fit
from pysephone.benchmarks.bloombench import runner as _runner


_ALL_DATASETS = [name for name, _ in _cfg.DATASETS_REQUESTED]
_ALL_MODELS = list(_fit.MODELS)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    df = _runner.run_benchmark(
        seeds=args.seeds,
        datasets=args.datasets,
        models=args.models,
        run_prefix=args.run_prefix,
        force_retrain=args.force_retrain,
        force_retrain_models=args.force_retrain_models or (),
        verbose=not args.quiet,
    )
    print(f'\n{len(df)} rows written to outputs/bloombench/runs/{args.run_prefix}/results.csv')
    return 0


def _cmd_hpo(args: argparse.Namespace) -> int:
    datasets_dict, _summary = _datasets.load_bloombench_datasets(
        datasets=args.datasets,
        verbose=not args.quiet,
    )
    rows = _fit.run_hpo(
        datasets_dict,
        models=args.models,
        seed=args.seed,
        force_retune=args.force_retune,
        n_iter_trees=args.n_iter_trees,
        n_trials_torch=args.n_trials_torch,
        verbose=not args.quiet,
    )
    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    print(f'\nHPO finished: {n_ok}/{len(rows)} pairs successful.')
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    report = _compare.run_comparison(
        seeds=args.seeds,
        datasets=args.datasets,
        models=args.models,
        run_prefix=args.run_prefix,
        metric=args.metric,
        split=args.split,
        save_plots=args.save_plots,
        comparison_id=args.comparison_id,
        verbose=not args.quiet,
    )
    print(report.summary())
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m pysephone.benchmarks.bloombench',
        description='BloomBench — fruit-tree phenology benchmark.',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # run --------------------------------------------------------------------
    p_run = sub.add_parser(
        'run',
        help='Fit & evaluate every (seed, dataset, model) triple.',
        description=(
            'For each (seed, dataset, model) triple: load the cached model if '
            'present; otherwise fit & save; then write an evaluation CSV per '
            'run and a tidy results.csv sidecar.'
        ),
    )
    p_run.add_argument('--seeds', type=int, nargs='+', default=[0])
    p_run.add_argument('--datasets', nargs='+', choices=_ALL_DATASETS,
                       default=None, metavar='NAME')
    p_run.add_argument('--models', nargs='+', choices=_ALL_MODELS,
                       default=None, metavar='KEY')
    p_run.add_argument('--run-prefix', default=_cfg.RUN_PREFIX)
    p_run.add_argument('--force-retrain', action='store_true',
                       help='Ignore the model cache for every model.')
    p_run.add_argument('--force-retrain-models', nargs='+', choices=_ALL_MODELS,
                       default=None, metavar='KEY',
                       help='Ignore the cache only for the listed models.')
    p_run.add_argument('--quiet', action='store_true')
    p_run.set_defaults(func=_cmd_run)

    # hpo --------------------------------------------------------------------
    p_hpo = sub.add_parser(
        'hpo',
        help='Hyperparameter search; cache best params per (dataset, model).',
        description=(
            'Run RandomizedSearchCV for tree models and a year-cutoff random '
            'search for torch models.  Best params are cached under '
            'outputs/bloombench/hyperparams/agera5/.'
        ),
    )
    p_hpo.add_argument('--seed', type=int, default=0)
    p_hpo.add_argument('--datasets', nargs='+', choices=_ALL_DATASETS,
                       default=None, metavar='NAME')
    p_hpo.add_argument('--models', nargs='+',
                       choices=[m for m in _ALL_MODELS
                                if _fit.is_tree_model(m) or _fit.is_torch_model(m)],
                       default=None, metavar='KEY',
                       help='Defaults to all tunable models.')
    p_hpo.add_argument('--n-iter-trees', type=int, default=_cfg.HPO_N_ITER_TREES)
    p_hpo.add_argument('--n-trials-torch', type=int, default=_cfg.HPO_N_TRIALS_TORCH)
    p_hpo.add_argument('--force-retune', action='store_true',
                       help='Re-run HPO even when cached params exist.')
    p_hpo.add_argument('--quiet', action='store_true')
    p_hpo.set_defaults(func=_cmd_hpo)

    # compare ---------------------------------------------------------------
    p_cmp = sub.add_parser(
        'compare',
        help='Friedman + Nemenyi + ranks across cached evaluations.',
        description=(
            'Aggregates cached SingleTargetRegression runs across seeds, '
            'computes the Friedman omnibus + Nemenyi post-hoc test, and saves '
            'the comparison report (scores / nemenyi / plots) under '
            'outputs/comparisons/.'
        ),
    )
    p_cmp.add_argument('--seeds', type=int, nargs='+', default=[0])
    p_cmp.add_argument('--datasets', nargs='+', choices=_ALL_DATASETS,
                       default=None, metavar='NAME')
    p_cmp.add_argument('--models', nargs='+', choices=_ALL_MODELS,
                       default=None, metavar='KEY')
    p_cmp.add_argument('--run-prefix', default=_cfg.RUN_PREFIX)
    p_cmp.add_argument('--metric', default='mae')
    p_cmp.add_argument('--split', default='test')
    p_cmp.add_argument('--comparison-id', default=None)
    p_cmp.add_argument('--save-plots', dest='save_plots', action='store_true', default=True)
    p_cmp.add_argument('--no-save-plots', dest='save_plots', action='store_false')
    p_cmp.add_argument('--quiet', action='store_true')
    p_cmp.set_defaults(func=_cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
