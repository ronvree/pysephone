"""
BloomBench — fruit-tree phenology benchmark.

Reproduce the canonical benchmark from Python::

    from pysephone.benchmarks.bloombench import (
        run_benchmark, run_comparison, run_hpo, load_bloombench_datasets,
    )

    # 1. (Optional) tune hyperparameters for tree + torch models.  Skip if
    #    you have a cached outputs/bloombench/hyperparams/ directory.
    datasets_dict, _ = load_bloombench_datasets()
    run_hpo(datasets_dict, models=['RandomForest', 'XGBoost', 'CNN1D', 'LSTM', 'Transformer'])

    # 2. Evaluate every (seed, dataset, model) triple.
    results_df = run_benchmark(seeds=[0, 1, 2], datasets_dict=datasets_dict)

    # 3. Friedman + Nemenyi + critical-difference plot across the results.
    report = run_comparison(seeds=[0, 1, 2])

The same operations are available as a CLI — see
:mod:`pysephone.benchmarks.bloombench.cli` and run::

    python -m pysephone.benchmarks.bloombench --help

Default climate provider is :class:`pysephone.dataset.util.agera5.AgEra5Features`
(Copernicus AgERA5).  Run ``notebooks/download_agera5.ipynb`` once to
populate the local HDF5 cache before the first ``run_benchmark`` call.
"""

from pysephone.benchmarks.bloombench import config
from pysephone.benchmarks.bloombench.compare import run_comparison
from pysephone.benchmarks.bloombench.datasets import (
    compute_train_feature_stats,
    load_bloombench_datasets,
    load_one,
    temporal_split,
)
from pysephone.benchmarks.bloombench.fit import (
    MODELS,
    fit_one,
    is_torch_model,
    is_tree_model,
    load_hp_cache,
    load_hp_trials,
    run_hpo,
    save_hp_cache,
)
from pysephone.benchmarks.bloombench.runner import run_benchmark

__all__ = [
    'MODELS',
    'compute_train_feature_stats',
    'config',
    'fit_one',
    'is_torch_model',
    'is_tree_model',
    'load_bloombench_datasets',
    'load_hp_cache',
    'load_hp_trials',
    'load_one',
    'run_benchmark',
    'run_comparison',
    'run_hpo',
    'save_hp_cache',
    'temporal_split',
]
