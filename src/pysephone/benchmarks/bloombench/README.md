# BloomBench

Multi-species fruit-tree phenology benchmark. Predicts the onset of flowering from a season-windowed daily temperature series, evaluated under a strict year-cutoff split so the test fold is strictly in the future of the train fold.

The benchmark is built on the rest of `pysephone`: [`Dataset`](../../dataset/dataset.py), [`AgEra5Features`](../../dataset/util/agera5.py), the model implementations in [`pysephone.models`](../../models/), and the statistical-comparison utilities in [`pysephone.evaluation.model_comparison`](../../evaluation/model_comparison.py). What lives in this folder is just the orchestration: a curated dataset list, fit/HPO dispatchers, a runner, a stats-comparison wrapper, and a thin CLI.

## What the benchmark contains

- **18 datasets** spanning three families — PEP725 (Europe), GMU Cherry (Japan/Switzerland/South Korea), USA-NPN (North America). Defined in [`config.DATASETS_REQUESTED`](config.py).
- **7 models** in canonical column order — `Mean`, `Linear`, `RandomForest`, `XGBoost`, `CNN1D`, `LSTM`, `Transformer`. Registered in [`fit.MODELS`](fit.py).
- **Climate driver** — daily mean 2 m temperature from AgERA5 (Copernicus CDS), accessed via [`AgEra5Features`](../../dataset/util/agera5.py).
- **Season window** — Oct 1 → 365 days, uniform across all datasets.
- **Split** — first 75 % of the years per dataset go to train, the rest to test.

## Prerequisites

1. Python ≥ 3.12 and a working install of `pysephone` (`pip install -e .` from the repo root).
2. Copernicus CDS credentials configured (so `AgEra5Features.download` can pull data). See [the CDS API guide](https://cds.climate.copernicus.eu/api-how-to) for the standard `~/.cdsapirc` setup.
3. *(Optional)* `pysephone[stats]` extra for the Friedman + Nemenyi + critical-difference outputs:

   ```bash
   pip install -e .[stats]
   ```

## End-to-end reproduction

A clean reproduction is four commands.

```bash
# 1. Download AgERA5 temperatures for every requested dataset.
#    One-off; cached in data/products/agera5/.
jupyter nbconvert --execute ../../notebooks/download_agera5.ipynb

# 2. Tune hyperparameters per (dataset, model).
#    Overnight on a single GPU; cached params land in
#    outputs/bloombench/hyperparams/agera5/.
python -m pysephone.benchmarks.bloombench hpo

# 3. Fit & evaluate every (seed, dataset, model) triple.
#    Results CSV + per-run prediction CSVs land in outputs/.
python -m pysephone.benchmarks.bloombench run --seeds 0 1 2

# 4. Friedman + Nemenyi + critical-difference plots.
python -m pysephone.benchmarks.bloombench compare --seeds 0 1 2
```

For an interactive flow with rendered tables, heatmaps, and CD plots, use the notebooks instead:

- [`notebooks/bloombench_extended_hpo.ipynb`](../../../../notebooks/bloombench_extended_hpo.ipynb) — one-time HPO.
- [`notebooks/bloombench_extended.ipynb`](../../../../notebooks/bloombench_extended.ipynb) — evaluation + statistical comparison.

## Python API

The same four steps, callable from Python or a custom script.

```python
from pysephone.benchmarks.bloombench import (
    load_bloombench_datasets, run_hpo, run_benchmark, run_comparison,
)

# Load + split + apply the size filter. Returns the kept dict and a per-dataset summary.
datasets, summary = load_bloombench_datasets(verbose=True)

# Tune hyperparameters. Skips (dataset, model) pairs that already have cached best_params.
run_hpo(datasets, seed=0)

# Fit & evaluate. Returns a tidy results DataFrame and writes a sidecar CSV.
results = run_benchmark(seeds=[0, 1, 2], datasets_dict=datasets)

# Friedman + Nemenyi over the cached evaluations.
report = run_comparison(seeds=[0, 1, 2], save_plots=True)
print(report.summary())
```

Each step uses on-disk caches keyed by `f'{run_prefix}_{dataset}_{model_key}_seed{seed}'` (see [`config.run_name`](config.py)), so re-running picks up where you left off. To force a refit, pass `force_retrain=True` (all models) or `force_retrain_models=['LSTM']` (specific models).

## Output layout

Everything lands under `<data_root>/outputs/` (where `<data_root>` is the repo root by default, overridable via the `PYSEPHONE_DATA_ROOT` env var):

```
outputs/
├── bloombench/
│   ├── hyperparams/agera5/         # best_params.json + _trials.json per (dataset, model)
│   └── runs/<run_prefix>/
│       ├── results.csv             # tidy (seed × dataset × model) metrics
│       └── datasets.json           # the kept-dataset list (for the compare step)
├── evaluations/<run_name>/         # per-run prediction CSVs + metadata.json
└── comparisons/<comparison_id>/    # scores.csv, nemenyi.csv, plots/
models/<run_name>/                  # pickled fitted models
```

## Configuration knobs

All defaults live in [`config.py`](config.py) as plain module-level constants — no dataclass, no yaml.

| Constant | Default | What it controls |
|---|---|---|
| `DATASETS_REQUESTED` | 18 pairs | The dataset list and per-dataset target observation key. |
| `MIN_DATASET_SAMPLES` | `100` | Drop datasets with fewer total samples after load. Mostly affects the small USA-NPN entries. |
| `FEATURE_KEYS` | `('Temperature_Air_2m_Mean_24h',)` | AgERA5 variable(s) consumed by the models. |
| `SEASON_START` / `SEASON_LENGTH` | `'10-01'` / `365` | Season window applied uniformly to every dataset. |
| `SPLIT_SIZE` | `0.75` | Train fraction in the year-cutoff split. |
| `HPO_N_ITER_TREES` | `20` | RandomizedSearchCV budget for `RandomForest` / `XGBoost`. |
| `HPO_N_TRIALS_TORCH` | `15` | Random-search budget for `CNN1D` / `LSTM` / `Transformer`. |
| `HPO_CV_FOLDS` | `5` | `GroupKFold(n_splits=...)` for tree HPO; groups are years. |
| `HPO_VAL_FRACTION` | `0.2` | Fraction of training years held out as the year-cutoff validation set during torch HPO. |
| `RUN_PREFIX` | `'bb_ext'` | Prefix for cache keys; change it to keep parallel runs from clashing. |
| `_TORCH_TRAIN_KWARGS_BASE` | 200 epochs / batch 32 / val every 10 / early stop @ patience 5 | Torch fit budget shared by HPO trials and final fits. |

Most of these are also surfaced as CLI flags. Run `python -m pysephone.benchmarks.bloombench <cmd> --help` for the full list.

## Implementation notes

A few things worth knowing if you intend to extend or troubleshoot:

- **AgERA5 vs OpenMeteo.** AgERA5 is the default climate provider. Temperatures are in Kelvin, and the variable key is `Temperature_Air_2m_Mean_24h` (not `temperature_2m_mean`). The torch models would normally KeyError on these because [`BaseTorchModel.get_default_norm_params`](../../models/torch_base.py) hardcodes the OpenMeteo names. The runner sidesteps this by computing per-dataset feature statistics from each train fold (via [`Dataset.compute_feature_stats`](../../dataset/dataset.py)) and injecting them as `feature_statistics=...` in `model_kwargs`. As a result, models see properly-normalised inputs regardless of the underlying unit or key — and the same flow works if you swap the provider later.
- **HP cache is provider-namespaced.** Files live under `outputs/bloombench/hyperparams/agera5/`. If you re-run HPO against a different provider, drop the JSON files into a sibling directory under the same parent — the `agera5` segment is wired into [`config.hp_cache_dir`](config.py).
- **HP cache writes are atomic.** Uses `tempfile.NamedTemporaryFile` + `os.replace`, so a Ctrl-C mid-write can't corrupt the JSON.
- **Model cache is defensive.** [`runner._try_load_model`](runner.py) catches `ModelException`, `EOFError`, `pickle.UnpicklingError`, `ModuleNotFoundError`, etc. — so a refactor of a model class doesn't poison every cached run; the affected cell just refits.
- **Torch device is resolved lazily.** [`config.torch_device`](config.py) is a function, not a module-level constant, so `CUDA_VISIBLE_DEVICES` set after import is honoured.
- **`scikit-posthocs`** is only needed for the `compare` step. Other commands run without it.

## Extending the benchmark

Three common extensions, in increasing order of effort.

1. **Add a dataset.** Register a builder in the appropriate file under [`pysephone.dataset.registry`](../../dataset/registry/) and append the `(name, target_key)` pair to [`config.DATASETS_REQUESTED`](config.py).
2. **Add a model.** Subclass [`BaseModel`](../../models/base.py) (or [`BaseTorchModel`](../../models/torch_base.py) for a torch model), add a fit dispatcher next to the others in [`fit.py`](fit.py), and register it in [`MODELS`](fit.py). If the new model is tree-based or torch-based, also add its key to `_TREE_MODEL_KEYS` / `_TORCH_MODEL_KEYS` so `run_hpo` picks it up.
3. **Add a feature.** Implement a [`FeatureProvider`](../../dataset/util/provider.py), update `FEATURE_KEYS`, and either pass the new provider into [`datasets.load_one`](datasets.py) or extend that function to attach multiple providers.

## Reference

If you use BloomBench in your work, please cite the corresponding paper:

> van Bree, R., Marcos, D., & Athanasiadis, I. N. (2026). BloomBench: A Multi-Species Benchmark for Evaluating the Generalization of Fruit Tree Phenology Models. *AAAI 2026 Workshop on AI in Agriculture (Agri AI)*.
