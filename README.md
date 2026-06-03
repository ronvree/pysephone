<div align="center">

<img src="assets/logo.png" alt="pysephone logo" width="280">

# pysephone

[![PyPI](https://img.shields.io/pypi/v/pysephone)](https://pypi.org/project/pysephone/)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

</div>
<br>

## Description

**pysephone** is a Python package for developing and benchmarking crop phenology models — models that predict the timing of key developmental events in plants, such as flowering, leaf-out, or harvest maturity. Accurate phenology predictions are essential for agricultural planning, yield forecasting, and understanding how ecosystems respond to climate variability and long-term change. As growing seasons shift under climate change, the ability to reliably model phenological timing across species and regions becomes increasingly important for both science and policy.

pysephone provides a standardised pipeline that connects observational phenology databases with meteorological drivers, and a suite of models ranging from classical process-based approaches to deep learning, all sharing a common interface.

The package is designed to make it straightforward to:
- load and preprocess phenological observation data from multiple sources,
- pair observations with season-windowed meteorological time series (ERA5 reanalysis supported out of the box; other drivers can be integrated),
- define phenology datasets for standardised intercomparison of models,
- fit and evaluate a variety of models, and
- systematically compare model behaviour across species, regions, and climate conditions.

<br>

## Installation

```bash
git clone https://github.com/ronvree/pysephone.git
cd pysephone
pip install -e .
```

Requires Python ≥ 3.12.

The base install is intentionally lightweight (process-based + scikit-learn models, datasets, evaluation). Heavier and source-specific dependencies are opt-in via extras:

| Extra | Adds | Needed for |
|---|---|---|
| `deep` | PyTorch | LSTM / GRU / CNN / Transformer / hybrid / Beta-GDD / BSpline-GDD models |
| `boost` | XGBoost | `XGBoostModel` |
| `agera5` | cdsapi, xarray, netCDF4 | Downloading AgERA5 from Copernicus CDS |
| `openmeteo` | openmeteo-requests | Downloading Open-Meteo ERA5 |
| `geo` | geopandas, shapely, rasterio | Map visualizations, WorldClim rasters |
| `earthengine` | earthengine-api | Fetching AlphaEarth embeddings |
| `stats` | scikit-posthocs, autorank | Friedman/Nemenyi comparison + critical-difference plots |
| `all` | everything above | Convenience meta-extra (every model + data source) |

Install one or more with e.g. `pip install "pysephone[deep]"` or `pip install "pysephone[deep,agera5]"`. Accessing a model whose extra isn't installed raises a clear error telling you which extra to add.

Reproducing **BloomBench** specifically needs `pip install "pysephone[deep,boost,agera5,stats]"` — its models are CNN/LSTM/Transformer (`deep`), XGBoost (`boost`), RandomForest/Mean/Linear (base); climate features come from AgERA5 (`agera5`); and the `compare` step's Nemenyi/critical-difference plots need `stats`. It does **not** use AlphaEarth/Earth Engine or Open-Meteo.

<br>

## Data Sources

| Source | Description |
|---|---|
| **PEP725** | Pan-European Phenology Database — multi-species observations across Europe |
| **GMU Cherry Blossom** | Cherry blossom bloom dates from Japan, Switzerland, and South Korea |
| **USA-NPN** | USA National Phenology Network — deciduous fruit-tree observations |
| **AgERA5** | Daily agrometeorological indicators from Copernicus CDS (downscaled temperature/radiation, Penman–Monteith inputs, etc.) |
| **Open-Meteo ERA5** | ERA5 reanalysis via the Open-Meteo archive |

Meteorological data is cached locally in HDF5 for fast repeated access. Additional providers can be integrated by implementing the `FeatureProvider` interface.

Some reference datasets (e.g. the cherry-blossom bloom records) are bundled with the package. These third-party datasets retain their original licenses and attribution requirements — see [DATA_SOURCES.md](DATA_SOURCES.md). Note that `liestal.csv` is non-commercial-use only and `kyoto.csv` is provided for academic use with required citations.

<br>

## Authentication & configuration

Some data sources reach external APIs that require **your own** account/project — pysephone ships no credentials and no default project. Set these up once before running the download steps:

**Copernicus CDS (AgERA5).** Authentication uses the [`cdsapi`](https://cds.climate.copernicus.eu/how-to-api) convention — pysephone handles no keys itself. Provide your credentials via either:
- a `~/.cdsapirc` file, or
- the `CDSAPI_URL` and `CDSAPI_KEY` environment variables.

**Google Earth Engine (AlphaEarth embeddings).** Requires a Google Cloud project with the Earth Engine API enabled:
1. Authenticate once: `python -c "import ee; ee.Authenticate()"`.
2. Tell pysephone which project to use, in priority order:
   - pass `ee_project="your-gcp-project"` to `fetch_alphaearth_embeddings_batched(...)`, or
   - set the `PYSEPHONE_EE_PROJECT` environment variable (Earth Engine's native `EARTHENGINE_PROJECT` is also honored), or
   - leave it unset to let Earth Engine resolve its own default project.

**Data location.** All caches, downloaded data, and outputs are written under a single data root. By default this is an OS-native per-user directory (`%LOCALAPPDATA%\pysephone` on Windows, `~/.local/share/pysephone` on Linux/macOS). Override it with the `PYSEPHONE_DATA_ROOT` environment variable — e.g. set `PYSEPHONE_DATA_ROOT=<repo>` to keep data inside a source checkout during development.

<br>

## Models

| Category | Models |
|---|---|
| Baseline | Mean |
| Process-based | GDD, Utah+GDD, ChillingDays+GDD, Dynamic+GDD |
| Machine learning | Random Forest |
| Deep learning | LSTM, Hybrid (TTCNN chilling + GDD forcing) |

All models share a common `fit` / `predict` interface, making it easy to add new models or swap them in evaluation pipelines.

<br>

## Pipeline Overview

```
Data source (PEP725 / GMU Cherry)
    ↓  preprocessing  (outlier removal, grid aggregation)
Observations  (indexed by source, location, year, species, obs type)
    ↓  paired with Calendar + meteorological feature provider
Dataset  (yields season-windowed feature arrays per sample)
    ↓
Model.fit(target_fn, dataset)  →  Model.predict(sample)
    ↓
SingleTargetRegression.run(...)  →  metrics, error DataFrames, plots
```

The `Calendar` defines the season window (start date + length) for each entry. Feature providers retrieve the corresponding meteorological time series for each sample.

<br>

## Reproducing BloomBench

[BloomBench](https://github.com/WUR-AI/BloomBench) is a multi-species benchmark for evaluating ML phenology models on fruit-tree flowering. The benchmark is shipped as a first-class library module: [`pysephone.benchmarks.bloombench`](src/pysephone/benchmarks/bloombench/).

It exposes both a Python API and a thin CLI. Reproducing the benchmark needs the deep-learning, boosting, AgERA5, and stats extras (it does **not** use Earth Engine or Open-Meteo):

```bash
pip install "pysephone[deep,boost,agera5,stats]"
```

```bash
# 1. Populate the AgERA5 cache once (Copernicus CDS credentials required).
jupyter nbconvert --execute notebooks/download_agera5.ipynb

# 2. Tune hyperparameters per (dataset, model) — overnight run.
python -m pysephone.benchmarks.bloombench hpo

# 3. Fit & evaluate every (seed, dataset, model) triple.
python -m pysephone.benchmarks.bloombench run --seeds 0 1 2

# 4. Friedman + Nemenyi + critical-difference plots.
python -m pysephone.benchmarks.bloombench compare --seeds 0 1 2
```

The same flow as Python:

```python
from pysephone.benchmarks.bloombench import (
    load_bloombench_datasets, run_benchmark, run_comparison, run_hpo,
)

datasets, _ = load_bloombench_datasets()
run_hpo(datasets)                                      # one-time HPO
results = run_benchmark(seeds=[0, 1, 2], datasets_dict=datasets)
report = run_comparison(seeds=[0, 1, 2])
```

For the interactive flow with tables / heatmaps / critical-difference plots, see [`notebooks/bloombench_extended_hpo.ipynb`](notebooks/bloombench_extended_hpo.ipynb) (one-time HPO) and [`notebooks/bloombench_extended.ipynb`](notebooks/bloombench_extended.ipynb) (replication).

<br>

## Project Structure

```
.
├── src/pysephone/
│   ├── benchmarks/     # End-to-end benchmark suites (BloomBench, …)
│   ├── data/           # Data ingestion and sources (PEP725, GMU Cherry, USA-NPN, AgERA5)
│   ├── dataset/        # Observations, Dataset, Calendar, feature providers, registry
│   ├── evaluation/     # Evaluation logic and regression metrics
│   ├── models/         # Model implementations (CF, RF, LSTM, Hybrid, …)
│   ├── utils/          # Shared utilities
│   └── visualize/      # Visualisation helpers
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/            # Standalone scripts
├── tests/              # Test suite
└── data/               # Raw and processed data (git-ignored)
```

<br>

## Dependencies

Base install: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `nlopt`, `tables`, `h5py`, `requests`, `requests-cache`, `retry-requests`, `tqdm`, `unidecode`, `platformdirs`.

Heavier and source-specific dependencies (`torch`, `xgboost`, `cdsapi`, `openmeteo-requests`, `geopandas`/`shapely`/`rasterio`, `earthengine-api`, …) are opt-in via the [extras](#installation) above.
