<div align="center">

# pysephone

![python](https://img.shields.io/badge/python-3.14%2B-blue)
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

Requires Python ≥ 3.14.

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

It exposes both a Python API and a thin CLI:

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

## Notebooks

| Notebook | Description |
|---|---|
| `cherry_blossom_cf_models.ipynb` | Process-based model evaluation on GMU Cherry datasets |
| `cf_models_pep725_fruit_trees.ipynb` | CF model evaluation across PEP725 fruit tree species |
| `unusual_year_model_eval.ipynb` | Model comparison on climatologically unusual vs normal years |
| `unusual_seasons_*.ipynb` | Exploration of unusual seasons in GMU / PEP725 data |
| `dataset_adequacy_*.ipynb` | Sample sufficiency analysis per dataset |
| `lstm_cherry_exploration.ipynb` | LSTM model exploration on cherry blossom data |
| `model_exploration.ipynb` | General model exploration notebook |
| `pvtt_winter_wheat.ipynb` | PVTT model for winter wheat phenology |

<br>

## Dependencies

`pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, `geopandas`, `shapely`, `nlopt`, `tables`, `openmeteo-requests`, `requests-cache`, `retry-requests`, `tqdm`, `requests`, `unidecode`
