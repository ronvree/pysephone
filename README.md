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

Meteorological data is fetched from ERA5 reanalysis via the Open-Meteo archive and cached locally in HDF5 for fast repeated access. Other meteorological drivers can be integrated by implementing the `FeatureProvider` interface.

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

## Project Structure

```
.
├── src/pysephone/
│   ├── data/           # Data ingestion and sources (PEP725, GMU Cherry, ERA5)
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
