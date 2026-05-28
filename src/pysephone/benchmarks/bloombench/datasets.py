"""
Dataset loading and splitting for BloomBench.

Provides the orchestration around :class:`~pysephone.dataset.dataset.Dataset.load`:
build an :class:`AgEra5Features` provider with a shared ``Oct 1 + 365 day``
calendar, load every requested dataset, download the climate cache,
apply the 75/25 year-cutoff split, and filter NPN datasets with too few
samples.

Returns the kept set as a dict ``{name: (ds_train, ds_test, target_key)}``
plus a list of per-dataset status rows suitable for ``pd.DataFrame``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pysephone.dataset.dataset import Dataset
from pysephone.dataset.util.agera5 import AgEra5Features
from pysephone.dataset.util.calendar import Calendar

from pysephone.benchmarks.bloombench import config as _cfg


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DatasetTriple = Tuple[Dataset, Dataset, str]
DatasetDict = Dict[str, DatasetTriple]
SummaryRow = Dict[str, object]


# ---------------------------------------------------------------------------
# Single-dataset loader
# ---------------------------------------------------------------------------

def load_one(
    dataset_name: str,
    *,
    feature_keys: Sequence[str] = _cfg.FEATURE_KEYS,
    season_start: str = _cfg.SEASON_START,
    season_length: int = _cfg.SEASON_LENGTH,
    download_mode: Optional[str] = None,
    verbose: bool = False,
) -> Dataset:
    """Load a single named BloomBench dataset and warm its feature cache.

    Args:
        dataset_name:  Key in :data:`pysephone.dataset.registry.REGISTRY`.
        feature_keys:  AgERA5 variable names to attach.
        season_start:  Season window start as ``'MM-DD'``.
        season_length: Season window length in days.
        download_mode: Forwarded to ``Dataset.download_features`` —
                       ``None`` downloads missing entries, ``'skip'`` skips,
                       ``'forced'`` re-downloads.
        verbose:       Show progress bars while downloading / preloading.

    Returns:
        Loaded :class:`Dataset` with AgEra5 features cached in memory.
    """
    calendar = Calendar(default_start=season_start, default_length=season_length)
    provider = AgEra5Features(calendar=calendar, data_keys=list(feature_keys))
    ds = Dataset.load(dataset_name, calendar=calendar, feature_providers=[provider])
    ds.download_features(download_mode=download_mode, verbose=verbose)
    return ds


# ---------------------------------------------------------------------------
# Year-cutoff split
# ---------------------------------------------------------------------------

def temporal_split(
    ds: Dataset,
    train_fraction: float = _cfg.SPLIT_SIZE,
) -> Tuple[Dataset, Dataset]:
    """Split *ds* into (train, test) at the ``train_fraction`` year-cutoff.

    Years are sorted ascending, the first ``train_fraction * n_years`` go
    into the train fold, the remainder into the test fold.  This is the
    canonical BloomBench split — the test fold is strictly in the future of
    the train fold, exercising temporal generalisation.
    """
    years_sorted = sorted(set(ds.years))
    ix = int(len(years_sorted) * train_fraction)
    cutoff = years_sorted[ix] if ix < len(years_sorted) else years_sorted[-1] + 1
    years_trn = [y for y in years_sorted if y < cutoff]
    years_tst = [y for y in years_sorted if y >= cutoff]
    return ds.select_years(years_trn), ds.select_years(years_tst)


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------

def compute_train_feature_stats(
    ds_train: Dataset,
    *,
    verbose: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Return ``{key: (mean, std)}`` computed over *ds_train*'s features.

    This is the input-normalisation source for torch models — it sidesteps
    the OpenMeteo-keyed defaults baked into
    :meth:`BaseTorchModel.get_default_norm_params` so the benchmark can use
    AgERA5 (Kelvin temperatures, different keys) without changes to the
    model layer.
    """
    return ds_train.compute_feature_stats(verbose=verbose)


# ---------------------------------------------------------------------------
# Full benchmark load
# ---------------------------------------------------------------------------

def load_bloombench_datasets(
    datasets: Optional[Sequence[str]] = None,
    *,
    min_samples: int = _cfg.MIN_DATASET_SAMPLES,
    feature_keys: Sequence[str] = _cfg.FEATURE_KEYS,
    season_start: str = _cfg.SEASON_START,
    season_length: int = _cfg.SEASON_LENGTH,
    train_fraction: float = _cfg.SPLIT_SIZE,
    download_mode: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DatasetDict, List[SummaryRow]]:
    """Load every requested BloomBench dataset, apply the size filter and split.

    Args:
        datasets:       Optional subset of dataset names to load.  If
                        ``None``, the full :data:`config.DATASETS_REQUESTED`
                        set is used.
        min_samples:    Datasets with fewer total samples are skipped
                        (``<{min_samples}``).  Applies uniformly to every
                        family; in practice only the smaller USA-NPN
                        datasets are affected at the default threshold.
        feature_keys:   AgERA5 variable names; forwarded to :func:`load_one`.
        season_start:   Season window start; forwarded to :func:`load_one`.
        season_length:  Season window length; forwarded to :func:`load_one`.
        train_fraction: Year-cutoff train fraction; forwarded to
                        :func:`temporal_split`.
        download_mode:  Forwarded to :func:`load_one` /
                        :meth:`Dataset.download_features`.
        verbose:        Print a per-dataset load summary.

    Returns:
        ``(datasets_dict, summary_rows)`` where ``datasets_dict`` maps
        ``name -> (ds_train, ds_test, target_key)`` for every dataset that
        successfully loaded, passed the size filter, and split into a
        non-empty train + test fold.  ``summary_rows`` is a list of dicts
        (one per requested dataset) suitable for ``pd.DataFrame``.
    """
    requested: List[Tuple[str, str]]
    if datasets is None:
        requested = list(_cfg.DATASETS_REQUESTED)
    else:
        requested = [
            (name, target) for name, target in _cfg.DATASETS_REQUESTED if name in set(datasets)
        ]

    out: DatasetDict = {}
    rows: List[SummaryRow] = []

    for ds_name, target in requested:
        try:
            ds_full = load_one(
                ds_name,
                feature_keys=feature_keys,
                season_start=season_start,
                season_length=season_length,
                download_mode=download_mode,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface load errors per row
            rows.append(dict(dataset=ds_name, n=0, status=f'LOAD FAIL: {exc}'))
            if verbose:
                print(f'  [{ds_name:28s}] LOAD FAILED: {type(exc).__name__}: {exc}')
            continue

        n_total = len(ds_full)
        if n_total < min_samples:
            rows.append(dict(dataset=ds_name, n=n_total,
                             status=f'skipped (<{min_samples})'))
            if verbose:
                print(f'  [{ds_name:28s}] SKIPPED ({n_total} samples < {min_samples})')
            continue

        ds_trn, ds_tst = temporal_split(ds_full, train_fraction=train_fraction)
        if len(ds_trn) == 0 or len(ds_tst) == 0:
            rows.append(dict(dataset=ds_name, n=n_total, status='empty split'))
            if verbose:
                print(f'  [{ds_name:28s}] SKIPPED (empty train or test after split)')
            continue

        out[ds_name] = (ds_trn, ds_tst, target)
        rows.append(dict(
            dataset=ds_name, n=n_total, status='kept',
            n_train=len(ds_trn), n_test=len(ds_tst),
            years=f'{min(ds_full.years)}-{max(ds_full.years)}',
        ))
        if verbose:
            print(
                f'  [{ds_name:28s}] total={n_total:5d}  '
                f'train={len(ds_trn):5d}  test={len(ds_tst):5d}  '
                f'years {min(ds_full.years)}-{max(ds_full.years)}'
            )

    if verbose:
        print(f'\nKept {len(out)} / {len(requested)} datasets.')

    return out, rows
