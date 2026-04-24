"""
Build (or rebuild) an OpenMeteo FeatureCache for a registered dataset.

Usage
-----
    python scripts/build_feature_cache.py PEP725_Apple
    python scripts/build_feature_cache.py PEP725_Apple --keys temperature_2m_mean daylight_duration
    python scripts/build_feature_cache.py GMU_Cherry_Japan_Y --force
    python scripts/build_feature_cache.py PEP725_Apple --step daily --keys temperature_2m_mean

Arguments
---------
dataset_key     Registry key of the dataset to cache (required).
--keys          One or more OpenMeteo data keys.
                Default: temperature_2m_mean
--step          Temporal resolution label stored in the cache filename.
                Informational only — must match what OpenMeteoFeatures uses.
                Default: daily
--force         Overwrite an existing cache file.
--root          Override the data root (defaults to PYSEPHONE_DATA_ROOT env var
                or the repository root).
--quiet         Suppress progress bars.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pysephone.dataset.dataset import Dataset
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.openmeteo import OpenMeteoFeatures
from pysephone.dataset.util.feature_cache import FeatureCache


DEFAULT_KEYS = ['temperature_2m_mean']
DEFAULT_STEP = 'daily'


def build_cache(
    dataset_key: str,
    data_keys: list[str],
    step: str,
    force: bool,
    root: Path | None,
    verbose: bool,
) -> None:
    path = FeatureCache.default_path(dataset_key, data_keys, step=step, root=root)

    if FeatureCache.exists(path) and not force:
        print(f'Cache already exists: {path}')
        print('Pass --force to overwrite.')
        return

    print(f'Dataset : {dataset_key}')
    print(f'Keys    : {data_keys}')
    print(f'Step    : {step}')
    print(f'Output  : {path}')
    print()

    cal   = Calendar()
    feats = OpenMeteoFeatures(calendar=cal, data_keys=data_keys)

    # Load observations (no feature provider needed yet)
    ds = Dataset.load(dataset_key, calendar=cal, feature_providers=[feats])

    print('Downloading missing OpenMeteo data ...')
    feats.download(ds.observations, verbose=verbose)

    print('Building cache ...')
    FeatureCache.build(feats, ds.observations, path=path, verbose=verbose)

    print(f'\nDone. Cache written to: {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build an OpenMeteo FeatureCache for a registered dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset_key',
        help='Registry key of the dataset (e.g. PEP725_Apple).',
    )
    parser.add_argument(
        '--keys',
        nargs='+',
        default=DEFAULT_KEYS,
        metavar='KEY',
        help='OpenMeteo variable names to include in the cache.',
    )
    parser.add_argument(
        '--step',
        default=DEFAULT_STEP,
        help='Temporal resolution label (informational; used in the cache filename).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite an existing cache file.',
    )
    parser.add_argument(
        '--root',
        default=None,
        type=Path,
        help='Data root directory (overrides PYSEPHONE_DATA_ROOT).',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bars.',
    )

    args = parser.parse_args()

    build_cache(
        dataset_key=args.dataset_key,
        data_keys=args.keys,
        step=args.step,
        force=args.force,
        root=args.root,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
