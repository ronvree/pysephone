"""
Build (or rebuild) an AgERA5 FeatureCache for a registered dataset.

Usage
-----
    python scripts/build_agera5_cache.py PEP725_Apple
    python scripts/build_agera5_cache.py PEP725_Apple --keys Temperature_Air_2m_Mean_24h Solar_Radiation_Flux
    python scripts/build_agera5_cache.py GMU_Cherry_Japan_Y --force

Arguments
---------
dataset_key     Registry key of the dataset to cache (required).
--keys          One or more AgERA5 data keys (variable names as they appear in
                the AgERA5 NetCDF files).
                Default: Temperature_Air_2m_Mean_24h
--force         Overwrite an existing cache file.
--root          Override the data root (defaults to PYSEPHONE_DATA_ROOT env var
                or the repository root).
--quiet         Suppress progress bars.

CDS authentication: this script delegates to cdsapi, which reads credentials
from ~/.cdsapirc or the CDSAPI_URL / CDSAPI_KEY environment variables.  See
https://cds.climate.copernicus.eu/api-how-to for setup instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pysephone.data.agera5.download import DEFAULT_TILE_DEG
from pysephone.dataset.dataset import Dataset
from pysephone.dataset.util.agera5 import AgEra5Features
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.feature_cache import FeatureCache


DEFAULT_KEYS = ['Temperature_Air_2m_Mean_24h']
STEP = 'daily'


def build_cache(
    dataset_key: str,
    data_keys: list[str],
    force: bool,
    root: Path | None,
    verbose: bool,
    tile_deg: float,
) -> None:
    path = FeatureCache.default_path(dataset_key, data_keys, step=STEP, root=root)

    if FeatureCache.exists(path) and not force:
        print(f'Cache already exists: {path}')
        print('Pass --force to overwrite.')
        return

    print(f'Dataset  : {dataset_key}')
    print(f'Keys     : {data_keys}')
    print(f'Step     : {STEP}')
    print(f'Tile size: {tile_deg}°')
    print(f'Output   : {path}')
    print()

    cal   = Calendar()
    feats = AgEra5Features(calendar=cal, data_keys=data_keys, root=root)

    ds = Dataset.load(dataset_key, calendar=cal, feature_providers=[feats])

    print('Downloading missing AgERA5 data ...')
    feats.download(ds.observations, verbose=verbose, tile_deg=tile_deg)

    print('Building cache ...')
    FeatureCache.build(feats, ds.observations, path=path, verbose=verbose)

    print(f'\nDone. Cache written to: {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build an AgERA5 FeatureCache for a registered dataset.',
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
        help='AgERA5 variable names to include in the cache.',
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
    parser.add_argument(
        '--tile-deg',
        default=DEFAULT_TILE_DEG,
        type=float,
        help=(
            'Geographic tile size (degrees) for chunking CDS requests. '
            'Shrink if a region\'s per-request cost exceeds CDS\'s 400-unit '
            'limit.'
        ),
    )

    args = parser.parse_args()

    build_cache(
        dataset_key=args.dataset_key,
        data_keys=args.keys,
        force=args.force,
        root=args.root,
        verbose=not args.quiet,
        tile_deg=args.tile_deg,
    )


if __name__ == '__main__':
    main()
