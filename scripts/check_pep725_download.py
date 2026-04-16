#!/usr/bin/env python
"""
One-off smoke test: can we still download data from pep725.eu?

All data is written to a temporary directory — the existing data folder is
never read or modified.

Usage::

    python scripts/check_pep725_download.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

from pysephone.data.pep725.download import (
    PEP725Entry,
    download_entries,
    extract_entries,
)
from pysephone.paths import get_data_root, get_observations_source_data_dir


# A small entry known to exist (Netherlands, Hazel, no subgroup)
_TEST_ENTRY = PEP725Entry(
    species_key='107_000',
    species_code=107,
    subgroup_code=0,
    country_code='NL',
    species_name='Corylus avellana',
    subgroup_name='',
    country_name='Netherlands',
)


def main():
    real_creds = get_data_root() / 'data' / 'observations' / 'pep725' / 'credentials.txt'
    if not real_creds.exists():
        print(f'SKIP: credentials not found at {real_creds}')
        sys.exit(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)

        # Copy credentials into the temp tree
        creds_dir = get_observations_source_data_dir(tmp_root, 'pep725')
        creds_dir.mkdir(parents=True)
        shutil.copy(real_creds, creds_dir / 'credentials.txt')

        # -- Step 1: download ------------------------------------------------
        print(f'Downloading {_TEST_ENTRY.download_url} ...')
        result = download_entries(
            entries={_TEST_ENTRY},
            root=tmp_root,
            credentials_path=creds_dir / 'credentials.txt',
            verbose=True,
        )

        n_ok = len(result['successful'])
        n_fail = len(result['failed'])
        print(f'Download: {n_ok} successful, {n_fail} failed')

        if n_fail > 0:
            print('FAIL: download returned non-gzip response.')
            print('The PEP725 server may have changed or credentials may be invalid.')
            sys.exit(1)

        tar_path = _TEST_ENTRY.path_download_file(tmp_root)
        assert tar_path.exists() and tar_path.stat().st_size > 0, 'Download file missing or empty'
        print(f'Saved {tar_path.stat().st_size} bytes to {tar_path.name}')

        # -- Step 2: extract --------------------------------------------------
        print('Extracting ...')
        extract_entries(result['successful'], root=tmp_root, verbose=True)

        data_folder = _TEST_ENTRY.path_data_folder(tmp_root)
        csv_files = list(data_folder.glob('*.csv'))
        print(f'Extracted {len(csv_files)} CSV files: {[f.name for f in csv_files]}')

        assert len(csv_files) >= 2, f'Expected at least 2 CSVs, got {len(csv_files)}'

        # -- Step 3: verify data is readable ----------------------------------
        stations_df = _TEST_ENTRY.get_stations_df(tmp_root)
        obs_df = _TEST_ENTRY.get_data_df(tmp_root)

        print(f'Stations: {len(stations_df)} rows  |  Observations: {len(obs_df)} rows')
        assert len(stations_df) > 0, 'Stations dataframe empty'
        assert len(obs_df) > 0, 'Observations dataframe empty'

        print('\nOK: PEP725 download, extraction, and parsing all work.')


if __name__ == '__main__':
    main()
