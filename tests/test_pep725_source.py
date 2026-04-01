"""
Tests for PEP725Source.

Download, extraction, and DataFrame creation are mocked — no network access or
credentials required. The metadata CSVs bundled with the package are read for real,
so the tests also exercise that the entry set is correctly constructed from them.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from pysephone.data.source import ObservationData
from pysephone.data.pep725.source import PEP725Source
from pysephone.constants import KEYS_INDEX, KEY_LOC_ID, KEY_OBS_TYPE


_MODULE = 'pysephone.data.pep725.source'

# Minimal stub DataFrames returned by mocked create_* functions
_STUB_OBSERVATIONS = pd.DataFrame(
    {'DAY': [100]},
    index=pd.MultiIndex.from_tuples(
        [('pep725', 1, 2020, 333, 1, 10)],
        names=[*KEYS_INDEX, KEY_OBS_TYPE],
    ),
)
_STUB_EVENTS = pd.DataFrame({'description': ['Flowering']}, index=pd.Index([60], name='bbch'))
_STUB_LOCATIONS = pd.DataFrame(
    {'LON': [5.0], 'LAT': [52.0], 'ALT': [10.0], 'NAME': ['Test Station'], 'country_code': ['NL']},
    index=pd.Index([1], name=KEY_LOC_ID),
)


@pytest.fixture
def source():
    return PEP725Source()


@pytest.fixture
def tmp_root(tmp_path):
    """Temporary data root — no real data present, simulating a fresh install."""
    return tmp_path


def _all_present():
    """Simulate check_entries_missing: all data already on disk, nothing to do."""
    return {'data': [], 'download': []}


def _no_downloads():
    return {'successful': set(), 'failed': set()}


@patch(f'{_MODULE}.create_events_df', return_value=_STUB_EVENTS)
@patch(f'{_MODULE}.create_locations_df', return_value=_STUB_LOCATIONS)
@patch(f'{_MODULE}.create_observations_df', return_value=_STUB_OBSERVATIONS)
@patch(f'{_MODULE}.extract_entries')
@patch(f'{_MODULE}.download_entries', return_value=_no_downloads())
@patch(f'{_MODULE}.check_entries_missing', return_value=_all_present())
def test_get_data_returns_observation_data(
    mock_check, mock_download, mock_extract,
    mock_obs, mock_loc, mock_events,
    source, tmp_root,
):
    result = source.get_data({}, root=tmp_root)

    assert isinstance(result, ObservationData)


@patch(f'{_MODULE}.create_events_df', return_value=_STUB_EVENTS)
@patch(f'{_MODULE}.create_locations_df', return_value=_STUB_LOCATIONS)
@patch(f'{_MODULE}.create_observations_df', return_value=_STUB_OBSERVATIONS)
@patch(f'{_MODULE}.extract_entries')
@patch(f'{_MODULE}.download_entries', return_value=_no_downloads())
@patch(f'{_MODULE}.check_entries_missing', return_value=_all_present())
def test_get_data_dataframes_have_expected_structure(
    mock_check, mock_download, mock_extract,
    mock_obs, mock_loc, mock_events,
    source, tmp_root,
):
    result = source.get_data({}, root=tmp_root)

    assert 'observations' in result.observations.columns
    assert result.observations.index.names == [*KEYS_INDEX, KEY_OBS_TYPE]


@patch(f'{_MODULE}.create_events_df', return_value=_STUB_EVENTS)
@patch(f'{_MODULE}.create_locations_df', return_value=_STUB_LOCATIONS)
@patch(f'{_MODULE}.create_observations_df', return_value=_STUB_OBSERVATIONS)
@patch(f'{_MODULE}.extract_entries')
@patch(f'{_MODULE}.download_entries', return_value=_no_downloads())
@patch(f'{_MODULE}.check_entries_missing', return_value=_all_present())
def test_no_download_or_extract_when_all_present(
    mock_check, mock_download, mock_extract,
    mock_obs, mock_loc, mock_events,
    source, tmp_root,
):
    source.get_data({}, root=tmp_root)

    mock_download.assert_called_once()
    entries_to_download = mock_download.call_args[0][0]
    assert len(entries_to_download) == 0

    mock_extract.assert_called_once()
    entries_to_extract = mock_extract.call_args[0][0]
    assert len(entries_to_extract) == 0


@patch(f'{_MODULE}.create_events_df', return_value=_STUB_EVENTS)
@patch(f'{_MODULE}.create_locations_df', return_value=_STUB_LOCATIONS)
@patch(f'{_MODULE}.create_observations_df', return_value=_STUB_OBSERVATIONS)
@patch(f'{_MODULE}.extract_entries')
@patch(f'{_MODULE}.download_entries', return_value=_no_downloads())
@patch(f'{_MODULE}.check_entries_missing', return_value=_all_present())
def test_check_not_called_when_force_download(
    mock_check, mock_download, mock_extract,
    mock_obs, mock_loc, mock_events,
    source, tmp_root,
):
    source.get_data({'force_download': True}, root=tmp_root)

    mock_check.assert_not_called()
