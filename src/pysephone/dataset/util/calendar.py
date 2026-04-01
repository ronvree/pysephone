from collections import defaultdict
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np


class Calendar:
    """
    Maps (src, species_id, subgroup_id) → season window definition.

    A season is defined by a start date (MM-DD string) and a length in days.
    The calendar determines for a given observation year where the season
    window begins and ends, handling cases where the season starts in the
    previous calendar year (e.g. winter crops sown in autumn).

    Usage::

        cal = Calendar()
        cal.set_season('pep725', species_id=333, subgroup_id=300,
                       start_date='10-01', length=365)
        info = cal.get_season_info(year=2020, species_id=333, subgroup_id=300,
                                   src='pep725')

    The returned dict contains:
        season_start  np.datetime64  – first day of the season window
        season_end    np.datetime64  – first day *after* the season window
        season_length np.timedelta64 – total window length
        year          int            – the observation year
    """

    # Default season applied to any species not explicitly registered:
    # starts October 1st of the year prior to the observation year, runs 365 days.
    _DEFAULT_START = '10-01'
    _DEFAULT_LENGTH = 365

    def __init__(self, calendar: Optional[Dict] = None) -> None:
        """
        Args:
            calendar: Pre-populated dict mapping (src, species_id, subgroup_id)
                      to (start_date: str, length: int).
                      Defaults to a ``defaultdict`` using the class defaults.
        """
        self._calendar = calendar or defaultdict(
            lambda: (self._DEFAULT_START, self._DEFAULT_LENGTH)
        )

    # ------------------------------------------------------------------
    # Season registration
    # ------------------------------------------------------------------

    def set_season(
        self,
        src: str,
        species_id,
        start_date: str,
        length: int,
        subgroup_id=None,
        loc_id: str = None,  # reserved for future location-specific calendars
        year: int = None,    # reserved for future year-specific calendars
    ) -> None:
        """Register or overwrite a season definition.

        Args:
            src:        Data-source identifier (e.g. ``'pep725'``).
            species_id: Species identifier.
            start_date: Season start as ``'MM-DD'`` string (e.g. ``'10-01'``).
            length:     Season length in days.
            subgroup_id: Optional subgroup / cultivar identifier.
            loc_id:     Placeholder — location-specific calendars not yet supported.
            year:       Placeholder — year-specific calendars not yet supported.
        """
        self._calendar[src, species_id, subgroup_id] = (start_date, length)
        # Invalidate any cached result that depended on the old value
        self.get_season_info.cache_clear()

    # ------------------------------------------------------------------
    # Season lookup
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def get_season_info(
        self,
        year: int,
        src: str,
        species_id,
        subgroup_id=None,
        loc_id: str = None,  # reserved
    ) -> Dict:
        """Return season boundaries for the given observation year.

        The season window is anchored to *year*: if the start-date in the
        previous calendar year falls within the correct range so that the
        season ends in *year*, that start is used; otherwise the start is
        placed in *year* itself.

        Args:
            year:       Observation year.
            src:        Data-source identifier.
            species_id: Species identifier.
            subgroup_id: Optional subgroup identifier.
            loc_id:     Currently unused; reserved for future location-aware calendars.

        Returns:
            Dict with keys ``season_start``, ``season_end``, ``season_length``,
            and ``year``.
        """
        start_date, length = self._calendar[src, species_id, subgroup_id]

        td = np.timedelta64(length, 'D')

        # Try season starting in the *previous* calendar year first
        season_start = np.datetime64(f'{year - 1}-{start_date}')
        season_end = season_start + td

        # If the season does not end in the target year, start in the target year
        end_year = int(season_end.astype('datetime64[Y]').astype(int)) + 1970
        if end_year != year:
            season_start = np.datetime64(f'{year}-{start_date}')
            season_end = season_start + td

        return {
            'season_start': season_start,
            'season_end': season_end,
            'season_length': td,
            'year': year,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> 'Calendar':
        """Create a Calendar with the default (unspecified) season definitions.

        Individual datasets should call :meth:`set_season` to register their
        species before use.
        """
        return cls()
