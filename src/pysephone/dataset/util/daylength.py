"""
Daylength (photoperiod) feature provider for use with Dataset.

Daylength is computed analytically from latitude and day-of-year using the
Forsythe et al. (1995) revised "CBM" model, so no download step is needed —
only a preload that builds the ``(src, loc_id) -> latitude`` map from an
Observations instance (mirroring :class:`ElevationFeatures`).

Reference:
    Forsythe, W. C., E. J. Rykiel Jr., R. S. Stahl, H.-i. Wu, and
    R. M. Schoolfield. 1995. "A model comparison for daylength as a
    function of latitude and day of year." Ecological Modelling 80:87-95.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.provider import FeatureProvider


# Sun-position coefficient (degrees) from Forsythe et al. 1995.
# 0.8333 = sunrise/sunset including standard atmospheric refraction.
_DAYLENGTH_COEFF_DEG = 0.8333


def compute_daylength_from_doy(latitude: float, doy) -> np.ndarray:
    """Daylength (hours) from latitude and day-of-year, vectorised.

    Implements the Forsythe et al. 1995 revised CBM model.

    Args:
        latitude: Latitude in degrees in [-90, 90].  Positive = northern hemisphere.
        doy:      Day-of-year, scalar int or array-like of ints in [1, 366].

    Returns:
        Daylength in hours as ``float32``.  Shape matches ``doy``.  Polar day
        is clamped to 24.0 h and polar night to 0.0 h (no NaN).
    """
    L = np.deg2rad(float(latitude))
    J = np.asarray(doy, dtype=np.float64)

    theta = 0.2163108 + 2.0 * np.arctan(
        0.9671396 * np.tan(0.00860 * (J - 186.0))
    )
    phi = np.arcsin(0.39795 * np.cos(theta))

    num = np.sin(np.deg2rad(_DAYLENGTH_COEFF_DEG)) + np.sin(L) * np.sin(phi)
    den = np.cos(L) * np.cos(phi)

    # At polar latitudes near the solstices, |num/den| exceeds 1.
    # Clipping yields D = 24 (polar day) or D = 0 (polar night).
    arg = np.clip(num / den, -1.0, 1.0)
    D = 24.0 - (24.0 / np.pi) * np.arccos(arg)

    return D.astype(np.float32)


def compute_daylength(latitude: float, when) -> Union[np.ndarray, float]:
    """Daylength (hours) for a date, a sequence of dates, or a day-of-year.

    Convenience wrapper around :func:`compute_daylength_from_doy` that
    accepts date-like inputs in addition to raw day-of-year integers.

    Args:
        latitude: Latitude in degrees.
        when:     One of
                  - ``datetime.date`` / ``datetime.datetime`` (returns ``float``),
                  - ``np.datetime64`` scalar (returns ``float``),
                  - ``np.ndarray[datetime64]`` (returns ``ndarray``),
                  - ``pd.DatetimeIndex`` or list/tuple of date-like (returns ``ndarray``),
                  - ``int`` / ``np.integer`` day-of-year (returns ``float``),
                  - ``np.ndarray`` of ints, day-of-year (returns ``ndarray``).

    Returns:
        ``float`` for scalar inputs, ``np.ndarray`` of ``float32`` otherwise.
    """
    if isinstance(when, np.ndarray):
        if np.issubdtype(when.dtype, np.datetime64):
            doy = pd.DatetimeIndex(when).dayofyear.to_numpy()
            return compute_daylength_from_doy(latitude, doy)
        if np.issubdtype(when.dtype, np.integer):
            return compute_daylength_from_doy(latitude, when)
        raise TypeError(
            f"compute_daylength: numpy array with dtype {when.dtype} is not supported. "
            "Use datetime64 or integer dtype."
        )

    if isinstance(when, pd.DatetimeIndex):
        return compute_daylength_from_doy(latitude, when.dayofyear.to_numpy())

    # Scalars
    if isinstance(when, np.datetime64):
        return float(compute_daylength_from_doy(latitude, pd.Timestamp(when).dayofyear))
    if isinstance(when, (_dt.datetime, _dt.date)):
        return float(compute_daylength_from_doy(latitude, pd.Timestamp(when).dayofyear))
    if isinstance(when, (int, np.integer)):
        return float(compute_daylength_from_doy(latitude, int(when)))

    if isinstance(when, (list, tuple)):
        return compute_daylength_from_doy(
            latitude, pd.DatetimeIndex(when).dayofyear.to_numpy()
        )

    raise TypeError(
        f"compute_daylength: unsupported type {type(when).__name__} for 'when'. "
        "Pass a date / datetime / datetime64 / DatetimeIndex / array of "
        "datetime64 / int / array of ints (day-of-year)."
    )


class DaylengthFeatures(FeatureProvider):
    """Per-day daylength (hours) as a feature.

    Args:
        calendar: :class:`Calendar` used to determine season start/end for each entry.
        key:      Feature name in the dict returned by :meth:`get_data`.

    Notes:
        Daylength is computed analytically, so there is no ``download()`` step;
        only :meth:`preload` (which captures latitude per location) is needed
        before :meth:`get_data` can be called.
    """

    DEFAULT_KEY = 'daylength'
    STEP = 'daily'

    def __init__(self, calendar: Calendar, key: str = DEFAULT_KEY) -> None:
        self._calendar = calendar
        self._key = str(key)
        # (src, loc_id) -> latitude in degrees
        self._loc_lat: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        return {self._key: self._get_array(index)}

    def preload(self, observations, verbose: bool = True) -> None:
        """Populate the ``(src, loc_id) -> latitude`` map from *observations*."""
        seen: set = set()
        pairs = []
        for ix in observations.iter_index():
            src, loc_id = ix[0], ix[1]
            if (src, loc_id) in seen:
                continue
            seen.add((src, loc_id))
            pairs.append((src, loc_id))

        it = tqdm(pairs, desc='Preloading daylength') if verbose else pairs
        for src, loc_id in it:
            coords = observations.get_location_coordinates((src, loc_id))
            self._loc_lat[(src, loc_id)] = float(coords['lat'])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return self._key

    @property
    def calendar(self) -> Calendar:
        return self._calendar

    @property
    def step(self) -> str:
        return self.STEP

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_array(self, index: Tuple) -> np.ndarray:
        src, loc_id, year, species_id, subgroup_id = index
        if (src, loc_id) not in self._loc_lat:
            raise RuntimeError(
                f"DaylengthFeatures: no latitude loaded for ({src!r}, {loc_id!r}). "
                "Call Dataset.preload_features() or preload(observations) directly."
            )

        season = self._calendar.get_season_info(
            year=year, src=src, species_id=species_id,
            subgroup_id=subgroup_id, loc_id=loc_id,
        )
        # season_end is exclusive; np.arange yields exactly season_length days.
        dates = np.arange(
            season['season_start'], season['season_end'], dtype='datetime64[D]',
        )
        doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
        return compute_daylength_from_doy(self._loc_lat[(src, loc_id)], doy)
