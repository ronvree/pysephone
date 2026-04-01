"""
Fake weather feature provider for testing and synthetic experiments.

Generates artificial daily temperature series from a parametric seasonal
model (cosine + Gaussian noise) without requiring any real meteorological
data.  Implements :class:`~pysephone.dataset.util.provider.FeatureProvider`
so it can be passed directly to :class:`~pysephone.dataset.dataset.Dataset`.

Example::

    from pysephone.dataset.util.fake_weather import FakeWeatherProvider

    provider = FakeWeatherProvider(
        calendar=cal,
        mean_winter_temp=2.0,
        mean_summer_temp=18.0,
        seed=42,
    )
    dataset = Dataset(obs, calendar=cal, feature_providers=[provider])
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.provider import FeatureProvider


def generate_temperature_series(
    n_days: int = 365,
    mean_winter_temp: float = 4.0,
    mean_summer_temp: float = 20.0,
    noise_std: float = 3.0,
    season_start_doy: int = 274,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic daily mean temperature series for a single season.

    The seasonal cycle is modelled as a cosine, with the minimum occurring
    around mid-winter (day 90 of the season, roughly Jan 1 when starting
    Oct 1) and the maximum around mid-summer.

    Parameters
    ----------
    n_days : int
        Season length in days.
    mean_winter_temp : float
        Mean temperature at the coldest point of the season (°C).
        Lower values -> more chilling available.
    mean_summer_temp : float
        Mean temperature at the warmest point of the season (°C).
        Higher values -> more forcing available.
    noise_std : float
        Standard deviation of day-to-day Gaussian noise (°C).
    season_start_doy : int
        Day-of-year on which the season starts (default 274 = Oct 1).
        Used only to phase the cosine correctly.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    temps : np.ndarray, shape (n_days,)
        Daily mean temperatures in °C.
    """
    rng = np.random.default_rng(seed)

    # Phase the cosine so the minimum aligns with ~Jan 1.
    # From Oct 1, mid-winter (Jan 1) is ~day 92.
    mid_winter_day = 92
    days = np.arange(int(n_days), dtype=np.float64)
    phase = 2 * np.pi * (days - mid_winter_day) / 365.0

    amplitude = (mean_summer_temp - mean_winter_temp) / 2.0
    baseline  = (mean_summer_temp + mean_winter_temp) / 2.0
    seasonal  = baseline - amplitude * np.cos(phase)

    noise = rng.normal(0, noise_std, size=n_days)
    return (seasonal + noise).astype(np.float32)


class FakeWeatherProvider(FeatureProvider):
    """Feature provider that returns synthetic temperature series.

    Season length is looked up from a :class:`Calendar` for each sample.
    All other parameters are shared across every sample.

    The generated array is returned under the key ``'temperature_2m_mean'``
    to match the naming convention used by
    :class:`~pysephone.dataset.util.openmeteo.OpenMeteoFeatures`.

    Args:
        calendar:        Calendar used to determine season length for each sample.
        mean_winter_temp: Mean temperature at the coldest point (°C).
        mean_summer_temp: Mean temperature at the warmest point (°C).
        noise_std:       Std. dev. of day-to-day Gaussian noise (°C).
        seed:            Base random seed.  Each sample derives a unique seed from
                         this base and its index so results are reproducible but
                         differ across samples.
    """

    KEY_TEMPERATURE = 'temperature_2m_mean'

    def __init__(
        self,
        calendar: Calendar,
        mean_winter_temp: float = 4.0,
        mean_summer_temp: float = 20.0,
        noise_std: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        self._calendar = calendar
        self._mean_winter_temp = mean_winter_temp
        self._mean_summer_temp = mean_summer_temp
        self._noise_std = noise_std
        self._base_seed = seed

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return a synthetic temperature series for *index*.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)``

        Returns:
            ``{'temperature_2m_mean': np.ndarray}`` — shape ``(season_length,)``
        """
        src, loc_id, year, species_id, subgroup_id = index

        season = self._calendar.get_season_info(
            year=year,
            src=src,
            species_id=species_id,
            subgroup_id=subgroup_id,
            loc_id=loc_id,
        )

        n_days: int = int(season['season_length'] / np.timedelta64(1, 'D'))
        season_start = season['season_start']
        jan1 = season_start.astype('datetime64[Y]').astype('datetime64[D]')
        season_start_doy: int = int((season_start.astype('datetime64[D]') - jan1) / np.timedelta64(1, 'D')) + 1

        # Derive a per-sample seed so samples differ but results are reproducible
        sample_seed: Optional[int] = None
        if self._base_seed is not None:
            sample_seed = hash((self._base_seed, index)) & 0xFFFFFFFF

        temps = generate_temperature_series(
            n_days=n_days,
            mean_winter_temp=self._mean_winter_temp,
            mean_summer_temp=self._mean_summer_temp,
            noise_std=self._noise_std,
            season_start_doy=season_start_doy,
            seed=sample_seed,
        )
        return {self.KEY_TEMPERATURE: temps}
