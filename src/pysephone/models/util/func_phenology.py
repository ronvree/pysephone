"""
Phenology calculation functions.

This module provides functions for calculating growing degree days,
vernalization factors, and photoperiod factors used in phenology modeling.
"""

from typing import Optional
import numpy as np


def func_growing_degree_units(
    temperature: np.ndarray, 
    t_base: float, 
    t_upper: Optional[float] = None
) -> np.ndarray:
    """
    Calculate growing degree days (GDD) from temperature.
    
    GDD is calculated as max(0, temperature - t_base). If t_upper is provided,
    temperatures above t_upper are clipped before calculation.
    
    Args:
        temperature: Array of temperature values.
        t_base: Base temperature threshold. GDD is 0 below this temperature.
        t_upper: Optional upper temperature limit. If provided, temperatures
                above this are clipped to t_upper.
    
    Returns:
        Array of growing degree days.
    
    Raises:
        ValueError: If t_upper is provided and t_upper < t_base.
    """
    if t_upper is not None:
        if t_upper < t_base:
            raise ValueError(f"t_upper ({t_upper}) must be >= t_base ({t_base})")
        temperature = temperature.clip(upper=t_upper)
    return (temperature - t_base).clip(min=0)


def func_growing_degree_units_2(
    temperature: np.ndarray, 
    t_base: float, 
    t_upper: float, 
    t_limit: float
) -> np.ndarray:
    """
    Calculate growing degree days with a two-segment temperature response.
    
    This function implements a more complex GDD calculation with different
    responses in two temperature ranges:
    - Between t_base and t_limit: GDD = temperature
    - Between t_limit and t_upper: GDD = (t_upper - temperature) * (t_limit - t_base) / (t_upper - t_limit)
    - Outside these ranges: GDD = 0
    
    Args:
        temperature: Array of temperature values.
        t_base: Lower temperature threshold.
        t_upper: Upper temperature threshold.
        t_limit: Temperature at which the response changes from linear to decreasing.
                Must satisfy: t_base < t_limit < t_upper.
    
    Returns:
        Array of growing degree days.
    
    Raises:
        ValueError: If parameter relationships are invalid (t_base >= t_limit,
                   t_limit >= t_upper, or t_upper == t_limit).
    """
    if t_base >= t_limit:
        raise ValueError(f"t_base ({t_base}) must be < t_limit ({t_limit})")
    if t_limit >= t_upper:
        raise ValueError(f"t_limit ({t_limit}) must be < t_upper ({t_upper})")
    if t_upper == t_limit:
        raise ValueError(f"t_upper ({t_upper}) cannot equal t_limit ({t_limit})")
    
    units = np.zeros_like(temperature)
    
    mask1 = (t_base <= temperature) & (temperature <= t_limit)
    mask2 = (t_limit < temperature) & (temperature <= t_upper)
    
    units[mask1] = temperature[mask1]
    
    # Safe division since we validated t_upper != t_limit
    units[mask2] = ((t_upper - temperature[mask2]) * (t_limit - t_base)) / (t_upper - t_limit)
    
    return units


# Winter wheat vernalization parameters based on "Climate change effects on wheat phenology depends on cultivar change":
# Vernalization Module was calibrated for winter wheat in germany
# v unit is 0 for days with mean temperature below -4 deg. C or above 17 deg. C
# v unit is 1 for days with mean temperature between 4 and 10 deg. C
# v unit is linearly interpolated in missing segments
# 30 units need to be accumulated for the vernalization factor to be 1
def func_vernalization_unit(x: np.ndarray) -> np.ndarray:
    """
    Calculate vernalization units from temperature.
    
    Vernalization units are calculated based on temperature:
    - 0 for temperatures below -4°C or above 17°C
    - 1 for temperatures between 4°C and 10°C
    - Linearly interpolated between these thresholds
    
    Args:
        x: Array of temperature values in degrees Celsius.
    
    Returns:
        Array of vernalization units (0 to 1).
    """
    return np.interp(x, [-4, 4, 10, 17], [0, 1, 1, 0], left=0, right=0)


def func_vernalization_tres(x: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """
    Calculate vernalization factor from accumulated vernalization units.
    
    Maps accumulated vernalization units to a factor between 0 and 1.
    The factor reaches 1 when accumulated units reach the threshold.
    
    Args:
        x: Array of accumulated vernalization units.
        threshold: Number of units required for full vernalization (default: 30.0).
    
    Returns:
        Array of vernalization factors (0 to 1).
    
    Raises:
        ValueError: If threshold <= 0.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    return np.interp(x, [0, threshold], [0, 1], left=0, right=1)


# In "Climate change effects on wheat phenology depends on cultivar change":
# pp factor is 0 for days shorter than 7h
# pp factor is 1 for days longer than 17h
# pp is linearly interpolated between
def func_photoperiod_factor(
    x: np.ndarray, 
    p_base: float = 7.0, 
    p_sat: float = 17.0
) -> np.ndarray:
    """
    Calculate photoperiod factor from daylight duration.
    
    Photoperiod factor is calculated based on daylight hours:
    - 0 for days shorter than p_base hours
    - 1 for days longer than p_sat hours
    - Linearly interpolated between p_base and p_sat
    
    Args:
        x: Array of daylight duration values in hours.
        p_base: Minimum daylight hours for any response (default: 7.0).
        p_sat: Daylight hours for maximum response (default: 17.0).
    
    Returns:
        Array of photoperiod factors (0 to 1).
    
    Raises:
        ValueError: If p_base >= p_sat.
    """
    if p_base >= p_sat:
        raise ValueError(f"p_base ({p_base}) must be < p_sat ({p_sat})")
    return np.interp(x, [p_base, p_sat], [0, 1], left=0, right=1)


def func_chilling_days(temperature: np.ndarray, t_threshold: float = 7.2) -> np.ndarray:
    """One chill unit per day when mean temperature is at or below *t_threshold*.

    A simple proxy for chilling hours when only daily mean temperature is
    available.  The classic threshold is 7.2 °C (45 °F).

    Args:
        temperature: Array of daily mean temperatures in °C.
        t_threshold: Temperature threshold in °C (default: 7.2).

    Returns:
        Array of chill units (0 or 1 per day).
    """
    return (temperature <= t_threshold).astype(float)


def func_utah_chill(
        x: np.ndarray,
) -> np.ndarray:

    bin_0 = (x <= 1.4).astype(x.dtype)
    bin_1 = ((1.4 < x) & (x <= 2.4)).astype(x.dtype)
    bin_2 = ((2.4 < x) & (x <= 9.1)).astype(x.dtype)
    bin_3 = ((9.1 < x) & (x <= 12.4)).astype(x.dtype)
    bin_4 = ((11.4 < x) & (x <= 15.9)).astype(x.dtype)
    bin_5 = ((15.9 < x) & (x <= 18)).astype(x.dtype)
    bin_6 = (18 < x).astype(x.dtype)

    bin_0 *= 0.
    bin_1 *= 0.5
    bin_2 *= 1.
    bin_3 *= 0.5
    bin_4 *= 0.
    bin_5 *= -0.5
    bin_6 *= -1

    return bin_0 + bin_1 + bin_2 + bin_3 + bin_4 + bin_5 + bin_6


def func_dynamic_chill_hourly(
    hour_temp: np.ndarray,
    E0: float = 4153.5,
    E1: float = 12888.8,
    A0: float = 139500.0,
    A1: float = 2.567e18,
    slope: float = 1.6,
    Tf: float = 277.0,
) -> np.ndarray:
    """Dynamic chill model (Fishman et al. 1987) — hourly chill-portion increments.

    Computes per-hour chill portions using the two-state kinetic model.
    Physical parameters (E0, E1, A0, A1, slope, Tf) are taken from
    Luedeling et al. and are not typically calibrated.

    Args:
        hour_temp: 1-D array of hourly temperatures in °C.
        E0:        Activation energy of the intermediate state (cal/mol).
        E1:        Activation energy of the high-energy state (cal/mol).
        A0:        Frequency factor for the intermediate state.
        A1:        Frequency factor for the high-energy state.
        slope:     Slope of the sigmoidal function.
        Tf:        Transition temperature of the sigmoidal function (K).

    Returns:
        1-D array of hourly chill-portion increments (same length as *hour_temp*).
    """
    temp_c = np.asarray(hour_temp, dtype=float)
    if temp_c.size == 0:
        return np.array([], dtype=float)

    TK = temp_c + 273.0
    aa = A0 / A1
    ee = E1 - E0

    sr = np.exp(slope * Tf * (TK - Tf) / TK)
    xi = sr / (1.0 + sr)
    xs = aa * np.exp(ee / TK)
    eak1 = np.exp(-A1 * np.exp(-E1 / TK))

    x = np.zeros(temp_c.size, dtype=float)
    for i in range(1, temp_c.size):
        S = x[i - 1]
        if x[i - 1] >= 1.0 and i >= 2:
            S = S * (1.0 - xi[i - 2])
        x[i] = xs[i - 1] - (xs[i - 1] - S) * eak1[i - 1]

    delta = np.zeros(temp_c.size, dtype=float)
    ii = np.where(x >= 1.0)[0]
    ii = ii[ii >= 1]
    delta[ii] = x[ii] * xi[ii - 1]
    return delta


def func_hourly_from_tmean(
    t_mean: np.ndarray,
    amplitude: float = 5.0,
    tmin_hour: int = 6,
) -> np.ndarray:
    """Reconstruct synthetic hourly temperatures from daily mean values.

    Uses a cosine curve centred on *t_mean* with a daily minimum at
    *tmin_hour* and a corresponding maximum 12 hours later.

    Args:
        t_mean:    1-D array of daily mean temperatures in °C.
        amplitude: Half the daily temperature range (°C).  Temperature
                   oscillates between ``t_mean ± amplitude``.
        tmin_hour: Hour of day (0–23) at which the daily minimum occurs.

    Returns:
        1-D array of length ``24 * len(t_mean)`` with hourly temperatures.
    """
    t_mean = np.asarray(t_mean, dtype=float)
    hours = np.arange(24)
    phase = 2.0 * np.pi * (hours - tmin_hour) / 24.0
    daily_pattern = -np.cos(phase)   # minimum at tmin_hour, maximum 12 h later
    return (t_mean[:, np.newaxis] + amplitude * daily_pattern).ravel()


def func_dynamic_chill_daily(
    t_mean: np.ndarray,
    amplitude: float = 5.0,
    tmin_hour: int = 6,
) -> np.ndarray:
    """Daily chill-portion increments via the Dynamic Model from daily mean temperatures.

    Reconstructs synthetic hourly temperatures using a cosine cycle
    (:func:`func_hourly_from_tmean`), runs :func:`func_dynamic_chill_hourly`,
    then sums 24-hour windows to obtain one value per day.

    Args:
        t_mean:    1-D array of daily mean temperatures in °C  (length *D*).
        amplitude: Half the synthetic daily temperature range (°C).
        tmin_hour: Hour of day at which the daily minimum occurs.

    Returns:
        1-D array of length *D* with daily chill-portion increments.
    """
    hourly_temp = func_hourly_from_tmean(t_mean, amplitude=amplitude, tmin_hour=tmin_hour)
    cp_hourly = func_dynamic_chill_hourly(hourly_temp)
    return cp_hourly.reshape(-1, 24).sum(axis=1)


