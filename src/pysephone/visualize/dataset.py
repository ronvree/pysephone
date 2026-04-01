"""
Visualisation functions for phenological observation datasets.

All functions accept an :class:`~pysephone.dataset.observations.Observations`
instance and return a :class:`matplotlib.figure.Figure`.  Nothing is written to
disk here — call :func:`save_figure` to persist output.

Example::

    from pysephone.dataset.observations import Observations
    from pysephone.visualize.dataset import (
        observation_doy_histograms,
        observation_doy_over_time,
        observation_mean_trend,
        observation_map,
        species_subgroup_timeline,
        save_figure,
    )

    obs = Observations(df_y, df_y_loc)

    fig = observation_doy_histograms(obs)
    save_figure(fig, path='outputs/', stem='histograms', formats=('png', 'svg'))
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from pysephone.constants import (
    KEY_LAT,
    KEY_LON,
    KEY_OBS_TYPE,
    KEY_OBSERVATIONS,
    KEY_YEAR,
)
from pysephone.dataset.observations import Observations


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_STYLE = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_colors(n: int) -> list:
    """Return *n* visually distinct colours from the tab10/tab20 palette."""
    if n <= 10:
        return [cm.tab10(i / 10) for i in range(n)]
    return [cm.tab20(i / 20) for i in range(n)]


def _as_axes_array(axs) -> np.ndarray:
    """Ensure ``axs`` is always a 1-D array, even for a single subplot."""
    axs = np.atleast_1d(axs)
    return axs.flatten()


def _map_bounds(
    obs: Observations, margin: float = 0.1
) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) with a proportional margin."""
    bb = obs.bounding_box
    dx = (bb['max_lon'] - bb['min_lon']) * margin or margin
    dy = (bb['max_lat'] - bb['min_lat']) * margin or margin
    return (
        bb['min_lon'] - dx,
        bb['min_lat'] - dy,
        bb['max_lon'] + dx,
        bb['max_lat'] + dy,
    )


def _naturalearth_countries():
    """Return a world countries GeoDataFrame.

    Uses the bundled ``world-administrative-boundaries.geojson`` when available;
    falls back to ``None`` (no background) if the file cannot be read.
    """
    from pathlib import Path as _Path
    bundled = (
        _Path(__file__).resolve().parents[1]
        / 'data' / 'resources' / 'world-administrative-boundaries.geojson'
    )
    try:
        import geopandas as gpd
        return gpd.read_file(bundled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Histogram of DOY distributions
# ---------------------------------------------------------------------------

def observation_doy_histograms(
    obs: Observations,
    n_bins: int = 40,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot DOY histograms for each observation type as stacked subplots.

    The x-axis is automatically fitted to the data range of each observation
    type (shared across subplots), so the distribution is always visible.

    Args:
        obs:     Observations instance.
        n_bins:  Number of histogram bins (default 40).
        figsize: Figure size override.  Defaults to ``(7, 2.5 * n_obs_types)``.

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    obs_types = obs.observation_types
    n = len(obs_types)
    colors = _get_colors(n)

    if figsize is None:
        figsize = (7, max(2.5 * n, 3))

    # Collect doys per obs_type and determine shared x range
    doys_by_type: dict[str, np.ndarray] = {}
    for obs_type in obs_types:
        col = obs.df_y.xs(obs_type, level=KEY_OBS_TYPE)[KEY_OBSERVATIONS]
        doys_by_type[obs_type] = col.apply(
            lambda d: d.dayofyear if hasattr(d, 'dayofyear') else int(d)
        ).values

    all_doys = np.concatenate(list(doys_by_type.values()))
    pad = max((all_doys.max() - all_doys.min()) * 0.06, 3)
    x_min, x_max = all_doys.min() - pad, all_doys.max() + pad

    with plt.rc_context(_STYLE):
        fig, axs = plt.subplots(nrows=n, sharex=True, sharey=False, figsize=figsize)
        axs = _as_axes_array(axs)

        for ax, obs_type, color in zip(axs, obs_types, colors):
            doys = doys_by_type[obs_type]
            ax.hist(doys, bins=n_bins, range=(x_min, x_max),
                    color=color, edgecolor='none', alpha=0.85)
            mean_doy = doys.mean()
            ax.axvline(mean_doy, color='#333333', linewidth=1.0, linestyle='--', alpha=0.7)
            label = obs_type.replace('BBCH_', 'BBCH ')
            ax.set_ylabel(f'{label}\n(n={len(doys):,})', fontsize=9, labelpad=4)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

        axs[-1].set_xlabel('Day of year')
        axs[-1].set_xlim(x_min, x_max)
        fig.suptitle('Observation DOY distributions', fontsize=12, y=1.01)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter: year vs DOY
# ---------------------------------------------------------------------------

def observation_doy_over_time(
    obs: Observations,
    figsize: Optional[Tuple[float, float]] = None,
    point_size: float = 2,
    alpha: float = 0.35,
) -> plt.Figure:
    """Scatter plots of observation DOY against year for each observation type.

    A horizontal dashed line marks the per-obs-type mean DOY.

    Args:
        obs:        Observations instance.
        figsize:    Figure size override.
        point_size: Marker size (default 2).
        alpha:      Marker transparency (default 0.35).

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    obs_types = obs.observation_types
    n = len(obs_types)
    colors = _get_colors(n)

    if figsize is None:
        figsize = (8, max(3 * n, 4))

    # Collect all (year, doy) pairs per obs_type in one pass
    xs: dict[str, list] = defaultdict(list)
    ys: dict[str, list] = defaultdict(list)
    for item in obs.iter_items():
        year = item[KEY_YEAR]
        for obs_type, observation in item[KEY_OBSERVATIONS].items():
            doy = observation.dayofyear if hasattr(observation, 'dayofyear') else int(observation)
            xs[obs_type].append(year)
            ys[obs_type].append(doy)

    with plt.rc_context(_STYLE):
        fig, axs = plt.subplots(nrows=n, sharex=True, figsize=figsize)
        axs = _as_axes_array(axs)

        for ax, obs_type, color in zip(axs, obs_types, colors):
            if not xs[obs_type]:
                continue
            ax.scatter(xs[obs_type], ys[obs_type],
                       s=point_size, alpha=alpha, color=color, linewidths=0)
            mean_val = float(np.mean(ys[obs_type]))
            ax.axhline(mean_val, color='#555555', linewidth=0.9, linestyle='--', alpha=0.8)
            label = obs_type.replace('BBCH_', 'BBCH ')
            ax.set_ylabel(label, fontsize=9)
            y_vals = ys[obs_type]
            margin = max((max(y_vals) - min(y_vals)) * 0.08, 2)
            ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

        axs[-1].set_xlabel('Year')
        fig.suptitle('Observation DOY over time', fontsize=12, y=1.01)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Mean DOY trend for one observation type
# ---------------------------------------------------------------------------

def observation_mean_trend(
    obs: Observations,
    obs_type: str,
    figsize: Tuple[float, float] = (7, 4),
    color: str = 'tomato',
) -> plt.Figure:
    """Scatter of per-year mean DOY with a fitted linear trend line.

    Args:
        obs:      Observations instance.
        obs_type: Which observation type to plot.
        figsize:  Figure dimensions.
        color:    Scatter point colour.

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    counts = obs.observation_counts()
    if obs_type not in counts or counts[obs_type] == 0:
        raise ValueError(f"No observations of type '{obs_type}' in dataset.")

    df = (
        obs.df_y
        .xs(obs_type, level=KEY_OBS_TYPE)
        .groupby(KEY_YEAR)
        .mean()
    )
    year_doy = {
        year: val.dayofyear if hasattr(val, 'dayofyear') else int(val)
        for year, val in df[KEY_OBSERVATIONS].items()
    }
    xs = np.array(sorted(year_doy.keys()), dtype=float)
    ys = np.array([year_doy[y] for y in xs.astype(int)], dtype=float)

    coeffs = np.polyfit(xs, ys, 1)
    slope, intercept = coeffs
    xs_line = np.linspace(xs.min(), xs.max(), 200)
    ys_line = np.polyval(coeffs, xs_line)

    sign = '+' if intercept >= 0 else ''
    label = f'Trend: {slope:+.3f} d/yr'

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(xs, ys, color=color, s=22, zorder=3, label='Annual mean', edgecolors='none')
        ax.plot(xs_line, ys_line, '--', color='#333333', linewidth=1.2, label=label)

        dy = max(ys) - min(ys)
        margin = max(dy * 0.15, 5)
        ax.set_ylim(max(1, min(ys) - margin), min(365, max(ys) + margin))
        ax.set_xlabel('Year')
        label_str = obs_type.replace('BBCH_', 'BBCH ')
        ax.set_ylabel(f'Mean day of year ({label_str})')
        ax.legend(fontsize=9, framealpha=0.8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spatial map helpers
# ---------------------------------------------------------------------------

def _build_obs_geodataframe(obs: Observations, obs_type: str, year: Optional[int] = None):
    """Return a GeoDataFrame of (lon, lat) points for *obs_type* [and *year*]."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required for map plots.") from exc

    df = obs.df_y
    if year is not None:
        try:
            df = df.xs(year, level=KEY_YEAR, drop_level=False)
        except KeyError:
            return gpd.GeoDataFrame(geometry=[])

    try:
        df_ot = df.xs(obs_type, level=KEY_OBS_TYPE, drop_level=True)
    except KeyError:
        return gpd.GeoDataFrame(geometry=[])

    points = []
    for idx in df_ot.index:
        coords = obs.get_location_coordinates(idx, from_index=True)
        points.append(Point(coords[KEY_LON], coords[KEY_LAT]))

    return gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')


def _setup_map_axes(ax: plt.Axes, gdf_admin, minx, miny, maxx, maxy) -> None:
    """Plot country boundaries (admin if provided, else Natural Earth) and set limits."""
    world = gdf_admin if gdf_admin is not None else _naturalearth_countries()
    if world is not None:
        clipped = world.cx[minx:maxx, miny:maxy]
        clipped.plot(ax=ax, color='#f5f5f2', edgecolor='#aaaaaa', linewidth=0.5)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_aspect('equal')
    # Remove grid for maps — geography already provides spatial context
    ax.grid(False)
    ax.set_facecolor('#d9eaf5')


# ---------------------------------------------------------------------------
# Observation map — all years
# ---------------------------------------------------------------------------

def observation_map(
    obs: Observations,
    gdf_admin=None,
    map_margin: float = 0.1,
    figsize: Tuple[float, float] = (7, 6),
    point_size: float = 3,
    alpha: float = 0.55,
) -> plt.Figure:
    """Map of every observation location, coloured by observation type.

    Country outlines are drawn automatically using geopandas' built-in
    Natural Earth dataset when *gdf_admin* is not supplied.

    Args:
        obs:        Observations instance.
        gdf_admin:  Optional :class:`geopandas.GeoDataFrame` of administrative
                    boundaries to use instead of Natural Earth.
        map_margin: Fractional padding around the bounding box (default 0.1).
        figsize:    Figure dimensions.
        point_size: Marker size.
        alpha:      Marker transparency.

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    obs_types = obs.observation_types
    colors = _get_colors(len(obs_types))
    minx, miny, maxx, maxy = _map_bounds(obs, margin=map_margin)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        _setup_map_axes(ax, gdf_admin, minx, miny, maxx, maxy)

        for obs_type, color in zip(obs_types, colors):
            gdf = _build_obs_geodataframe(obs, obs_type)
            if gdf.empty:
                continue
            label = obs_type.replace('BBCH_', 'BBCH ')
            gdf.plot(ax=ax, color=color, marker='o',
                     markersize=point_size, alpha=alpha, label=label)

        if len(obs_types) > 1:
            ax.legend(fontsize=8, markerscale=2, framealpha=0.8)

        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Observation map — single year
# ---------------------------------------------------------------------------

def observation_map_for_year(
    obs: Observations,
    year: int,
    gdf_admin=None,
    map_margin: float = 0.1,
    figsize: Tuple[float, float] = (7, 6),
    point_size: float = 4,
    alpha: float = 0.65,
) -> plt.Figure:
    """Map of observation locations for one specific year.

    Args:
        obs:        Observations instance.
        year:       Calendar year to visualise.
        gdf_admin:  Optional admin boundaries GeoDataFrame.
        map_margin: Fractional padding around the bounding box.
        figsize:    Figure dimensions.
        point_size: Marker size.
        alpha:      Marker transparency.

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    obs_types = obs.observation_types
    colors = _get_colors(len(obs_types))
    minx, miny, maxx, maxy = _map_bounds(obs, margin=map_margin)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        _setup_map_axes(ax, gdf_admin, minx, miny, maxx, maxy)

        for obs_type, color in zip(obs_types, colors):
            gdf = _build_obs_geodataframe(obs, obs_type, year=year)
            if gdf.empty:
                continue
            label = obs_type.replace('BBCH_', 'BBCH ')
            gdf.plot(ax=ax, color=color, marker='o',
                     markersize=point_size, alpha=alpha, label=label)

        if len(obs_types) > 1:
            ax.legend(fontsize=8, markerscale=2, framealpha=0.8)

        ax.set_title(str(year))
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Species-subgroup temporal occurrence
# ---------------------------------------------------------------------------

def species_subgroup_timeline(
    obs: Observations,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = '#4878cf',
    bar_height: float = 0.6,
) -> plt.Figure:
    """Broken-bar chart showing which years each (species, subgroup) was observed.

    Args:
        obs:        Observations instance.
        figsize:    Figure size.  Defaults to ``(9, 0.7 * n_groups + 1)``.
        color:      Bar fill colour.
        bar_height: Height of each bar (default 0.6).

    Returns:
        :class:`~matplotlib.figure.Figure`
    """
    occurrence: dict[tuple, set] = defaultdict(set)
    for src, _, year, species, subgroup in obs.iter_index():
        occurrence[src, species, subgroup].add(year)

    groups = sorted(occurrence.keys())
    n = len(groups)

    if figsize is None:
        figsize = (9, max(0.7 * n + 1, 3))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        # Suppress horizontal grid lines — they clash with the bars
        ax.grid(axis='x', color='#e0e0e0', linewidth=0.6)
        ax.grid(axis='y', visible=False)

        gap = bar_height / 2
        y_positions = []
        y_labels = []

        for i, key in enumerate(groups):
            src, species, subgroup = key
            years = sorted(occurrence[key])
            y_bottom = i - bar_height / 2
            ranges = _consecutive_year_ranges(years)
            ax.broken_barh(
                [(start, length) for start, length in ranges],
                (y_bottom, bar_height),
                facecolors=color,
                edgecolors='white',
                linewidth=0.5,
            )
            y_positions.append(i)
            if subgroup == 0:
                label = f'{src} · {species}'
            else:
                label = f'{src} · {species} / {subgroup}'
            y_labels.append(label)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_ylim(-gap - 0.2, n - 1 + gap + 0.2)
        ax.set_xlabel('Year')
        ax.set_title('Species / subgroup temporal coverage')
        ax.margins(x=0.02)
        fig.tight_layout()
    return fig


def _consecutive_year_ranges(years: list[int]) -> list[tuple[int, int]]:
    """Convert a sorted list of years into (start, length) ranges for broken_barh."""
    if not years:
        return []
    ranges = []
    start = years[0]
    prev = years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
        else:
            ranges.append((start, prev - start + 1))
            start = prev = y
    ranges.append((start, prev - start + 1))
    return ranges


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    stem: str,
    formats: Sequence[str] = ('png',),
    dpi: int = 150,
) -> None:
    """Save *fig* to *path* under the filename *stem* in each of *formats*.

    Args:
        fig:     Figure to save.
        path:    Output directory (created if absent).
        stem:    Filename without extension (e.g. ``'observation_map'``).
        formats: Iterable of file-format strings understood by matplotlib
                 (e.g. ``('png', 'svg', 'pdf')``).  Default ``('png',)``.
        dpi:     Resolution for raster formats (default 150).
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(
            out / f'{stem}.{fmt}',
            bbox_inches='tight',
            dpi=dpi if fmt != 'svg' else None,
        )
