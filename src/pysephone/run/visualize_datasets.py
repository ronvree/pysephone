"""
Visualise all registered datasets and save the figures to outputs/.

Usage::

    python -m pysephone.run.visualize_datasets [--formats png svg] [--dpi 150]

For every dataset key in the registry the script:

1. Loads the dataset (skips and reports if loading fails, e.g. data not present).
2. Generates the following figures:

   * ``doy_histograms``       – DOY distribution per observation type
   * ``doy_over_time``        – scatter of DOY vs year per observation type
   * ``mean_trend_<obs_type>``– per-year mean with linear trend, one figure per type
   * ``species_timeline``     – broken-bar temporal coverage per (species, subgroup)
   * ``map``                  – spatial map of all locations (requires geopandas)

3. Saves each figure under::

       <repo_root>/outputs/visualize_datasets/<dataset_key>/<stem>.<fmt>

All figures are closed after saving to keep memory usage low.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt

from pysephone.dataset.registry import REGISTRY
from pysephone.dataset.dataset import Dataset
from pysephone.paths import get_data_root
from pysephone.visualize.dataset import (
    observation_doy_histograms,
    observation_doy_over_time,
    observation_mean_trend,
    observation_map,
    species_subgroup_timeline,
    save_figure,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset visualisation
# ---------------------------------------------------------------------------

def _visualize_dataset(key: str, out_dir: Path, formats: tuple[str, ...], dpi: int) -> None:
    """Load *key*, generate all figures, and save to *out_dir*."""
    log.info("Loading dataset: %s", key)
    dataset = Dataset.load(key)
    obs = dataset.observations

    if len(obs) == 0:
        log.warning("  [%s] No observations — skipping.", key)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("  %d observations, saving figures to %s", len(obs), out_dir)

    # DOY histograms
    try:
        fig = observation_doy_histograms(obs)
        save_figure(fig, out_dir, 'doy_histograms', formats=formats, dpi=dpi)
        plt.close(fig)
    except Exception:
        log.warning("  [%s] doy_histograms failed:\n%s", key, traceback.format_exc())

    # DOY over time
    try:
        fig = observation_doy_over_time(obs)
        save_figure(fig, out_dir, 'doy_over_time', formats=formats, dpi=dpi)
        plt.close(fig)
    except Exception:
        log.warning("  [%s] doy_over_time failed:\n%s", key, traceback.format_exc())

    # Mean trend — one figure per observation type
    for obs_type in obs.observation_types:
        stem = f'mean_trend_{obs_type}'
        try:
            fig = observation_mean_trend(obs, obs_type)
            save_figure(fig, out_dir, stem, formats=formats, dpi=dpi)
            plt.close(fig)
        except Exception:
            log.warning("  [%s] %s failed:\n%s", key, stem, traceback.format_exc())

    # Species/subgroup timeline
    try:
        fig = species_subgroup_timeline(obs)
        save_figure(fig, out_dir, 'species_timeline', formats=formats, dpi=dpi)
        plt.close(fig)
    except Exception:
        log.warning("  [%s] species_timeline failed:\n%s", key, traceback.format_exc())

    # Spatial map (optional — requires geopandas)
    try:
        fig = observation_map(obs)
        save_figure(fig, out_dir, 'map', formats=formats, dpi=dpi)
        plt.close(fig)
    except ImportError:
        log.info("  [%s] map skipped (geopandas not installed).", key)
    except Exception:
        log.warning("  [%s] map failed:\n%s", key, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Visualise all registered datasets and save figures to outputs/.'
    )
    parser.add_argument(
        '--formats', nargs='+', default=['png'], metavar='FMT',
        help='Output format(s) understood by matplotlib, e.g. png svg pdf (default: png).',
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for raster formats (default: 150).',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None, metavar='KEY',
        help='Subset of dataset keys to process.  Default: all registered datasets.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG logging.',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s  %(message)s',
    )

    keys = args.datasets if args.datasets is not None else sorted(REGISTRY)
    formats = tuple(args.formats)
    base_out = get_data_root() / 'outputs' / 'visualize_datasets'

    failed: list[str] = []

    for key in keys:
        if key not in REGISTRY:
            log.error("Unknown dataset key: %s", key)
            failed.append(key)
            continue

        out_dir = base_out / key
        try:
            _visualize_dataset(key, out_dir, formats=formats, dpi=args.dpi)
        except Exception:
            log.error("[%s] Failed to load or process:\n%s", key, traceback.format_exc())
            failed.append(key)

    if failed:
        log.warning("The following datasets could not be visualised: %s", failed)
        return 1

    log.info("Done. Figures saved under %s", base_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
