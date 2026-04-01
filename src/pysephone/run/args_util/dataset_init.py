"""
Argparse helpers for dataset loading and splitting.
"""

import argparse
from typing import Optional, Tuple

import sklearn.model_selection

from pysephone.dataset.dataset import Dataset
from pysephone.run.args_util import ExperimentConfigException


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def configure_argparser_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Key passed to Dataset.load().',
    )
    return parser


def init_dataset_from_args(args: argparse.Namespace, calendar=None, feature_providers=None) -> Dataset:
    """Load a dataset by name.

    Args:
        args:              Parsed namespace (must include ``dataset_name``).
        calendar:          Optional calendar to attach.
        feature_providers: Optional list of feature providers to attach.
    """
    return Dataset.load(args.dataset_name, calendar=calendar, feature_providers=feature_providers)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def configure_argparser_dataset_split(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset-split arguments to *parser*.

    Spatial split (at most one of):
      ``--split_locations gridded_random`` with ``--split_locations_grid_size``
      and ``--split_locations_size``
      ``--split_at_latitude <float>`` / ``--split_at_latitude_inverted <float>``
      ``--split_at_longitude <float>`` / ``--split_at_longitude_inverted <float>``

    Temporal split (at most one of):
      ``--split_years random`` with ``--split_years_size``
      ``--split_years cutoff`` with ``--split_years_cutoff_year`` or
      ``--split_years_size``
    """
    # Spatial
    parser.add_argument(
        '--split_locations',
        type=str,
        default=None,
        choices=['gridded_random'],
        help='Spatial split method.',
    )
    parser.add_argument(
        '--split_locations_size',
        type=float,
        default=0.75,
        help='Fraction of grid cells assigned to train (default: 0.75).',
    )
    parser.add_argument(
        '--split_locations_grid_size',
        type=float,
        default=1.0,
        help='Grid cell size in degrees (default: 1.0).',
    )
    parser.add_argument(
        '--split_at_latitude',
        type=float,
        default=None,
        help='Train: north of this latitude.  Test: south.',
    )
    parser.add_argument(
        '--split_at_latitude_inverted',
        type=float,
        default=None,
        help='Train: south of this latitude.  Test: north.',
    )
    parser.add_argument(
        '--split_at_longitude',
        type=float,
        default=None,
        help='Train: west of this longitude.  Test: east.',
    )
    parser.add_argument(
        '--split_at_longitude_inverted',
        type=float,
        default=None,
        help='Train: east of this longitude.  Test: west.',
    )

    # Temporal
    parser.add_argument(
        '--split_years',
        type=str,
        default=None,
        choices=['random', 'cutoff'],
        help='Temporal split method.',
    )
    parser.add_argument(
        '--split_years_size',
        type=float,
        default=0.75,
        help='Fraction of years assigned to train (default: 0.75).',
    )
    parser.add_argument(
        '--split_years_cutoff_year',
        type=int,
        default=None,
        help='Explicit cutoff year: train < year, test >= year.',
    )

    # Shared seed
    parser.add_argument(
        '--seed_data_split',
        type=int,
        default=None,
        help='Random seed for the data split.  Falls back to --seed if omitted.',
    )
    return parser


def split_dataset_from_args(
    dataset: Dataset,
    args: argparse.Namespace,
) -> Tuple[Dataset, Dataset]:
    """Apply spatial and/or temporal splits according to *args*.

    Args:
        dataset: Full dataset to split.
        args:    Parsed namespace from a parser configured with
                 :func:`configure_argparser_dataset_split`.

    Returns:
        ``(dataset_train, dataset_test)``

    Raises:
        ExperimentConfigException: On invalid or conflicting split arguments.
    """
    _validate_split_args(args)

    seed: Optional[int] = (
        args.seed_data_split
        if args.seed_data_split is not None
        else getattr(args, 'seed', None)
    )

    dataset_trn, dataset_tst = dataset, dataset

    # ------------------------------------------------------------------
    # Spatial split
    # ------------------------------------------------------------------

    if args.split_locations is not None:
        match args.split_locations:
            case 'gridded_random':
                grid_size = (args.split_locations_grid_size, args.split_locations_grid_size)
                dataset_trn, dataset_tst, _ = dataset.split_by_grid(
                    grid_size=grid_size,
                    split_size=args.split_locations_size,
                    shuffle=True,
                    random_state=seed,
                )
            case _:
                raise ExperimentConfigException(
                    f'Unknown spatial split method: {args.split_locations!r}'
                )

    if args.split_at_latitude is not None:
        dataset_trn, _, _ = dataset_trn.split_by_lat_border(args.split_at_latitude)
        _, dataset_tst, _ = dataset_tst.split_by_lat_border(args.split_at_latitude)

    if args.split_at_latitude_inverted is not None:
        _, dataset_trn, _ = dataset_trn.split_by_lat_border(args.split_at_latitude_inverted)
        dataset_tst, _, _ = dataset_tst.split_by_lat_border(args.split_at_latitude_inverted)

    if args.split_at_longitude is not None:
        dataset_trn, _, _ = dataset_trn.split_by_lon_border(args.split_at_longitude)
        _, dataset_tst, _ = dataset_tst.split_by_lon_border(args.split_at_longitude)

    if args.split_at_longitude_inverted is not None:
        _, dataset_trn, _ = dataset_trn.split_by_lon_border(args.split_at_longitude_inverted)
        dataset_tst, _, _ = dataset_tst.split_by_lon_border(args.split_at_longitude_inverted)

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------

    if args.split_years is not None:
        years = dataset.years
        years_sorted = sorted(set(years))

        match args.split_years:
            case 'random':
                years_trn, years_tst = sklearn.model_selection.train_test_split(
                    years_sorted,
                    train_size=args.split_years_size,
                    shuffle=True,
                    random_state=seed,
                )

            case 'cutoff':
                if args.split_years_cutoff_year is not None:
                    cutoff = args.split_years_cutoff_year
                else:
                    ix = int(len(years_sorted) * args.split_years_size)
                    cutoff = years_sorted[ix]
                years_trn = [y for y in years_sorted if y < cutoff]
                years_tst = [y for y in years_sorted if y >= cutoff]

            case _:
                raise ExperimentConfigException(
                    f'Unknown temporal split method: {args.split_years!r}'
                )

        dataset_trn = dataset_trn.select_years(years_trn)
        dataset_tst = dataset_tst.select_years(years_tst)

    return dataset_trn, dataset_tst


def _validate_split_args(args: argparse.Namespace) -> None:
    if args.split_at_latitude is not None and args.split_at_latitude_inverted is not None:
        raise ExperimentConfigException(
            '--split_at_latitude and --split_at_latitude_inverted are mutually exclusive.'
        )
    if args.split_at_longitude is not None and args.split_at_longitude_inverted is not None:
        raise ExperimentConfigException(
            '--split_at_longitude and --split_at_longitude_inverted are mutually exclusive.'
        )
    if args.split_years == 'cutoff':
        if args.split_years_cutoff_year is None and args.split_years_size is None:
            raise ExperimentConfigException(
                "split_years='cutoff' requires --split_years_cutoff_year or --split_years_size."
            )
