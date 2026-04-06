"""
Fit a phenology model and evaluate it on a train/test split.

Usage::

    python -m pysephone.run.fit_eval \\
        --run_name my_run \\
        --target BBCH_60 \\
        --dataset_name my_dataset \\
        --split_years cutoff --split_years_cutoff_year 2015 \\
        --model_cls_path pysephone.models.pvtt.PVTTModel \\
        --threshold_pvtt 800.0 --threshold_vern 30.0 \\
        --t_base 1.0 --t_limit 32.0 --t_upper 40.0 \\
        --p_base 7.0 --p_saturation 17.0

Flow:
    1. Parse arguments (two-pass: model class is resolved before its args are registered)
    2. Load the dataset
    3. Split into train / test
    4. Fit the model
    5. Evaluate on both splits
    6. Save results and print metrics
"""

from __future__ import annotations

import argparse
import sys

from pysephone.constants import KEY_OBSERVATIONS
from pysephone.run.args_util.dataset_init import (
    configure_argparser_dataset,
    configure_argparser_dataset_split,
    init_dataset_from_args,
    split_dataset_from_args,
)
from pysephone.run.args_util.eval import configure_argparser_eval, evaluate_from_args
from pysephone.run.args_util.model import configure_argparser_model, get_model_cls_from_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Fit a phenology model and evaluate it on a train/test split.',
    )

    # Resolve the model class first (two-pass) so it can register its own args.
    configure_argparser_model(parser)
    known, _ = parser.parse_known_args()
    model_cls = get_model_cls_from_args(known)

    configure_argparser_dataset(parser)
    configure_argparser_dataset_split(parser)
    configure_argparser_eval(parser)
    model_cls.configure_argparser(parser)

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Global random seed (fallback for data splits).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print progress updates during loading, fitting, and evaluation.',
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    verbose = args.verbose

    model_cls = get_model_cls_from_args(args)
    model_args = model_cls.model_args_from_namespace(args)
    if model_args.model_name is None:
        model_args.model_name = args.run_name

    target_key = args.target

    def target_fn(s):
        return s[KEY_OBSERVATIONS][target_key]

    # Load
    if verbose:
        print(f"Loading dataset '{args.dataset_name}'...")
    dataset = init_dataset_from_args(args)
    if verbose:
        print(f"Loaded {len(dataset)} samples.")

    # Split
    ds_trn, ds_tst = split_dataset_from_args(dataset, args)
    if verbose:
        print(f"Split: {len(ds_trn)} train, {len(ds_tst)} test.")

    # Fit
    if verbose:
        print(f"Fitting {model_cls.__name__}...")
    model, _ = model_cls.fit_from_args(target_fn=target_fn, dataset=ds_trn, model_args=model_args)
    if verbose:
        params_str = str(getattr(model, 'params', None))
        print(f"Done. Params: {params_str}")

    # Evaluate
    if verbose:
        print("Evaluating...")
    result = evaluate_from_args(model, ds_trn, ds_tst, args)

    # Save and report
    result.save()

    metrics = result.compute_metrics()
    print('\nTrain')
    print(metrics['train'])
    print('\nTest')
    print(metrics['test'])

    model.save(args.run_name)

    return 0


if __name__ == '__main__':
    sys.exit(main())
