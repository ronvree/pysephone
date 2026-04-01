"""
Argparse helpers for the evaluation step.
"""

import argparse

from pysephone.constants import KEY_OBSERVATIONS
from pysephone.dataset.dataset import Dataset
from pysephone.evaluation.regression import SingleTargetRegression
from pysephone.models.base import BaseModel


def configure_argparser_eval(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='Identifier for this evaluation run (used when saving results).',
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help="Observation key to predict, e.g. 'BBCH_60'.",
    )
    return parser


def evaluate_from_args(
    model: BaseModel,
    dataset_trn: Dataset,
    dataset_tst: Dataset,
    args: argparse.Namespace,
) -> SingleTargetRegression:
    """Run evaluation and return a :class:`SingleTargetRegression` result.

    Args:
        model:       Fitted model.
        dataset_trn: Training dataset.
        dataset_tst: Test dataset.
        args:        Parsed namespace (must include ``run_name`` and ``target``).

    Returns:
        :class:`~pysephone.evaluation.regression.SingleTargetRegression`
    """
    target_key = args.target

    def target_fn(s):
        return s[KEY_OBSERVATIONS][target_key]

    return SingleTargetRegression.run(
        model=model,
        dataset_train=dataset_trn,
        dataset_test=dataset_tst,
        target_fn=target_fn,
        run_name=args.run_name,
    )
