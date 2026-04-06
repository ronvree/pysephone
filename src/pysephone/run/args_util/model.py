"""
Argparse helpers for model class selection.
"""

import argparse
import importlib

from pysephone.models.base import BaseModel


def configure_argparser_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model-selection arguments to *parser*.

    Added arguments:
      ``--model_cls_path``: dotted import path to the model class,
        e.g. ``pysephone.models.pvtt.PVTTModel``.
      ``--model_name``: optional run identifier (defaults to the class name).
    """
    parser.add_argument(
        '--model_cls_path',
        type=str,
        required=True,
        help=(
            'Dotted import path to the model class, '
            'e.g. pysephone.models.pvtt.PVTTModel'
        ),
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Optional name for this model run (used when saving). Defaults to the class name.',
    )
    return parser


def get_model_cls_from_path(path: str) -> type:
    """Import and return the class at the dotted *path*.

    Args:
        path: Fully-qualified class path, e.g. ``'pysephone.models.pvtt.PVTTModel'``.

    Returns:
        The class object.

    Raises:
        ImportError:   If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_path, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_model_cls_from_args(args: argparse.Namespace) -> type:
    """Resolve the model class from ``args.model_cls_path``."""
    return get_model_cls_from_path(args.model_cls_path)
