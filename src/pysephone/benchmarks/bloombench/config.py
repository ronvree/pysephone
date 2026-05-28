"""
Static configuration for BloomBench.

Constants live here as plain module-level values (matching the codebase
precedent in ``pysephone.constants`` / ``pysephone.paths``).  Path and
device resolution are exposed as lazy functions so importing this module
has no side effects (no ``mkdir``, no ``torch.cuda.is_available()``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pysephone.paths import get_data_root


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

#: Pairs of (dataset name, target observation key) included in BloomBench.
#:
#: The 18 entries cover the three families described in the paper plus the
#: extended USA-NPN tier (small datasets dropped at load time by
#: :data:`MIN_DATASET_SAMPLES`).
DATASETS_REQUESTED: List[Tuple[str, str]] = [
    # PEP725 (Tier A, Hazel + Blackthorn dropped, Cherry de-duplicated against GMU)
    ('PEP725_Apple',          'BBCH_60'),
    ('PEP725_Pear',           'BBCH_60'),
    ('PEP725_Peach',          'BBCH_60'),
    ('PEP725_Almond',         'BBCH_60'),
    ('PEP725_Apricot',        'BBCH_60'),
    ('PEP725_Plum',           'BBCH_60'),
    ('PEP725_Cherry_NoGMU',   'BBCH_60'),
    # GMU Cherry (Tier A — Japan composite replaced by Y / S splits)
    ('GMU_Cherry_Japan_Y',       'gmu_0'),  # Cerasus yedoensis
    ('GMU_Cherry_Japan_S',       'gmu_0'),  # Cerasus sargentii
    ('GMU_Cherry_Switzerland',   'gmu_1'),
    ('GMU_Cherry_South_Korea',   'gmu_2'),
    # USA-NPN matched species (Tier B)
    ('USA_NPN_Apple',    'NPN_501'),
    ('USA_NPN_Pear',     'NPN_501'),
    ('USA_NPN_Peach',    'NPN_501'),
    ('USA_NPN_Almond',   'NPN_501'),
    ('USA_NPN_Apricot',  'NPN_501'),
    ('USA_NPN_Plum',     'NPN_501'),
    ('USA_NPN_Cherry',   'NPN_501'),
]

#: Datasets with fewer than this many samples after load are skipped.
#: Primarily relevant for the smaller USA-NPN tier; the PEP725 / GMU Cherry
#: datasets all exceed this threshold comfortably.
MIN_DATASET_SAMPLES: int = 100


# ---------------------------------------------------------------------------
# Climate provider
# ---------------------------------------------------------------------------

#: AgERA5 variable keys consumed by the benchmark.  Temperature is in Kelvin;
#: torch models receive per-dataset feature statistics computed at fit time,
#: so the unit difference vs. OpenMeteo does not propagate as a hidden bug.
FEATURE_KEYS: Tuple[str, ...] = ('Temperature_Air_2m_Mean_24h',)


# ---------------------------------------------------------------------------
# Season window + temporal split
# ---------------------------------------------------------------------------

SEASON_START: str = '10-01'
SEASON_LENGTH: int = 365

#: Fraction of years (sorted ascending) assigned to the train fold.
SPLIT_SIZE: float = 0.75


# ---------------------------------------------------------------------------
# HPO budgets
# ---------------------------------------------------------------------------

HPO_N_ITER_TREES: int = 20
HPO_N_TRIALS_TORCH: int = 15
HPO_CV_FOLDS: int = 5
HPO_VAL_FRACTION: float = 0.2


# ---------------------------------------------------------------------------
# Torch fit budget (used by both HPO trials and final fits)
# ---------------------------------------------------------------------------

#: Training-loop kwargs shared by every torch model in the benchmark.
#: ``device`` is filled in lazily by :func:`torch_train_kwargs`.
_TORCH_TRAIN_KWARGS_BASE: Dict[str, Any] = dict(
    num_epochs=200,
    batch_size=32,
    val_period=10,
    early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=1e-3,
)


def torch_device() -> str:
    """Return ``'cuda'`` if a CUDA device is visible at call time, else ``'cpu'``.

    Evaluated lazily so ``CUDA_VISIBLE_DEVICES`` set after import is honoured.
    """
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def torch_train_kwargs() -> Dict[str, Any]:
    """Return the torch training kwargs dict with ``device`` filled in."""
    return dict(_TORCH_TRAIN_KWARGS_BASE, device=torch_device())


# ---------------------------------------------------------------------------
# Run identification
# ---------------------------------------------------------------------------

#: Default prefix for the model / evaluation cache keys.
RUN_PREFIX: str = 'bb_ext'


def run_name(dataset: str, model_key: str, seed: int, *, run_prefix: str = RUN_PREFIX) -> str:
    """Build a stable identifier for a (dataset, model, seed) run.

    The name is used as the directory key for both the saved model
    (:meth:`BaseModel.save`) and the saved evaluation
    (:meth:`SingleTargetRegression.save`).
    """
    return f'{run_prefix}_{dataset}_{model_key}_seed{seed}'


# ---------------------------------------------------------------------------
# Output paths (lazy — no mkdir at import time)
# ---------------------------------------------------------------------------

def hp_cache_dir(root: Optional[Path] = None) -> Path:
    """Directory holding ``<dataset>_<model>.json`` HP cache files.

    The path is namespaced by climate provider so a future switch back to
    OpenMeteo would not silently mix incompatible best-params files.  No
    directory is created here; callers create it on write.
    """
    return (root or get_data_root()) / 'outputs' / 'bloombench' / 'hyperparams' / 'agera5'


def runs_dir(root: Optional[Path] = None) -> Path:
    """Directory holding ``runs/<run_prefix>/{results.csv, datasets.json}``."""
    return (root or get_data_root()) / 'outputs' / 'bloombench' / 'runs'
