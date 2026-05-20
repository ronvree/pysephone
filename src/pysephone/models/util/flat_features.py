"""Flat per-season feature extraction for non-sequence models.

Helpers shared by phenology models that operate on a flattened season-level
feature vector (e.g. Random Forest, XGBoost) rather than a per-day sequence.
Each sample's daily features are concatenated across keys into a single
``(sum_k T_k,)`` vector; the regression target is the within-season day
index of the observed event.

No feature scaling is applied — tree-based learners split on thresholds, so
any monotone rescale produces an equivalent model.  Add normalisation in the
caller if a scale-sensitive learner is plugged in.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pysephone.constants import KEY_FEATURES


def extract_flat_features(
    sample: Dict[str, Any],
    data_keys: List[str],
) -> np.ndarray:
    """Return a flat ``(sum_k T_k,)`` feature vector for one sample."""
    parts = [np.asarray(sample[KEY_FEATURES][k], dtype=float) for k in data_keys]
    return np.concatenate(parts)


def build_flat_feature_matrix(
    samples: List[Dict[str, Any]],
    data_keys: List[str],
) -> np.ndarray:
    rows = [extract_flat_features(s, data_keys) for s in samples]
    return np.vstack(rows)


def build_day_index_targets(
    samples: List[Dict[str, Any]],
    target_fn: Callable[[Dict[str, Any]], Any],
) -> np.ndarray:
    ys = []
    for s in samples:
        target_dt = np.datetime64(target_fn(s), 'D')
        start_dt  = np.datetime64(s['season_start'], 'D')
        ys.append(int((target_dt - start_dt) / np.timedelta64(1, 'D')))
    return np.array(ys, dtype=float)
