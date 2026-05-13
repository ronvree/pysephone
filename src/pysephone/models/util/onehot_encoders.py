"""One-hot encoders for categorical sample identity.

These encoders map a batch dict (the ``xs`` produced by
:meth:`BaseTorchModel.collate_fn`) to either:

* an integer index array ``(B,)`` together with a boolean valid mask ``(B,)``
  — for consumers that use ``nn.Embedding`` or other index-based lookups, and
* a one-hot ``(B, N)`` array — for consumers that prefer dense one-hot
  encodings (concatenation into an LSTM head, an MLP input, etc.).

Both batched dicts (``list`` values under each id field) and single-sample
dicts (any non-list value, e.g. straight from ``dataset[ix]``) are accepted;
the convention is that a ``list`` means "batch" and anything else means
"single sample".  A single sample is treated as a batch of one, so the
output shape is always ``(B, N)`` / ``(B,)`` with ``B = 1`` for the scalar
case.

All outputs are :class:`numpy.ndarray`.  Downstream models cast to
:class:`torch.Tensor` themselves (and move to the right device) since the
encoder is framework-agnostic state — it owns the key→index table, not any
learnable parameters.

The ``unknown`` policy controls what happens when a batch sample's key is not
in the fitted set:

* ``'zero'``  — the row is marked invalid (``valid[i] = False``); ``indices``
  returns a safe sentinel ``0`` for that row, and ``one_hot`` returns an
  all-zero row.  This is the partial-pooling default: an unknown sample
  contributes no signal and the model falls back to whatever its zero-input
  behaviour is.
* ``'error'`` — raise :class:`KeyError` on the first unseen key.

The three concrete encoders provided here are:

* :class:`OneHotSpeciesEncoder`         — one-hot over ``(src, species_id)``.
* :class:`OneHotLocationEncoder`        — one-hot over ``(src, loc_id)``.
* :class:`OneHotSpeciesSubgroupEncoder` — one-hot over the joint
  ``(src, species_id, subgroup_id)`` triple.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_LOC_ID,
    KEY_SPECIES_ID,
    KEY_SUBGROUP_ID,
)


SpeciesKey         = Tuple[str, int]
LocationKey        = Tuple[str, str]
SpeciesSubgroupKey = Tuple[str, int, int]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OneHotEncoder(ABC):
    """Categorical identity → one-hot / index lookup, returning NumPy arrays.

    Args:
        keys:    Ordered iterable of distinct hashable keys.  The one-hot
                 dimension follows this order; duplicates raise
                 :class:`ValueError`.
        unknown: Behaviour for batch keys not in *keys* — either ``'zero'``
                 (mark invalid, emit zero row) or ``'error'`` (raise).
    """

    def __init__(
        self,
        keys: Iterable[Hashable],
        unknown: str = 'zero',
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        keys_list: List[Hashable] = list(keys)
        if not keys_list:
            raise ValueError('keys must be non-empty')

        index: Dict[Hashable, int] = {}
        for i, k in enumerate(keys_list):
            if k in index:
                raise ValueError(f'duplicate key {k!r} at positions {index[k]} and {i}')
            index[k] = i

        self._keys: Tuple[Hashable, ...] = tuple(keys_list)
        self._index: Dict[Hashable, int] = index
        self._unknown: str = unknown

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def num_categories(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> Tuple[Hashable, ...]:
        return self._keys

    @property
    def unknown(self) -> str:
        return self._unknown

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _as_batch(v: Any) -> Sequence[Any]:
        """Wrap a single-sample field value in a 1-element list.

        Convention: a ``list`` indicates a batch (and is passed through
        unchanged); anything else is treated as a single sample and wrapped
        in ``[v]``.  This matches the output of
        :meth:`BaseTorchModel.collate_fn`, which produces plain lists for the
        id-style fields the encoders read.
        """
        if isinstance(v, list):
            return v
        return [v]

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_keys(self, xs: Dict[str, Any]) -> List[Hashable]:
        """Pull the per-sample key tuple out of a batch dict.

        Implementations should route each field value through
        :meth:`_as_batch` so that both batched and single-sample inputs are
        accepted.
        """

    # ------------------------------------------------------------------
    # Public encoding
    # ------------------------------------------------------------------

    def indices(self, xs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Look up batch keys.

        Args:
            xs: Collated batch dict.

        Returns:
            ``(idx, valid)`` where:

            * ``idx`` is an ``int64`` array of shape ``(B,)``.  Invalid rows
              (key not in fitted set, only possible when ``unknown='zero'``)
              hold the sentinel ``0`` so the array is safe to feed into an
              embedding lookup — pair with *valid* to mask the result.
            * ``valid`` is a ``bool`` array of shape ``(B,)``; ``True`` iff
              the row's key was known at fit time.
        """
        batch_keys = self._extract_keys(xs)
        B = len(batch_keys)
        idx   = np.zeros(B, dtype=np.int64)
        valid = np.zeros(B, dtype=bool)
        for i, k in enumerate(batch_keys):
            row = self._index.get(k)
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Key {k!r} not in fitted encoder '
                        f'(known: {list(self._index.keys())[:5]}...)'
                    )
                continue   # idx[i] stays 0, valid[i] stays False
            idx[i]   = row
            valid[i] = True
        return idx, valid

    def one_hot(self, xs: Dict[str, Any]) -> np.ndarray:
        """Return a ``(B, N)`` one-hot ``float32`` array.

        Unknown rows (with ``unknown='zero'``) become all-zero rows.
        """
        idx, valid = self.indices(xs)
        out = np.zeros((idx.size, self.num_categories), dtype=np.float32)
        if valid.any():
            rows = np.nonzero(valid)[0]
            out[rows, idx[rows]] = 1.0
        return out


# ---------------------------------------------------------------------------
# Concrete: (src, species_id)
# ---------------------------------------------------------------------------

class OneHotSpeciesEncoder(OneHotEncoder):
    """One-hot encoder over distinct ``(src, species_id)`` pairs."""

    def __init__(
        self,
        species_keys: Sequence[SpeciesKey],
        unknown: str = 'zero',
    ) -> None:
        normalised: List[SpeciesKey] = [
            (str(s), int(sid)) for s, sid in species_keys
        ]
        super().__init__(keys=normalised, unknown=unknown)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        complete: bool = True,
    ) -> 'OneHotSpeciesEncoder':
        """Build from the unique ``(src, species_id)`` pairs in *dataset*.

        Args:
            dataset:  Dataset exposing :attr:`species` (post-split) and
                      :attr:`species_complete` (pre-split) properties.
            unknown:  Behaviour for unseen species at encode time.
            complete: If True, draw keys from ``dataset.species_complete`` so
                      the encoder covers every species in the original
                      dataset (before train/val/test splits or filters).
                      Useful when you fit on a training subset but want the
                      validation/test species to also produce valid one-hots.
        """
        species = dataset.species_complete if complete else dataset.species
        return cls(species_keys=sorted(species), unknown=unknown)

    def _extract_keys(self, xs: Dict[str, Any]) -> List[Hashable]:
        srcs = self._as_batch(xs[KEY_DATA_SOURCE])
        sids = self._as_batch(xs[KEY_SPECIES_ID])
        return [(str(s), int(sid)) for s, sid in zip(srcs, sids)]


# ---------------------------------------------------------------------------
# Concrete: (src, loc_id)
# ---------------------------------------------------------------------------

class OneHotLocationEncoder(OneHotEncoder):
    """One-hot encoder over distinct ``(src, loc_id)`` pairs.

    ``loc_id`` is coerced to ``str`` for hashing, since some sources use
    integer station IDs and others use alphanumeric codes.
    """

    def __init__(
        self,
        location_keys: Sequence[Tuple[str, Any]],
        unknown: str = 'zero',
    ) -> None:
        normalised: List[LocationKey] = [
            (str(s), str(lid)) for s, lid in location_keys
        ]
        super().__init__(keys=normalised, unknown=unknown)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        complete: bool = True,
    ) -> 'OneHotLocationEncoder':
        """Build from the unique ``(src, loc_id)`` pairs in *dataset*.

        Args:
            dataset:  Dataset exposing :attr:`locations` (post-split) and
                      :attr:`locations_complete` (pre-split) properties.
            unknown:  Behaviour for unseen locations at encode time.
            complete: If True, draw keys from ``dataset.locations_complete``
                      so the encoder covers every location in the original
                      dataset (before train/val/test splits or filters).
        """
        locations = dataset.locations_complete if complete else dataset.locations
        return cls(location_keys=sorted(locations), unknown=unknown)

    def _extract_keys(self, xs: Dict[str, Any]) -> List[Hashable]:
        srcs = self._as_batch(xs[KEY_DATA_SOURCE])
        lids = self._as_batch(xs[KEY_LOC_ID])
        return [(str(s), str(lid)) for s, lid in zip(srcs, lids)]


# ---------------------------------------------------------------------------
# Concrete: (src, species_id, subgroup_id)
# ---------------------------------------------------------------------------

class OneHotSpeciesSubgroupEncoder(OneHotEncoder):
    """One-hot encoder over distinct ``(src, species_id, subgroup_id)`` triples.

    The ``subgroup_id`` field (see :data:`KEY_SUBGROUP_ID`) is part of the
    dataset's primary index and is used by datasets like PEP725 to record a
    sub-species grouping (e.g. cultivar).
    """

    def __init__(
        self,
        species_subgroup_keys: Sequence[Tuple[str, int, int]],
        unknown: str = 'zero',
    ) -> None:
        normalised: List[SpeciesSubgroupKey] = [
            (str(s), int(sid), int(sg)) for s, sid, sg in species_subgroup_keys
        ]
        super().__init__(keys=normalised, unknown=unknown)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        unknown: str = 'zero',
        complete: bool = True,
    ) -> 'OneHotSpeciesSubgroupEncoder':
        """Build from the unique ``(src, species_id, subgroup_id)`` triples.

        Args:
            dataset:  Dataset exposing :attr:`species_subgroups` (post-split)
                      and :attr:`species_subgroups_complete` (pre-split)
                      properties.
            unknown:  Behaviour for unseen triples at encode time.
            complete: If True, draw keys from
                      ``dataset.species_subgroups_complete`` so the encoder
                      covers every (species, subgroup) in the original
                      dataset (before train/val/test splits or filters).
        """
        triples = (
            dataset.species_subgroups_complete if complete
            else dataset.species_subgroups
        )
        return cls(species_subgroup_keys=sorted(triples), unknown=unknown)

    def _extract_keys(self, xs: Dict[str, Any]) -> List[Hashable]:
        srcs = self._as_batch(xs[KEY_DATA_SOURCE])
        sids = self._as_batch(xs[KEY_SPECIES_ID])
        sgs  = self._as_batch(xs[KEY_SUBGROUP_ID])
        return [
            (str(s), int(sid), int(sg)) for s, sid, sg in zip(srcs, sids, sgs)
        ]
