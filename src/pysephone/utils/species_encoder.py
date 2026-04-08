from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Species are identified by (source_id, species_id)
Species = Tuple[str, int]


class SpeciesOneHotEncoder:
    """One-hot encoder for species identified by ``(source_id, species_id)`` pairs.

    Usage::

        encoder = SpeciesOneHotEncoder()
        encoder.fit([('pep725', 333), ('pep725', 59), ('gmu', 1)])

        encoder.transform(('pep725', 333))          # shape (3,)
        encoder.transform_batch([('pep725', 333),
                                 ('gmu', 1)])       # shape (2, 3)
    """

    def __init__(self) -> None:
        self._index: dict[Species, int] = {}

    def fit(self, species: Sequence[Species]) -> 'SpeciesOneHotEncoder':
        """Build the species → index mapping from a collection of species keys.

        Duplicate entries are ignored.  Order is determined by first occurrence.

        Args:
            species: Iterable of ``(source_id, species_id)`` tuples.

        Returns:
            ``self``, for chaining.
        """
        self._index = {}
        for s in species:
            key = (str(s[0]), int(s[1]))
            if key not in self._index:
                self._index[key] = len(self._index)
        return self

    @property
    def n_species(self) -> int:
        """Number of distinct species known to the encoder."""
        return len(self._index)

    @property
    def species(self) -> List[Species]:
        """Ordered list of species in index order."""
        return [s for s, _ in sorted(self._index.items(), key=lambda kv: kv[1])]

    def transform(self, species: Species) -> np.ndarray:
        """Return the one-hot vector for a single species.

        Args:
            species: ``(source_id, species_id)`` tuple.

        Returns:
            1-D boolean array of shape ``(n_species,)``.

        Raises:
            KeyError: If *species* was not seen during :meth:`fit`.
            RuntimeError: If the encoder has not been fitted yet.
        """
        if not self._index:
            raise RuntimeError('Encoder has not been fitted. Call fit() first.')
        key = (str(species[0]), int(species[1]))
        if key not in self._index:
            raise KeyError(f'Unknown species {key!r}. Known: {self.species}')
        vec = np.zeros(self.n_species, dtype=bool)
        vec[self._index[key]] = True
        return vec

    def transform_batch(self, species: Sequence[Species]) -> np.ndarray:
        """Return a one-hot matrix for a sequence of species.

        Args:
            species: Sequence of ``(source_id, species_id)`` tuples.

        Returns:
            2-D boolean array of shape ``(len(species), n_species)``.
        """
        mat = np.zeros((len(species), self.n_species), dtype=bool)
        for i, s in enumerate(species):
            mat[i] = self.transform(s)
        return mat

    def inverse_transform(self, vec: np.ndarray) -> Species:
        """Return the species corresponding to a one-hot vector.

        Args:
            vec: 1-D array of shape ``(n_species,)`` with exactly one True element.

        Raises:
            ValueError: If *vec* does not have exactly one non-zero element.
        """
        indices = np.flatnonzero(vec)
        if len(indices) != 1:
            raise ValueError(f'Expected exactly one non-zero element, got {len(indices)}.')
        return self.species[int(indices[0])]
