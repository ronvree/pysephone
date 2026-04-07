"""
Phylogeny-based feature provider.

Uses the OpenTree of Life API to resolve species names, build a phylogenetic
tree, compute patristic distances, and embed species into a low-dimensional
space via classical MDS.

Usage::

    from pysephone.dataset.util.phylogeny import PhylogenyFeatures

    phylo = PhylogenyFeatures(k_embed=8, output=['mds', 'distances'])
    phylo.fit(observations)          # resolves names, builds tree, computes MDS

    # Inside a Dataset with this provider attached:
    item = dataset[ix]
    item['features']['phylo_mds_1']       # first MDS coordinate (scalar array)
    item['features']['phylo_distances']   # distances to all species (1-D array)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from pysephone.dataset.util.provider import FeatureProvider


class PhylogenyFeatures(FeatureProvider):
    """
    Feature provider that returns phylogenetic embeddings per sample species.

    Resolves scientific species names via OpenTree TNRS, retrieves the induced
    subtree, computes an all-pairs patristic distance matrix, and optionally
    embeds it via classical MDS.

    Args:
        k_embed:  Number of MDS dimensions.  Only used when ``'mds'`` is in
                  *output*.
        output:   List of output types to include in ``get_data`` results.
                  Supported values:

                  - ``'mds'``       — ``phylo_mds_1`` … ``phylo_mds_k``
                    (scalar ``np.ndarray`` of shape ``(1,)`` per coordinate)
                  - ``'distances'`` — ``phylo_distances``, a 1-D array of
                    patristic distances to every other species in the dataset,
                    sorted by ``(src, species_id)``

    Call :meth:`fit` with an :class:`~pysephone.dataset.observations.Observations`
    instance before attaching to a dataset.
    """

    def __init__(
        self,
        k_embed: int = 8,
        output: Optional[List[str]] = None,
    ) -> None:
        valid = {'mds', 'distances'}
        self._output: List[str] = list(output or ['mds'])
        unknown = set(self._output) - valid
        if unknown:
            raise ValueError(f"Unknown output type(s): {unknown}. Valid: {valid}")
        self._k = k_embed

        # Set after fit()
        self._species_keys: List[Tuple[str, int]] = []   # sorted (src, species_id)
        self._distance_matrix: Optional[np.ndarray] = None  # shape (N, N)
        self._mds_coords: Optional[np.ndarray] = None       # shape (N, k)
        self._species_index: Dict[Tuple[str, int], int] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, observations) -> 'PhylogenyFeatures':
        """Build the phylogenetic distance matrix and MDS embedding.

        Args:
            observations: An :class:`~pysephone.dataset.observations.Observations`
                          instance that has a populated ``species_names`` dict.

        Returns:
            *self* (for chaining).

        Raises:
            ValueError: If no species names are available on *observations*.
            RuntimeError: If the OpenTree API cannot be reached or returns no
                          usable matches.
        """
        species_names: Dict[Tuple[str, int], str] = observations.species_names
        if not species_names:
            raise ValueError(
                "observations.species_names is empty. "
                "Make sure the data source provides species names "
                "(e.g. PEP725Source sets 'species_name' in the species DataFrame)."
            )

        # Stable ordering so distance rows/cols are reproducible
        self._species_keys = sorted(species_names.keys())
        self._species_index = {k: i for i, k in enumerate(self._species_keys)}
        names = [species_names[k] for k in self._species_keys]

        ott_ids = self._resolve_names(names)
        dist = self._compute_distances(ott_ids, names)
        self._distance_matrix = dist

        if 'mds' in self._output:
            self._mds_coords = self._classical_mds(dist, self._k)

        return self

    # ------------------------------------------------------------------
    # FeatureProvider interface
    # ------------------------------------------------------------------

    def get_data(self, index: Tuple) -> Dict[str, np.ndarray]:
        """Return phylogenetic features for one sample.

        Args:
            index: ``(src, loc_id, year, species_id, subgroup_id)``

        Returns:
            Dict with entries determined by the *output* constructor parameter.
        """
        if self._distance_matrix is None:
            raise RuntimeError("PhylogenyFeatures.fit() has not been called.")

        src, _loc_id, _year, species_id, _subgroup_id = index
        key = (src, species_id)

        if key not in self._species_index:
            raise KeyError(
                f"Species {key} not found in fitted phylogeny. "
                f"Known keys: {self._species_keys}"
            )

        row = self._species_index[key]
        result: Dict[str, np.ndarray] = {}

        if 'mds' in self._output:
            coords = self._mds_coords[row]  # shape (k,)
            for j in range(self._k):
                result[f'phylo_mds_{j + 1}'] = np.array([coords[j]], dtype=np.float32)

        if 'distances' in self._output:
            result['phylo_distances'] = self._distance_matrix[row].astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Internal: name resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_names(names: List[str]) -> Dict[str, int]:
        """Resolve scientific names to OTT IDs via OpenTree TNRS.

        Returns:
            ``{name: ott_id}`` for successfully matched names.  Unmatched names
            produce a warning and are excluded from the returned dict.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for PhylogenyFeatures (pip install requests)")

        url = 'https://api.opentreeoflife.org/v3/tnrs/match_names'
        payload = {'names': names, 'do_approximate_matching': False}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        ott_ids: Dict[str, int] = {}
        for result in data.get('results', []):
            query = result.get('name', '')
            matches = result.get('matches', [])
            if not matches:
                warnings.warn(f"PhylogenyFeatures: no TNRS match for '{query}'")
                continue
            best = matches[0]
            if best.get('is_approximate_match'):
                warnings.warn(
                    f"PhylogenyFeatures: approximate match for '{query}' "
                    f"→ '{best.get('matched_name', '')}'"
                )
            taxon = best.get('taxon', {})
            ott_id = taxon.get('ott_id')
            if ott_id is not None:
                ott_ids[query] = int(ott_id)
            else:
                warnings.warn(f"PhylogenyFeatures: TNRS returned no ott_id for '{query}'")

        return ott_ids

    # ------------------------------------------------------------------
    # Internal: tree retrieval and distance computation
    # ------------------------------------------------------------------

    def _compute_distances(
        self,
        ott_ids: Dict[str, int],
        all_names: List[str],
    ) -> np.ndarray:
        """Build an N×N patristic distance matrix (N = len(all_names)).

        Species that could not be resolved get distance 0 to themselves and
        the maximum observed distance to all others (a conservative fallback).

        Args:
            ott_ids:   ``{name: ott_id}`` from :meth:`_resolve_names`.
            all_names: Full list of species names in index order.

        Returns:
            Symmetric distance matrix, shape ``(N, N)``.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for PhylogenyFeatures")

        N = len(all_names)
        dist = np.zeros((N, N), dtype=np.float64)

        resolved_names = [n for n in all_names if n in ott_ids]
        resolved_ids = [ott_ids[n] for n in resolved_names]

        if len(resolved_ids) < 2:
            warnings.warn(
                "PhylogenyFeatures: fewer than 2 species resolved — "
                "distance matrix will be all zeros."
            )
            return dist

        # Fetch induced subtree in Newick format
        url = 'https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree'
        payload = {'ott_ids': resolved_ids, 'label_format': 'id'}
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        newick = response.json().get('newick', '')

        if not newick:
            warnings.warn("PhylogenyFeatures: OpenTree returned empty Newick string.")
            return dist

        sub_dist = self._newick_distances(newick, resolved_ids)

        # Map back to the full N×N matrix
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        resolved_idx = [name_to_idx[n] for n in resolved_names]

        for i, gi in enumerate(resolved_idx):
            for j, gj in enumerate(resolved_idx):
                dist[gi, gj] = sub_dist[i, j]

        # Unresolved species: fill with max distance
        max_dist = dist.max()
        unresolved_idx = [
            name_to_idx[n] for n in all_names if n not in ott_ids
        ]
        for ui in unresolved_idx:
            dist[ui, :] = max_dist
            dist[:, ui] = max_dist
            dist[ui, ui] = 0.0

        return dist

    @staticmethod
    def _newick_distances(newick: str, ott_ids: List[int]) -> np.ndarray:
        """Parse Newick and compute all-pairs patristic distances.

        Falls back to a uniform distance matrix if parsing fails.
        """
        try:
            return PhylogenyFeatures._parse_and_compute(newick, ott_ids)
        except Exception as exc:
            warnings.warn(
                f"PhylogenyFeatures: Newick parsing failed ({exc}); "
                "using uniform distances."
            )
            n = len(ott_ids)
            d = np.ones((n, n), dtype=np.float64)
            np.fill_diagonal(d, 0.0)
            return d

    @staticmethod
    def _parse_and_compute(newick: str, ott_ids: List[int]) -> np.ndarray:
        """Parse a Newick string and compute patristic distances.

        Labels in the OpenTree induced-subtree response have the form
        ``ottNNNNNN`` (with optional suffixes like ``_mrca``).  We match
        labels to the requested OTT IDs by numeric OTT ID.

        Args:
            newick:  Newick string returned by OpenTree.
            ott_ids: OTT IDs in the same order as the rows/cols of the
                     returned matrix.

        Returns:
            Symmetric patristic distance matrix, shape ``(n, n)``.
        """
        import re

        # ----------------------------------------------------------------
        # Minimal recursive Newick parser
        # ----------------------------------------------------------------
        class Node:
            __slots__ = ('label', 'length', 'children')

            def __init__(self, label: str = '', length: float = 1.0):
                self.label = label
                self.length = length
                self.children: List['Node'] = []

        def _split_children(s: str) -> List[str]:
            """Split a comma-separated child list, respecting nested parens."""
            parts: List[str] = []
            depth = 0
            start = 0
            for i, ch in enumerate(s):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    parts.append(s[start:i])
                    start = i + 1
            parts.append(s[start:])
            return parts

        def _set_label_length(s: str, node: Node) -> None:
            if ':' in s:
                label_part, length_part = s.rsplit(':', 1)
                node.label = label_part.strip()
                try:
                    node.length = float(length_part.strip())
                except ValueError:
                    node.length = 1.0
            else:
                node.label = s.strip()
                node.length = 1.0

        def _parse_node(s: str, node: Node) -> None:
            if s.startswith('('):
                # Find matching closing paren
                depth = 0
                end = -1
                for i, ch in enumerate(s):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                for cs in _split_children(s[1:end]):
                    child = Node()
                    _parse_node(cs.strip(), child)
                    node.children.append(child)
                _set_label_length(s[end + 1:], node)
            else:
                _set_label_length(s, node)

        root = Node()
        _parse_node(newick.strip().rstrip(';'), root)

        # ----------------------------------------------------------------
        # Compute patristic distances via root-distance + LCA
        #
        # Strategy: DFS assigns each node a root-distance (sum of branch
        # lengths from root).  Patristic distance between two leaves =
        # dist(leaf_a) + dist(leaf_b) - 2 * dist(LCA(leaf_a, leaf_b)).
        # We compute this by collecting, for each leaf, the full path of
        # (node, cumulative_dist) pairs from root → leaf.  LCA is the
        # last common entry in both paths.
        # ----------------------------------------------------------------

        def _get_leaf_paths(node: Node, path: List[Tuple]) -> Dict[str, List[Tuple]]:
            """Map leaf label → list of (node, cum_dist_from_root) pairs."""
            current_path = path + [(node, (path[-1][1] if path else 0.0) + node.length)]
            if not node.children:
                return {node.label: current_path}
            result: Dict[str, List[Tuple]] = {}
            for c in node.children:
                result.update(_get_leaf_paths(c, current_path))
            return result

        leaf_paths = _get_leaf_paths(root, [])

        # Map OTT IDs to leaf labels present in the tree
        ott_set = set(ott_ids)
        id_to_label: Dict[int, str] = {}
        for label in leaf_paths:
            m = re.search(r'ott(\d+)', label)
            if m:
                ott = int(m.group(1))
                if ott in ott_set:
                    id_to_label[ott] = label

        n = len(ott_ids)
        dist = np.zeros((n, n), dtype=np.float64)

        for i, id_i in enumerate(ott_ids):
            for j in range(i + 1, n):
                id_j = ott_ids[j]
                label_i = id_to_label.get(id_i)
                label_j = id_to_label.get(id_j)
                if label_i is None or label_j is None:
                    d = 1.0
                else:
                    path_i = leaf_paths[label_i]
                    path_j = leaf_paths[label_j]
                    # LCA: last node common to both paths
                    nodes_i = {id(entry[0]): entry[1] for entry in path_i}
                    lca_dist = 0.0
                    for entry in reversed(path_j):
                        nd, cum = entry
                        if id(nd) in nodes_i:
                            lca_dist = nodes_i[id(nd)]
                            break
                    d_i = path_i[-1][1]
                    d_j = path_j[-1][1]
                    d = d_i + d_j - 2.0 * lca_dist
                dist[i, j] = d
                dist[j, i] = d

        return dist

    # ------------------------------------------------------------------
    # Internal: classical MDS
    # ------------------------------------------------------------------

    @staticmethod
    def _classical_mds(dist: np.ndarray, k: int) -> np.ndarray:
        """Classical (metric) MDS on a distance matrix.

        Args:
            dist: Symmetric distance matrix, shape ``(N, N)``.
            k:    Number of dimensions to keep.

        Returns:
            Coordinate matrix, shape ``(N, k)``.
        """
        N = dist.shape[0]
        D2 = dist ** 2
        H = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * H @ D2 @ H

        # Symmetrize to suppress floating-point asymmetry
        B = (B + B.T) / 2

        eigvals, eigvecs = np.linalg.eigh(B)
        # eigh returns ascending order; take the k largest positive eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Clamp tiny negatives to zero before sqrt
        k_actual = min(k, N - 1)
        lam = np.maximum(eigvals[:k_actual], 0.0)
        coords = eigvecs[:, :k_actual] * np.sqrt(lam)

        # Pad with zeros if fewer than k eigenvalues are available
        if k_actual < k:
            coords = np.hstack([coords, np.zeros((N, k - k_actual))])

        return coords.astype(np.float32)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def species_keys(self) -> List[Tuple[str, int]]:
        """Ordered list of ``(src, species_id)`` keys in distance matrix order."""
        return list(self._species_keys)

    @property
    def distance_matrix(self) -> Optional[np.ndarray]:
        """All-pairs patristic distance matrix, shape ``(N, N)``."""
        return self._distance_matrix

    @property
    def mds_coords(self) -> Optional[np.ndarray]:
        """MDS coordinate matrix, shape ``(N, k)``.  ``None`` if not computed."""
        return self._mds_coords
