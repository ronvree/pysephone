"""
Tests for PhylogenyFeatures.

All tests are self-contained — no network access.  API calls are patched
with minimal stubs that return deterministic Newick / TNRS responses.
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pysephone.dataset.util.phylogeny import PhylogenyFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SRC = 'pep725'

# Simple 3-species Newick: ((A:1,B:2):0.5,C:3)
#   d(A,B) = 1 + 2 = 3
#   d(A,C) = 1 + 0.5 + 3 = 4.5   (A leaf depth = 1+0.5=1.5, C depth=3, LCA=root depth=0)
#   d(B,C) = 2 + 0.5 + 3 = 5.5
_NEWICK_3 = '((ott111:1.0,ott222:2.0):0.5,ott333:3.0)'
_OTT_IDS_3 = [111, 222, 333]

_SPECIES_NAMES_3 = {
    (_SRC, 111): 'Species alpha',
    (_SRC, 222): 'Species beta',
    (_SRC, 333): 'Species gamma',
}

_EXPECTED_DIST_3 = np.array([
    [0.0, 3.0, 4.5],
    [3.0, 0.0, 5.5],
    [4.5, 5.5, 0.0],
])


class _FakeObservations:
    """Minimal stand-in for Observations."""
    def __init__(self, species_names):
        self.species_names = species_names


def _make_tnrs_response(names, name_to_ott):
    """Build a mock TNRS JSON response."""
    results = []
    for name in names:
        ott = name_to_ott.get(name)
        if ott is None:
            results.append({'name': name, 'matches': []})
        else:
            results.append({
                'name': name,
                'matches': [{
                    'is_approximate_match': False,
                    'matched_name': name,
                    'taxon': {'ott_id': ott},
                }],
            })
    return {'results': results}


def _make_subtree_response(newick):
    return {'newick': newick}


# ---------------------------------------------------------------------------
# Unit tests: Newick parser + distance computation
# ---------------------------------------------------------------------------

class TestParseAndCompute:
    def test_distances_3_species(self):
        dist = PhylogenyFeatures._parse_and_compute(_NEWICK_3, _OTT_IDS_3)
        np.testing.assert_allclose(dist, _EXPECTED_DIST_3, atol=1e-9)

    def test_symmetric(self):
        dist = PhylogenyFeatures._parse_and_compute(_NEWICK_3, _OTT_IDS_3)
        np.testing.assert_allclose(dist, dist.T)

    def test_diagonal_zero(self):
        dist = PhylogenyFeatures._parse_and_compute(_NEWICK_3, _OTT_IDS_3)
        np.testing.assert_allclose(np.diag(dist), 0.0)

    def test_missing_ott_id_falls_back(self):
        # OTT ID 999 is not in the Newick; should get distance 1.0 to others
        dist = PhylogenyFeatures._parse_and_compute(_NEWICK_3, [111, 999])
        assert dist[0, 1] == pytest.approx(1.0)
        assert dist[1, 0] == pytest.approx(1.0)
        assert dist[0, 0] == pytest.approx(0.0)

    def test_star_tree(self):
        # ((A:2,B:3):0) → d(A,B) = 2+3 = 5 when root branch = 0
        newick = '(ott10:2.0,ott20:3.0)'
        dist = PhylogenyFeatures._parse_and_compute(newick, [10, 20])
        assert dist[0, 1] == pytest.approx(5.0)

    def test_newick_distances_fallback_on_exception(self):
        # If _parse_and_compute raises, _newick_distances should warn and
        # return a uniform (ones off-diagonal) fallback matrix.
        with patch.object(
            PhylogenyFeatures, '_parse_and_compute',
            side_effect=ValueError('simulated parse failure'),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                dist = PhylogenyFeatures._newick_distances('anything', [1, 2, 3])
        assert dist.shape == (3, 3)
        np.testing.assert_allclose(np.diag(dist), 0.0)
        assert len(w) >= 1
        assert any('simulated parse failure' in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Unit tests: classical MDS
# ---------------------------------------------------------------------------

class TestClassicalMDS:
    def test_output_shape(self):
        coords = PhylogenyFeatures._classical_mds(_EXPECTED_DIST_3, k=2)
        assert coords.shape == (3, 2)

    def test_dtype_float32(self):
        coords = PhylogenyFeatures._classical_mds(_EXPECTED_DIST_3, k=2)
        assert coords.dtype == np.float32

    def test_k_larger_than_n_minus_1_capped(self):
        # k=5 but only 3 species → capped at n-1=2 meaningful dims
        coords = PhylogenyFeatures._classical_mds(_EXPECTED_DIST_3, k=5)
        assert coords.shape == (3, 2)

    def test_zero_distance_matrix(self):
        # All species identical → MDS should return zeros
        dist = np.zeros((3, 3))
        coords = PhylogenyFeatures._classical_mds(dist, k=2)
        np.testing.assert_allclose(coords, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_output_is_mds(self):
        p = PhylogenyFeatures()
        assert p._output == ['mds']

    def test_unknown_output_raises(self):
        with pytest.raises(ValueError, match='Unknown output type'):
            PhylogenyFeatures(output=['bogus'])

    def test_valid_outputs_accepted(self):
        p = PhylogenyFeatures(output=['mds', 'distances'])
        assert set(p._output) == {'mds', 'distances'}


# ---------------------------------------------------------------------------
# Unit tests: get_data (no network — inject fitted state)
# ---------------------------------------------------------------------------

class TestGetData:
    @pytest.fixture
    def fitted_phylo(self):
        p = PhylogenyFeatures(k_embed=2, output=['mds', 'distances'])
        p._species_keys = [(_SRC, 111), (_SRC, 222), (_SRC, 333)]
        p._species_index = {(_SRC, 111): 0, (_SRC, 222): 1, (_SRC, 333): 2}
        p._distance_matrix = _EXPECTED_DIST_3.astype(np.float32)
        p._mds_coords = PhylogenyFeatures._classical_mds(_EXPECTED_DIST_3, k=2)
        return p

    def test_mds_keys_present(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert 'phylo_mds_1' in result
        assert 'phylo_mds_2' in result

    def test_mds_shape(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert result['phylo_mds_1'].shape == (1,)

    def test_distances_key_present(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert 'phylo_distances' in result

    def test_distances_shape(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert result['phylo_distances'].shape == (3,)

    def test_self_distance_is_zero(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert result['phylo_distances'][0] == pytest.approx(0.0)

    def test_distance_values_correct(self, fitted_phylo):
        result = fitted_phylo.get_data((_SRC, 'loc1', 2020, 111, 0))
        # Row 0 of _EXPECTED_DIST_3
        np.testing.assert_allclose(result['phylo_distances'], [0.0, 3.0, 4.5], atol=1e-5)

    def test_unknown_species_raises(self, fitted_phylo):
        with pytest.raises(KeyError):
            fitted_phylo.get_data((_SRC, 'loc1', 2020, 999, 0))

    def test_not_fitted_raises(self):
        p = PhylogenyFeatures()
        with pytest.raises(RuntimeError, match='fit\\(\\)'):
            p.get_data((_SRC, 'loc1', 2020, 111, 0))

    def test_mds_only_output(self):
        p = PhylogenyFeatures(k_embed=2, output=['mds'])
        p._species_keys = [(_SRC, 111)]
        p._species_index = {(_SRC, 111): 0}
        p._distance_matrix = np.zeros((1, 1))
        p._mds_coords = np.zeros((1, 2), dtype=np.float32)
        result = p.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert 'phylo_mds_1' in result
        assert 'phylo_distances' not in result

    def test_distances_only_output(self):
        p = PhylogenyFeatures(output=['distances'])
        p._species_keys = [(_SRC, 111)]
        p._species_index = {(_SRC, 111): 0}
        p._distance_matrix = np.zeros((1, 1), dtype=np.float32)
        p._mds_coords = None
        result = p.get_data((_SRC, 'loc1', 2020, 111, 0))
        assert 'phylo_distances' in result
        assert 'phylo_mds_1' not in result


# ---------------------------------------------------------------------------
# Integration test: fit() with mocked network calls
# ---------------------------------------------------------------------------

class TestFitMocked:
    """Tests for fit() using patched requests.post to avoid network access."""

    def _mock_post(self, url, json=None, timeout=None):
        """Side-effect function for requests.post."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if 'tnrs' in url:
            name_to_ott = {
                'Species alpha': 111,
                'Species beta':  222,
                'Species gamma': 333,
            }
            names = (json or {}).get('names', [])
            resp.json.return_value = _make_tnrs_response(names, name_to_ott)
        elif 'induced_subtree' in url:
            resp.json.return_value = _make_subtree_response(_NEWICK_3)
        else:
            resp.json.return_value = {}
        return resp

    @pytest.fixture
    def phylo(self):
        return PhylogenyFeatures(k_embed=2, output=['mds', 'distances'])

    def test_fit_sets_species_keys(self, phylo):
        obs = _FakeObservations(_SPECIES_NAMES_3)
        import requests as _req
        orig = _req.post
        try:
            _req.post = self._mock_post
            phylo.fit(obs)
        finally:
            _req.post = orig
        assert set(phylo.species_keys) == set(_SPECIES_NAMES_3.keys())

    def test_fit_distance_matrix_shape(self, phylo):
        obs = _FakeObservations(_SPECIES_NAMES_3)
        import requests as _req
        orig = _req.post
        try:
            _req.post = self._mock_post
            phylo.fit(obs)
        finally:
            _req.post = orig
        assert phylo.distance_matrix.shape == (3, 3)

    def test_fit_distance_values(self, phylo):
        obs = _FakeObservations(_SPECIES_NAMES_3)
        import requests as _req
        orig = _req.post
        try:
            _req.post = self._mock_post
            phylo.fit(obs)
        finally:
            _req.post = orig
        np.testing.assert_allclose(phylo.distance_matrix, _EXPECTED_DIST_3, atol=1e-9)

    def test_fit_mds_coords_shape(self, phylo):
        obs = _FakeObservations(_SPECIES_NAMES_3)
        import requests as _req
        orig = _req.post
        try:
            _req.post = self._mock_post
            phylo.fit(obs)
        finally:
            _req.post = orig
        assert phylo.mds_coords.shape == (3, 2)

    def test_fit_empty_species_names_raises(self, phylo):
        obs = _FakeObservations({})
        with pytest.raises(ValueError, match='species_names is empty'):
            phylo.fit(obs)

    def test_fit_warns_on_unresolved_name(self):
        obs = _FakeObservations({(_SRC, 111): 'Species alpha', (_SRC, 999): 'Unknown species'})

        def mock_post_partial(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if 'tnrs' in url:
                name_to_ott = {'Species alpha': 111}
                resp.json.return_value = _make_tnrs_response(
                    (json or {}).get('names', []), name_to_ott
                )
            elif 'induced_subtree' in url:
                resp.json.return_value = _make_subtree_response('(ott111:1.0)')
            else:
                resp.json.return_value = {}
            return resp

        phylo = PhylogenyFeatures(k_embed=1, output=['distances'])
        import requests as _req
        orig = _req.post
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                _req.post = mock_post_partial
                phylo.fit(obs)
            finally:
                _req.post = orig
        warning_msgs = [str(warning.message) for warning in w]
        assert any('Unknown species' in m or 'no TNRS match' in m for m in warning_msgs)
