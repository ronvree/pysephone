"""
LSTM phenology models with sample-specific contextual features.

:class:`LSTMCtxModel` extends :class:`~pysephone.models.lstm.LSTMModel` with
a per-sample context vector that is **concatenated to the LSTM output before
the pointwise head** — i.e. context is *not* fed into the recurrent layers
but acts as a static side-channel that conditions the time-invariant head.

Subclasses implement :meth:`get_context_vectors`, which receives the collated
batch dict and returns a ``(B, ctx_dim)`` tensor.

Concrete subclasses provided here:

- :class:`PhylogeneticLSTMModel` — context = phylogenetic MDS embedding per
  ``(src, species_id)``.
- :class:`OneHotSpeciesLSTMModel` — context = one-hot encoding over the set
  of distinct species in the dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pysephone.constants import (
    KEY_DATA_SOURCE,
    KEY_FEATURES,
    KEY_OBSERVATIONS_INDEX,
    KEY_SPECIES_ID,
)
from pysephone.models.lstm import LSTMModel, LSTMModelArgs
from pysephone.utils.func_torch import create_left_mask


SpeciesKey = Tuple[str, int]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class LSTMCtxModelArgs(LSTMModelArgs):
    """Arguments for :class:`LSTMCtxModel` subclasses.

    Subclasses are responsible for declaring how the context vector is built
    (constructor arguments such as fitted MDS coords or species lists), so
    only the LSTM-side fields are inherited.
    """
    pass


class LSTMCtxModel(LSTMModel, ABC):
    """Abstract LSTM with a sample-specific context vector.

    Architecture::

        x  ──►  LSTM  ──►  (B, T, H)
                                │
        ctx (B, D) ── broadcast ┴──►  cat ──►  (B, T, H + D)  ──►  head ──►  ps

    Subclasses implement :meth:`get_context_vectors` to produce the
    ``(B, D)`` per-sample vector from the collated batch.

    Args:
        data_keys:           Same as :class:`LSTMModel`.
        ctx_dim:             Dimensionality of the per-sample context vector.
        hidden_size:         LSTM hidden size.
        num_layers:          LSTM depth.
        num_layers_lin:      Depth of the pointwise head (≥ 1).
        feature_statistics:  Per-key ``(mean, std)`` for input normalisation.
        obs_features:        Optional list of observation-index keys to mask
                             into the input as binary channels.
    """

    def __init__(
        self,
        data_keys: List[str],
        ctx_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_layers_lin: int = 2,
        feature_statistics: Optional[Dict[str, Tuple[float, float]]] = None,
        obs_features: Optional[List[str]] = None,
    ) -> None:
        if ctx_dim <= 0:
            raise ValueError(f'ctx_dim must be > 0, got {ctx_dim}')

        super().__init__(
            data_keys=data_keys,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_layers_lin=num_layers_lin,
            feature_statistics=feature_statistics,
            obs_features=obs_features,
        )

        self._ctx_dim = int(ctx_dim)

        # Replace the head: it now takes (hidden_size + ctx_dim) inputs.
        self._lin = self._build_head(
            num_layers=num_layers_lin,
            in_size=hidden_size + self._ctx_dim,
            hidden_size=hidden_size,
            out_size=1,
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        """Return the per-sample context tensor of shape ``(B, ctx_dim)``.

        Args:
            xs: Collated batch dict from
                :meth:`~BaseTorchModel.collate_fn`.  Use ``xs['src']`` and
                ``xs['species_id']`` (Python lists) for species lookup.

        Returns:
            Float tensor of shape ``(B, ctx_dim)`` on the same device as the
            model parameters.
        """

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, xs: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features: Dict[str, torch.Tensor] = xs[KEY_FEATURES]
        features = {
            k: (v - self._feature_statistics[k][0]) / self._feature_statistics[k][1]
            for k, v in features.items()
        }
        fs: List[torch.Tensor] = [features[k] for k in self._data_keys]

        if self._obs_features is not None:
            obs_index: Dict[str, torch.Tensor] = xs[KEY_OBSERVATIONS_INDEX]
            season_length = fs[0].size(1)
            for key in self._obs_features:
                obs_ixs = obs_index[key].long()
                fs.append(create_left_mask(season_length, obs_ixs).float())

        x = torch.cat([f.unsqueeze(-1) for f in fs], dim=-1)   # (B, T, M)
        x = torch.nan_to_num(x)

        x, _ = self._rnn(x)                                    # (B, T, H)
        x = F.relu(x)

        # ── Sample-specific context ───────────────────────────────────────
        ctx = self.get_context_vectors(xs).to(x.device)        # (B, D)
        if ctx.dim() != 2 or ctx.size(0) != x.size(0) or ctx.size(1) != self._ctx_dim:
            raise ValueError(
                f'get_context_vectors returned shape {tuple(ctx.shape)}, '
                f'expected ({x.size(0)}, {self._ctx_dim})'
            )
        ctx = ctx.unsqueeze(1).expand(-1, x.size(1), -1)       # (B, T, D)
        x = torch.cat([x, ctx], dim=-1)                        # (B, T, H+D)
        # ──────────────────────────────────────────────────────────────────

        x = x.permute(0, 2, 1)
        ps = torch.sigmoid(self._lin(x)).squeeze(1)

        diff = ps - torch.roll(ps, 1, dims=-1)
        ixs = torch.argmax(diff, dim=-1).clamp(0, ps.size(-1) - 1)

        return ixs.float(), {'ps': ps}


# ---------------------------------------------------------------------------
# Concrete: phylogenetic-embedding LSTM
# ---------------------------------------------------------------------------

class PhylogeneticLSTMModel(LSTMCtxModel):
    """LSTM whose context is a per-species phylogenetic MDS embedding.

    Args:
        data_keys:    Meteorological feature keys.
        species_keys: Ordered list of ``(src, species_id)`` tuples — one per
                      row of *mds_coords*.
        mds_coords:   ``(N_species, k)`` float array of MDS coordinates,
                      typically obtained from a fitted
                      :class:`~pysephone.dataset.util.phylogeny.PhylogenyFeatures`.
        unknown:      Behaviour for batch entries whose ``(src, species_id)``
                      is not in *species_keys*:
                      ``'zero'`` → emit a zero context vector,
                      ``'error'`` → raise :class:`KeyError`.
        ...           Remaining kwargs forwarded to :class:`LSTMCtxModel`.
    """

    def __init__(
        self,
        data_keys: List[str],
        species_keys: Sequence[SpeciesKey],
        mds_coords: np.ndarray,
        unknown: str = 'zero',
        **lstm_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if len(species_keys) != mds_coords.shape[0]:
            raise ValueError(
                f'species_keys ({len(species_keys)}) and mds_coords '
                f'({mds_coords.shape[0]}) row count must match'
            )

        ctx_dim = int(mds_coords.shape[1])
        super().__init__(data_keys=data_keys, ctx_dim=ctx_dim, **lstm_kwargs)

        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }
        coords = np.asarray(mds_coords, dtype=np.float32)
        mean = coords.mean(axis=0)
        std  = coords.std(axis=0).clip(min=1e-8)
        self.register_buffer('_mds_table', torch.from_numpy((coords - mean) / std))
        self.register_buffer('_mds_mean',  torch.from_numpy(mean))
        self.register_buffer('_mds_std',   torch.from_numpy(std))

    @classmethod
    def from_phylogeny_features(
        cls,
        phylo,                          # PhylogenyFeatures (already fitted)
        data_keys: List[str],
        unknown: str = 'zero',
        **lstm_kwargs: Any,
    ) -> 'PhylogeneticLSTMModel':
        """Convenience constructor from a fitted ``PhylogenyFeatures`` instance."""
        if phylo.mds_coords is None:
            raise ValueError(
                'PhylogenyFeatures must be fitted with output containing "mds" '
                'before constructing a PhylogeneticLSTMModel.'
            )
        return cls(
            data_keys=data_keys,
            species_keys=list(phylo.species_keys),
            mds_coords=np.asarray(phylo.mds_coords),
            unknown=unknown,
            **lstm_kwargs,
        )

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = self._mds_table.device

        rows = []
        for s, sid in zip(srcs, sids):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in fitted phylogeny '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
                rows.append(None)
            else:
                rows.append(row)

        ctx = torch.zeros(len(rows), self._ctx_dim, device=device)
        for i, r in enumerate(rows):
            if r is not None:
                ctx[i] = self._mds_table[r]
        return ctx


# ---------------------------------------------------------------------------
# Concrete: one-hot species LSTM
# ---------------------------------------------------------------------------

class OneHotSpeciesLSTMModel(LSTMCtxModel):
    """LSTM whose context is a one-hot encoding of the sample's species.

    Args:
        data_keys:    Meteorological feature keys.
        species_keys: Ordered list of ``(src, species_id)`` tuples; the
                      one-hot dimension follows this order.
        unknown:      Behaviour for unseen species (same semantics as
                      :class:`PhylogeneticLSTMModel`).
        ...           Remaining kwargs forwarded to :class:`LSTMCtxModel`.
    """

    def __init__(
        self,
        data_keys: List[str],
        species_keys: Sequence[SpeciesKey],
        unknown: str = 'zero',
        **lstm_kwargs: Any,
    ) -> None:
        if unknown not in ('zero', 'error'):
            raise ValueError(f"unknown must be 'zero' or 'error', got {unknown!r}")
        species_keys = list(species_keys)
        if not species_keys:
            raise ValueError('species_keys must be non-empty')

        super().__init__(
            data_keys=data_keys,
            ctx_dim=len(species_keys),
            **lstm_kwargs,
        )
        self._unknown = unknown
        self._species_index: Dict[SpeciesKey, int] = {
            (str(s), int(sid)): i for i, (s, sid) in enumerate(species_keys)
        }

    @classmethod
    def from_dataset(
        cls,
        dataset,
        data_keys: List[str],
        unknown: str = 'zero',
        **lstm_kwargs: Any,
    ) -> 'OneHotSpeciesLSTMModel':
        """Build from the unique ``(src, species_id)`` pairs in *dataset*."""
        seen: Dict[SpeciesKey, None] = {}
        for ix in dataset.iter_index():
            key = (str(ix[0]), int(ix[3]))
            if key not in seen:
                seen[key] = None
        species_keys = sorted(seen)
        return cls(
            data_keys=data_keys,
            species_keys=species_keys,
            unknown=unknown,
            **lstm_kwargs,
        )

    def get_context_vectors(self, xs: Dict[str, Any]) -> torch.Tensor:
        srcs = xs[KEY_DATA_SOURCE]
        sids = xs[KEY_SPECIES_ID]
        device = next(self.parameters()).device

        ctx = torch.zeros(len(srcs), self._ctx_dim, device=device)
        for i, (s, sid) in enumerate(zip(srcs, sids)):
            row = self._species_index.get((str(s), int(sid)))
            if row is None:
                if self._unknown == 'error':
                    raise KeyError(
                        f'Species ({s!r}, {int(sid)}) not in species_keys '
                        f'(known: {list(self._species_index.keys())[:5]}...)'
                    )
                continue   # leave the zero vector
            ctx[i, row] = 1.0
        return ctx
