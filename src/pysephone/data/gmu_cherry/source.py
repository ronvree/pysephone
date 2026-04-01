from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pysephone.constants import KEY_OBSERVATIONS
from pysephone.data.source import ObservationData, ObservationSource
from pysephone.data.gmu_cherry.bloom_doy import (
    get_df_japan,
    get_df_switzerland,
    get_df_south_korea,
)

# Single obs_type used for all GMU cherry observations
_OBS_TYPE = 'bloom_doy'
# Single species covering all observations in this dataset
_SPECIES_ID = 'prunus'
# No subgroup information is recorded in the source CSVs
_SUBGROUP_ID = 'default'


class GMUCherrySource(ObservationSource):
    """
    Data source for the GMU Cherry Blossom prediction competition dataset.

    Covers Japan, South Korea, and Switzerland (Prunus sp., first bloom DOY).
    Data is bundled with the package — no download required.

    Supported cfg keys: none.
    """

    KEY = 'gmu_cherry'

    def _get_data(self, cfg: Mapping[str, Any], root: Path) -> ObservationData:
        df = pd.concat([
            get_df_japan(),
            get_df_switzerland(),
            get_df_south_korea(),
        ], ignore_index=True)

        return ObservationData(
            observations=self._build_observations(df),
            events=self._build_events(),
            locations=self._build_locations(df),
            species=self._build_species(),
            subgroups=self._build_subgroups(),
        )

    def _build_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Index:   src, loc_id (location string), year,
                 species_id ('prunus'), subgroup_id ('default'), obs_type ('bloom_doy')
        Columns: date (datetime)
        """
        obs = df[['location', 'year', 'bloom_date']].copy()
        obs['src'] = self.KEY
        obs['loc_id'] = obs['location']
        obs['species_id'] = _SPECIES_ID
        obs['subgroup_id'] = _SUBGROUP_ID
        obs['obs_type'] = _OBS_TYPE
        obs[KEY_OBSERVATIONS] = pd.to_datetime(obs['bloom_date'])
        return obs.set_index(['src', 'loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type'])[[KEY_OBSERVATIONS]]

    def _build_events(self) -> pd.DataFrame:
        """
        Index:   src, event ('bloom_doy')
        Columns: description
        """
        return pd.DataFrame(
            {'description': ['Day of year of first bloom']},
            index=pd.MultiIndex.from_tuples(
                [(self.KEY, _OBS_TYPE)],
                names=['src', 'event'],
            ),
        )

    def _build_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Index:   src, loc_id (location string)
        Columns: lat, lon
        """
        locs = (
            df[['location', 'lat', 'long']]
            .drop_duplicates(subset='location')
            .rename(columns={'long': 'lon'})
        )
        locs['src'] = self.KEY
        locs['loc_id'] = locs['location']
        return locs.set_index(['src', 'loc_id'])[['lat', 'lon']]

    def _build_species(self) -> pd.DataFrame:
        """
        Index:   src, species_id ('prunus')
        Columns: genus, scientific_name
        """
        return pd.DataFrame(
            {'genus': ['Prunus'], 'scientific_name': ['Prunus sp.']},
            index=pd.MultiIndex.from_tuples(
                [(self.KEY, _SPECIES_ID)],
                names=['src', 'species_id'],
            ),
        )

    def _build_subgroups(self) -> pd.DataFrame:
        """
        Index:   src, subgroup_id ('default')
        Columns: species_id
        """
        return pd.DataFrame(
            {'species_id': [_SPECIES_ID]},
            index=pd.MultiIndex.from_tuples(
                [(self.KEY, _SUBGROUP_ID)],
                names=['src', 'subgroup_id'],
            ),
        )
