from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pysephone.constants import KEY_OBSERVATIONS
from pysephone.data.source import ObservationData, ObservationSource
from pysephone.data.usa_npn.download import (
    fetch_species_table,
    fetch_all_phenometrics,
    filter_species_by_genus,
    create_observations_df,
    create_events_df,
    create_locations_df,
    create_species_df,
    create_subgroups_df,
)


# Default phenophases: 501 = Open flowers (≈ PEP725 BBCH_60),
#                      371 = Breaking leaf buds (≈ PEP725 BBCH_11).
_DEFAULT_PHENOPHASES = (501, 371)
_DEFAULT_START_DATE  = '2009-01-01'
_DEFAULT_END_DATE    = '2024-12-31'


class USANPNSource(ObservationSource):
    """
    Data source for the USA National Phenology Network (USA-NPN).

    Pulls Individual Phenometrics from the public web service at
    https://services.usanpn.org/npn_portal/ and shapes the response into the
    standard ObservationData container.

    Downloads are cached on disk under
    ``<data_root>/data/observations/usa_npn/raw/``. Cache filenames match the
    convention used by the USA-NPN exploration notebooks, so a cache produced
    by either side is reused by the other.

    Supported cfg keys
    ------------------
    genera : Iterable[str] | None
        Restrict the species set to these genera (case-sensitive). If ``None``,
        the entire catalogue is requested. Default: ``None``.
    species_ids : Iterable[int] | None
        If provided, overrides ``genera`` and limits the request to these
        explicit species_ids. Default: ``None``.
    phenophase_ids : Iterable[int]
        Phenophases to download. Default: ``(501, 371)`` — open flowers and
        breaking leaf buds.
    start_date : str
        Inclusive start date (ISO 'YYYY-MM-DD'). Default: ``'2009-01-01'``.
    end_date : str
        Inclusive end date (ISO 'YYYY-MM-DD'). Default: ``'2024-12-31'``.
    request_src : str
        ``request_src`` query-string parameter sent to USA-NPN; identifies the
        caller in their server logs. Default: ``'pysephone'``.
    force_download : bool
        Re-download the species catalogue and phenometrics even if cached.
        Default: ``False``.
    verbose : bool
        Show progress lines / bars during download. Default: ``True``.
    """

    KEY = 'usa_npn'

    def _get_data(self, cfg: Mapping[str, Any], root: Path) -> ObservationData:
        genera         = cfg.get('genera')
        species_ids    = cfg.get('species_ids')
        phenophase_ids = tuple(cfg.get('phenophase_ids', _DEFAULT_PHENOPHASES))
        start_date     = cfg.get('start_date', _DEFAULT_START_DATE)
        end_date       = cfg.get('end_date', _DEFAULT_END_DATE)
        request_src    = cfg.get('request_src', 'pysephone')
        force_download = cfg.get('force_download', False)
        verbose        = cfg.get('verbose', True)

        # 1. Catalogue + species filter
        df_catalogue = fetch_species_table(
            root,
            request_src=request_src,
            force_download=force_download,
            verbose=verbose,
        )
        if species_ids is not None:
            df_filtered = df_catalogue[
                df_catalogue['species_id'].astype(int).isin({int(s) for s in species_ids})
            ].reset_index(drop=True)
        else:
            df_filtered = filter_species_by_genus(df_catalogue, genera)

        if df_filtered.empty:
            raise ValueError(
                'USA-NPN species filter matched zero species. Check `genera` '
                'or `species_ids` in cfg.'
            )

        # 2. Bulk phenometrics, one request per phenophase
        phenometrics = fetch_all_phenometrics(
            root,
            species_ids=df_filtered['species_id'].astype(int).tolist(),
            phenophase_ids=phenophase_ids,
            start_date=start_date,
            end_date=end_date,
            request_src=request_src,
            force_download=force_download,
            verbose=verbose,
        )

        return ObservationData(
            observations=self._build_observations(phenometrics),
            events=self._build_events(phenometrics),
            locations=self._build_locations(phenometrics),
            species=self._build_species(df_filtered),
            subgroups=self._build_subgroups(phenometrics),
            metadata={
                'phenophase_ids': list(phenophase_ids),
                'start_date':     start_date,
                'end_date':       end_date,
                'n_species_requested': int(len(df_filtered)),
            },
        )

    # ------------------------------------------------------------------ tables

    def _build_observations(self, phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Index:   src, loc_id, year, species_id, subgroup_id, obs_type
        Columns: observations  (datetime — first-yes date)
        """
        df = create_observations_df(phenometrics).reset_index()
        df['src'] = self.KEY
        df = df.rename(columns={'first_yes_date': KEY_OBSERVATIONS})
        return df.set_index(
            ['src', 'loc_id', 'year', 'species_id', 'subgroup_id', 'obs_type']
        )[[KEY_OBSERVATIONS]]

    def _build_events(self, phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Index:   src, event ('NPN_{phenophase_id}')
        Columns: description
        """
        df = create_events_df(phenometrics).reset_index()
        df['src'] = self.KEY
        return df.set_index(['src', 'event'])[['description']]

    def _build_locations(self, phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Index:   src, loc_id (site_id, int)
        Columns: lat, lon, alt, state
        """
        df = create_locations_df(phenometrics).reset_index()
        df['src'] = self.KEY
        keep = [c for c in ['lat', 'lon', 'alt', 'state'] if c in df.columns]
        return df.set_index(['src', 'loc_id'])[keep]

    def _build_species(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Index:   src, species_id (int)
        Columns: genus, species, common_name, family_name, kingdom (whichever exist)
        """
        df = create_species_df(df_filtered).reset_index()
        df['src'] = self.KEY
        keep = [c for c in df.columns if c not in ('src', 'species_id')]
        return df.set_index(['src', 'species_id'])[keep]

    def _build_subgroups(self, phenometrics: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Index:   src, subgroup_id (individual_id, int)
        Columns: species_id

        Each `individual_id` is one observed plant; subgroup ↔ individual.
        """
        df = create_subgroups_df(phenometrics).reset_index()
        df['src'] = self.KEY
        return df.set_index(['src', 'subgroup_id'])[['species_id']]
