from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pysephone.paths import get_data_root


@dataclass(frozen=True)
class ObservationData:
    """
    Normalized container for phenological observation data from a single source.

    All tables are keyed by `src` (the source KEY) as the first index level,
    enabling safe concatenation across sources.

    Table schemas
    -------------

    observations : pd.DataFrame
        One row per recorded observation.

        Index (MultiIndex):
            src         str            – data source identifier (ObservationSource.KEY)
            loc_id      *              – location identifier (source-specific type)
            year        int            – calendar year of the observation
            species_id  *              – species identifier (source-specific type)
            subgroup_id *              – subgroup/cultivar identifier (source-specific type)
            obs_type    *              – event type; foreign key into events.event

        Columns:
            date (KEY_OBSERVATIONS)  datetime64[ns] – observed calendar date

    events : pd.DataFrame
        Phenological event type definitions.

        Index (MultiIndex):
            src    str  – data source identifier
            event  *    – event identifier; matches observations.obs_type

        Columns:
            description  str  – human-readable description of the event

    locations : pd.DataFrame
        Site metadata.

        Index (MultiIndex):
            src     str  – data source identifier
            loc_id  *    – location identifier; matches observations.loc_id

        Columns:
            lat  float  – latitude (decimal degrees)
            lon  float  – longitude (decimal degrees)

    species : pd.DataFrame
        Species taxonomy.

        Index (MultiIndex):
            src        str  – data source identifier
            species_id  *   – species identifier; matches observations.species_id

        Columns:
            (source-specific)

    subgroups : pd.DataFrame
        Sub-classifications within a species (e.g. cultivars).

        Index (MultiIndex):
            src         str  – data source identifier
            subgroup_id  *   – subgroup identifier; matches observations.subgroup_id

        Columns:
            species_id  *  – species this subgroup belongs to; foreign key into species
            (additional source-specific columns)

    metadata : dict, optional
        Source-specific metadata (version, license, etc.).
    """
    observations: pd.DataFrame
    events: pd.DataFrame
    locations: pd.DataFrame
    species: pd.DataFrame
    subgroups: pd.DataFrame
    metadata: dict[str, Any] | None = None


class ObservationSource(ABC):
    """
    Abstract base class for integrating phenological observation data sources.

    Each concrete subclass represents one data source (e.g. a specific database
    or file format) and is responsible for downloading, caching, and loading its
    data into a standardized ObservationData container.

    Class attributes:
        KEY:    Unique string identifier for this source, used for directory
                naming and source registry lookups.
    """

    KEY: str

    def get_data(self, cfg: Mapping[str, Any], root: Path = None) -> ObservationData:
        """
        Load observation data for this source.

        Resolves the data root directory and delegates to _get_data. The
        source-specific subdirectory under root is accessible via
        paths.get_observations_source_data_dir(root, self.KEY).

        Args:
            cfg:    Source-specific configuration (e.g. date range, species filter,
                    bounding box). Keys and value types are defined by each subclass.
            root:   Data root directory. Defaults to get_data_root() which reads
                    the PYSEPHONE_DATA_ROOT environment variable or falls back
                    to ~/.pysephone.

        Returns:
            ObservationData with all tables populated.
        """
        if root is None:
            root = get_data_root()
        return self._get_data(cfg, root)

    @abstractmethod
    def _get_data(self, cfg: Mapping[str, Any], root: Path) -> ObservationData:
        """
        Source-specific implementation of data loading.

        Called by get_data with a resolved root path. Implementations should
        handle downloading and caching to disk when data is not yet available
        locally.

        Args:
            cfg:    Source-specific configuration.
            root:   Resolved data root directory.

        Returns:
            ObservationData with all tables populated.
        """
        raise NotImplementedError
