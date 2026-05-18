"""
USA-NPN dataset definitions.

Each public builder returns an Observations object for one pre-configured
dataset variant.  Shared configuration lives at module level so it is easy
to spot and adjust without touching individual builders.

USA-NPN observation coverage starts in 2009. Phenophases used here:
    NPN_501 — Open flowers          (≈ PEP725 BBCH_60)
    NPN_371 — Breaking leaf buds    (≈ PEP725 BBCH_11)
"""

from datetime import datetime
from functools import reduce
from typing import List, Sequence, Tuple, Union

from pysephone.data.usa_npn.source import USANPNSource
from pysephone.dataset.observations import Observations
from pysephone.dataset.preprocessing.usa_npn import get_usa_npn_dataframes

_SRC = USANPNSource.KEY  # 'usa_npn'

# Obs-type aliases (built from phenophase id by the source)
_OBS_FLOWER   = 'NPN_501'  # Open flowers
_OBS_LEAF_BUD = 'NPN_371'  # Breaking leaf buds

# Phenophase ids requested from the USA-NPN web service. Pin both so a single
# cache file pair covers every builder in this registry.
_PHENOPHASE_IDS: Tuple[int, ...] = (501, 371)

# Date window for the underlying download. NPN data starts 2009-01-01.
_DOWNLOAD_START_DATE = '2009-01-01'
_DOWNLOAD_END_DATE   = '2024-12-31'

# Curated deciduous-fruit / nut / berry genera. Matches the set used by
# ``notebooks/usa_npn_deciduous_fruit_trees.ipynb`` so a fresh download and a
# notebook download share the same cache file. Genera absent from NPN
# (Mespilus, Punica) are kept here as a documentation reference; the source
# filter just returns zero species for them.
_FRUIT_GENERA: Tuple[str, ...] = (
    # Rosaceae fruit trees — counterparts of PEP725_fruit_trees
    'Malus',     # apple
    'Pyrus',     # pear
    'Prunus',    # cherry / peach / plum / almond / apricot
    'Cydonia',   # quince
    'Mespilus',  # medlar (absent from NPN)
    # Other deciduous fruit / nut trees
    'Corylus',     # hazel
    'Castanea',    # chestnut
    'Juglans',     # walnut
    'Carya',       # hickory / pecan
    'Diospyros',   # persimmon
    'Asimina',     # pawpaw
    'Morus',       # mulberry
    'Punica',      # pomegranate (absent from NPN)
    'Ficus',       # fig
    # Deciduous fruiting shrubs / vines
    'Vitis',       # grape
    'Vaccinium',   # blueberry / cranberry
    'Ribes',       # currant / gooseberry
    'Rubus',       # raspberry / blackberry
)

# General benchmark (bm) — mirrors the PEP725 registry knobs.
_BM_YEAR_MIN: int = 2009
_BM_YEAR_MAX: int = datetime.now().year - 1
_BM_YEARS: List[int] = list(range(_BM_YEAR_MIN, _BM_YEAR_MAX + 1))
_BM_DO_AGG: bool = True
_BM_AGG_METHOD: str = 'median'
_BM_REMOVE_OUTLIERS: bool = True
_BM_ASSERT_TARGET: bool = True


# Type alias for the taxon-selector list accepted by the preprocessing helper.
Taxon = Union[str, Tuple[str, str]]


def _usa_npn_data():
    """Load USA-NPN fruit-tree ObservationData (reads from disk / triggers download)."""
    return USANPNSource().get_data({
        'genera':         _FRUIT_GENERA,
        'phenophase_ids': _PHENOPHASE_IDS,
        'start_date':     _DOWNLOAD_START_DATE,
        'end_date':       _DOWNLOAD_END_DATE,
        'verbose':        False,
    })


# ---------------------------------------------------------------------------
# Helper: build a USA-NPN observation dataset with benchmark defaults
# ---------------------------------------------------------------------------

def _build_bm(taxa: Sequence[Taxon],
              obs_types: List[str],
              target_obs: str) -> Observations:
    dfs = get_usa_npn_dataframes(
        _usa_npn_data(),
        remove_outliers=_BM_REMOVE_OUTLIERS,
        filter_on_taxa=list(taxa),
        filter_on_observation_types=obs_types,
        filter_on_years=_BM_YEARS,
        datetime_observations=True,
    )
    obs = Observations(
        dfs['data'], dfs['locations'], species_names=dfs.get('species_names'),
    )
    if _BM_ASSERT_TARGET:
        obs = obs.select_by_observation_requirement(target_obs)
    if _BM_DO_AGG:
        obs = obs.aggregate_in_grid(method=_BM_AGG_METHOD)
    return obs


# ---------------------------------------------------------------------------
# USA-NPN individual fruit trees — flowering benchmark
# Default obs set: [open flowers, breaking leaf buds], target = open flowers.
# ---------------------------------------------------------------------------

_DEFAULT_OBS_TYPES = [_OBS_FLOWER, _OBS_LEAF_BUD]
_DEFAULT_TARGET    = _OBS_FLOWER


def build_usa_npn_apple(**kwargs) -> Observations:
    # Malus — apple. Includes 'Malus spp.', Malus domestica, Malus toringo.
    return _build_bm(['Malus'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_pear(**kwargs) -> Observations:
    # Pyrus — pear. Includes Pyrus communis and the ornamental Pyrus calleryana.
    return _build_bm(['Pyrus'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_cherry(**kwargs) -> Observations:
    # Prunus cherries (excludes plums, peach, apricot, almond).
    taxa: List[Taxon] = [
        ('Prunus', 'avium'),         # sweet cherry
        ('Prunus', 'cerasus'),       # sour cherry
        ('Prunus', 'serotina'),      # black cherry
        ('Prunus', 'virginiana'),    # chokecherry
        ('Prunus', 'yedoensis'),     # Yoshino cherry
        ('Prunus', 'serrulata'),     # Japanese cherry
        ('Prunus', 'pensylvanica'),  # pin cherry
        ('Prunus', 'emarginata'),    # bitter cherry
    ]
    return _build_bm(taxa, _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_peach(**kwargs) -> Observations:
    # Prunus persica
    return _build_bm([('Prunus', 'persica')], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_plum(**kwargs) -> Observations:
    # Prunus plums (Old World + North American).
    taxa: List[Taxon] = [
        ('Prunus', 'americana'),  # American plum
        ('Prunus', 'maritima'),   # beach plum
        ('Prunus', 'domestica'),  # European plum
        ('Prunus', 'subcordata'), # Klamath plum
        ('Prunus', 'nigra'),      # Canada plum
        ('Prunus', 'angustifolia'),  # Chickasaw plum
    ]
    return _build_bm(taxa, _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_apricot(**kwargs) -> Observations:
    # Prunus armeniaca
    return _build_bm([('Prunus', 'armeniaca')], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_almond(**kwargs) -> Observations:
    # Prunus dulcis / amygdalus (almond)
    taxa: List[Taxon] = [
        ('Prunus', 'dulcis'),
        ('Prunus', 'amygdalus'),
    ]
    return _build_bm(taxa, _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_hazel(**kwargs) -> Observations:
    # Corylus — hazel / filbert.
    return _build_bm(['Corylus'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_quince(**kwargs) -> Observations:
    # Cydonia oblonga
    return _build_bm(['Cydonia'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_walnut(**kwargs) -> Observations:
    # Juglans — walnut.
    return _build_bm(['Juglans'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_hickory(**kwargs) -> Observations:
    # Carya — hickory + pecan.
    return _build_bm(['Carya'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_pawpaw(**kwargs) -> Observations:
    # Asimina — pawpaw.
    return _build_bm(['Asimina'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_mulberry(**kwargs) -> Observations:
    # Morus — mulberry.
    return _build_bm(['Morus'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_persimmon(**kwargs) -> Observations:
    # Diospyros — persimmon.
    return _build_bm(['Diospyros'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_chestnut(**kwargs) -> Observations:
    # Castanea — chestnut.
    return _build_bm(['Castanea'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_grape(**kwargs) -> Observations:
    # Vitis — grape.
    return _build_bm(['Vitis'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_blueberry(**kwargs) -> Observations:
    # Vaccinium — blueberry / cranberry / huckleberry / lingonberry.
    return _build_bm(['Vaccinium'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_currant(**kwargs) -> Observations:
    # Ribes — currant / gooseberry.
    return _build_bm(['Ribes'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


def build_usa_npn_bramble(**kwargs) -> Observations:
    # Rubus — raspberry / blackberry / salmonberry / dewberry.
    return _build_bm(['Rubus'], _DEFAULT_OBS_TYPES, _DEFAULT_TARGET)


# ---------------------------------------------------------------------------
# Composite fruit tree collections (use the registry so individual filters
# are reused without re-running per-genus selection logic).
# ---------------------------------------------------------------------------

def build_usa_npn_fruit_trees(**kwargs) -> Observations:
    """Rosaceae fruit / nut trees + Corylus — PEP725-aligned counterpart."""
    keys = [
        'USA_NPN_Apple',
        'USA_NPN_Pear',
        'USA_NPN_Cherry',
        'USA_NPN_Peach',
        'USA_NPN_Plum',
        'USA_NPN_Apricot',
        'USA_NPN_Almond',
        'USA_NPN_Hazel',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_usa_npn_fruit_trees_extended(**kwargs) -> Observations:
    """``USA_NPN_fruit_trees`` plus North-American deciduous tree fruits / nuts."""
    keys = [
        'USA_NPN_Apple',
        'USA_NPN_Pear',
        'USA_NPN_Cherry',
        'USA_NPN_Peach',
        'USA_NPN_Plum',
        'USA_NPN_Apricot',
        'USA_NPN_Almond',
        'USA_NPN_Hazel',
        'USA_NPN_Quince',
        'USA_NPN_Walnut',
        'USA_NPN_Hickory',
        'USA_NPN_Pawpaw',
        'USA_NPN_Mulberry',
        'USA_NPN_Persimmon',
        'USA_NPN_Chestnut',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_usa_npn_fruit_trees_all(**kwargs) -> Observations:
    """``USA_NPN_fruit_trees_extended`` plus berry shrubs and grape vines."""
    keys = [
        'USA_NPN_Apple',
        'USA_NPN_Pear',
        'USA_NPN_Cherry',
        'USA_NPN_Peach',
        'USA_NPN_Plum',
        'USA_NPN_Apricot',
        'USA_NPN_Almond',
        'USA_NPN_Hazel',
        'USA_NPN_Quince',
        'USA_NPN_Walnut',
        'USA_NPN_Hickory',
        'USA_NPN_Pawpaw',
        'USA_NPN_Mulberry',
        'USA_NPN_Persimmon',
        'USA_NPN_Chestnut',
        'USA_NPN_Grape',
        'USA_NPN_Blueberry',
        'USA_NPN_Currant',
        'USA_NPN_Bramble',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


def build_usa_npn_fruit_trees_rosaceae(**kwargs) -> Observations:
    """Rosaceae genera only — direct PEP725 ``fruit_trees_5`` counterpart."""
    keys = [
        'USA_NPN_Apple',
        'USA_NPN_Pear',
        'USA_NPN_Cherry',
        'USA_NPN_Peach',
        'USA_NPN_Plum',
    ]
    return reduce(Observations.merge, [DATASETS[k]() for k in keys])


# ---------------------------------------------------------------------------
# Calendar configuration (USA-NPN fruit trees are all spring-flowering
# deciduous — the default calendar handles them fine, so no overrides).
# ---------------------------------------------------------------------------

CALENDAR_CONFIGS: dict = {}


# ---------------------------------------------------------------------------
# Registry mapping for USA-NPN datasets
# ---------------------------------------------------------------------------

DATASETS = {
    # Individual fruit / nut / berry groups
    'USA_NPN_Apple':     build_usa_npn_apple,
    'USA_NPN_Pear':      build_usa_npn_pear,
    'USA_NPN_Cherry':    build_usa_npn_cherry,
    'USA_NPN_Peach':     build_usa_npn_peach,
    'USA_NPN_Plum':      build_usa_npn_plum,
    'USA_NPN_Apricot':   build_usa_npn_apricot,
    'USA_NPN_Almond':    build_usa_npn_almond,
    'USA_NPN_Hazel':     build_usa_npn_hazel,
    'USA_NPN_Quince':    build_usa_npn_quince,
    'USA_NPN_Walnut':    build_usa_npn_walnut,
    'USA_NPN_Hickory':   build_usa_npn_hickory,
    'USA_NPN_Pawpaw':    build_usa_npn_pawpaw,
    'USA_NPN_Mulberry':  build_usa_npn_mulberry,
    'USA_NPN_Persimmon': build_usa_npn_persimmon,
    'USA_NPN_Chestnut':  build_usa_npn_chestnut,
    'USA_NPN_Grape':     build_usa_npn_grape,
    'USA_NPN_Blueberry': build_usa_npn_blueberry,
    'USA_NPN_Currant':   build_usa_npn_currant,
    'USA_NPN_Bramble':   build_usa_npn_bramble,

    # Composite collections
    'USA_NPN_fruit_trees':          build_usa_npn_fruit_trees,
    'USA_NPN_fruit_trees_extended': build_usa_npn_fruit_trees_extended,
    'USA_NPN_fruit_trees_all':      build_usa_npn_fruit_trees_all,
    'USA_NPN_fruit_trees_rosaceae': build_usa_npn_fruit_trees_rosaceae,
}
