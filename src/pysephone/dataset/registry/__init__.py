"""
Dataset registry.

Maps dataset names to builder callables that each return an Observations object.
Add a new module here and merge its DATASETS dict into REGISTRY to register
additional dataset families.

``CALENDAR_CONFIGS`` maps dataset names to callables that configure a
:class:`~pysephone.dataset.util.calendar.Calendar` with sensible season
windows for the species in that dataset.  :meth:`Dataset.load` calls the
matching config automatically when a Calendar is provided.
"""

from functools import reduce

from pysephone.dataset.registry import gmu_cherry, pep725

# Merge all per-family registries into a single lookup table.
# Import order determines priority when names clash (last writer wins).
REGISTRY = {
    **pep725.DATASETS,
    **gmu_cherry.DATASETS,
    # 'all_fruit_trees' needs both families so it is wired up after both are imported
}

CALENDAR_CONFIGS = {
    **pep725.CALENDAR_CONFIGS,
    **gmu_cherry.CALENDAR_CONFIGS,
}

# Wire up cross-family composites now that both registries are available.
# We import these lazily inside the builder so we need to patch the cross-
# references in gmu_cherry after pep725 is available.
gmu_cherry.DATASETS['PEP725_fruit_trees'] = pep725.DATASETS['PEP725_fruit_trees']
REGISTRY['all_fruit_trees'] = gmu_cherry.build_all_fruit_trees
