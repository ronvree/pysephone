"""pysephone — machine-learning models for predicting plant phenology.

The lightweight core (datasets, observations, season calendars, the feature-
provider interface) is exported here. Models live under ``pysephone.models``
and are loaded lazily so that importing the package does not pull heavyweight
optional dependencies such as PyTorch.
"""
from pysephone.dataset import Dataset, Observations
from pysephone.dataset.util.calendar import Calendar
from pysephone.dataset.util.provider import FeatureProvider

__all__ = ["Dataset", "Observations", "Calendar", "FeatureProvider"]
