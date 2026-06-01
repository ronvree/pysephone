"""
Public namespace for pysephone models, with lazy loading.

Models are imported on first attribute access (PEP 562 ``__getattr__``) so that
``import pysephone.models`` — or importing a lightweight model such as
``GDDModel`` — does not pull heavyweight optional dependencies (PyTorch,
XGBoost). The heavy import happens only when you access a model that needs it,
at which point a missing dependency raises a clear "install pysephone[...]"
error.
"""
import importlib
from typing import TYPE_CHECKING

# submodule (under pysephone.models) -> the names it exports
_EXPORTS: dict[str, tuple[str, ...]] = {
    "base": ("BaseModel", "ModelArgs", "ModelException", "NullModel"),
    "mean": ("MeanModel", "MeanModelArgs"),
    "linear_trend": ("LinearTrendModel", "LinearTrendModelArgs"),
    "torch_base": ("BaseTorchModel", "BaseTorchModelArgs"),
    "process_based": ("BasePBModel", "BasePBModelArgs"),
    "gdd": ("GDDModel", "GDDModelArgs", "observation_start", "zero_start"),
    "pvtt": ("CalibratedPVTTModel", "PVTTModel", "PVTTModelArgs"),
    "cf": (
        "BaseCFModel", "BaseCFModelArgs",
        "UtahGDDModel", "UtahGDDModelArgs",
        "ChillingDaysGDDModel", "ChillingDaysGDDModelArgs",
        "DynamicGDDModel", "DynamicGDDModelArgs",
    ),
    "lstm": ("LSTMModel", "LSTMModelArgs"),
    "gru": ("GRUModel", "GRUModelArgs"),
    "cnn_1d": ("CNN1DModel", "CNN1DModelArgs"),
    "transformer": ("TransformerModel", "TransformerModelArgs"),
    "lstm_ctx": (
        "LSTMCtxModel", "LSTMCtxModelArgs",
        "OneHotSpeciesLSTMModel", "PhylogeneticLSTMModel",
    ),
    "random_forest": ("RandomForestModel", "RandomForestModelArgs"),
    "xgb": ("XGBoostModel", "XGBoostModelArgs"),
    "hybrid": ("HybridModel", "HybridModelArgs"),
    "unimodal_hybrid": ("UnimodalHybridModel", "UnimodalHybridModelArgs"),
    "wheat_hybrid": ("WheatHybridModel", "WheatHybridModelArgs"),
    "beta_gdd": (
        "BetaGDDModel", "BetaGDDModelArgs", "GlobalBetaGDDModel",
        "CtxBetaGDDModel", "CtxBetaGDDModelArgs", "OneHotSpeciesBetaGDDModel",
        "PhylogeneticBetaGDDModel", "AlphaEarthBetaGDDModel",
        "PhyloAlphaEarthBetaGDDModel",
    ),
    "bspline_gdd": (
        "BSplineGDDModelArgs", "GlobalBSplineGDDModel", "CtxBSplineGDDModel",
        "CtxBSplineGDDModelArgs", "OneHotSpeciesBSplineGDDModel",
        "OneHotLocationBSplineGDDModel", "PhylogeneticBSplineGDDModel",
        "AlphaEarthBSplineGDDModel",
    ),
}

# name -> submodule
_NAME_TO_SUBMODULE = {
    name: submodule for submodule, names in _EXPORTS.items() for name in names
}

# missing optional-dependency module name -> pip extra that provides it
_EXTRA_FOR_DEP = {
    "torch": "deep",
    "xgboost": "boost",
}

__all__ = sorted(_NAME_TO_SUBMODULE)


def __getattr__(name: str):
    submodule = _NAME_TO_SUBMODULE.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{submodule}")
    except ImportError as exc:
        extra = _EXTRA_FOR_DEP.get(getattr(exc, "name", None))
        if extra is not None:
            raise ImportError(
                f"{name} requires the optional '{exc.name}' dependency. "
                f'Install it with: pip install "pysephone[{extra}]"'
            ) from exc
        raise
    return getattr(module, name)


def __dir__():
    return __all__


if TYPE_CHECKING:  # static re-exports so type checkers / IDEs keep autocomplete
    from pysephone.models.base import BaseModel, ModelArgs, ModelException, NullModel
    from pysephone.models.mean import MeanModel, MeanModelArgs
    from pysephone.models.linear_trend import LinearTrendModel, LinearTrendModelArgs
    from pysephone.models.torch_base import BaseTorchModel, BaseTorchModelArgs
    from pysephone.models.process_based import BasePBModel, BasePBModelArgs
    from pysephone.models.gdd import GDDModel, GDDModelArgs, observation_start, zero_start
    from pysephone.models.pvtt import CalibratedPVTTModel, PVTTModel, PVTTModelArgs
    from pysephone.models.cf import (
        BaseCFModel, BaseCFModelArgs,
        UtahGDDModel, UtahGDDModelArgs,
        ChillingDaysGDDModel, ChillingDaysGDDModelArgs,
        DynamicGDDModel, DynamicGDDModelArgs,
    )
    from pysephone.models.lstm import LSTMModel, LSTMModelArgs
    from pysephone.models.gru import GRUModel, GRUModelArgs
    from pysephone.models.cnn_1d import CNN1DModel, CNN1DModelArgs
    from pysephone.models.transformer import TransformerModel, TransformerModelArgs
    from pysephone.models.lstm_ctx import (
        LSTMCtxModel, LSTMCtxModelArgs, OneHotSpeciesLSTMModel, PhylogeneticLSTMModel,
    )
    from pysephone.models.random_forest import RandomForestModel, RandomForestModelArgs
    from pysephone.models.xgb import XGBoostModel, XGBoostModelArgs
    from pysephone.models.hybrid import HybridModel, HybridModelArgs
    from pysephone.models.unimodal_hybrid import UnimodalHybridModel, UnimodalHybridModelArgs
    from pysephone.models.wheat_hybrid import WheatHybridModel, WheatHybridModelArgs
    from pysephone.models.beta_gdd import (
        BetaGDDModel, BetaGDDModelArgs, GlobalBetaGDDModel, CtxBetaGDDModel,
        CtxBetaGDDModelArgs, OneHotSpeciesBetaGDDModel, PhylogeneticBetaGDDModel,
        AlphaEarthBetaGDDModel, PhyloAlphaEarthBetaGDDModel,
    )
    from pysephone.models.bspline_gdd import (
        BSplineGDDModelArgs, GlobalBSplineGDDModel, CtxBSplineGDDModel,
        CtxBSplineGDDModelArgs, OneHotSpeciesBSplineGDDModel,
        OneHotLocationBSplineGDDModel, PhylogeneticBSplineGDDModel,
        AlphaEarthBSplineGDDModel,
    )
