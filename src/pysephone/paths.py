import os
import re

from pathlib import Path

from platformdirs import user_data_dir


ENV_DATA_ROOT = "PYSEPHONE_DATA_ROOT"

# Google Earth Engine project (GCP project with the Earth Engine API enabled).
# Used to initialize Earth Engine when fetching AlphaEarth embeddings. No project
# is shipped with the library — users supply their own via the ee_project argument
# or one of these environment variables.
ENV_EE_PROJECT = "PYSEPHONE_EE_PROJECT"
# Earth Engine's own native env var, honored as a secondary fallback.
ENV_EE_PROJECT_NATIVE = "EARTHENGINE_PROJECT"

"""
    Roots
"""

def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_root() -> Path:
    """
    Resolve the root directory under which all data, caches, and outputs live.

    Order:
      1. the PYSEPHONE_DATA_ROOT environment variable, if set
      2. an OS-native per-user data directory (e.g. %LOCALAPPDATA%\\pysephone on
         Windows, ~/.local/share/pysephone on Linux/macOS)

    Note: this intentionally does NOT default to the repo/package directory — an
    installed package lives in a read-only site-packages location. To keep data
    inside a source checkout during development, set PYSEPHONE_DATA_ROOT=<repo>.
    """
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).expanduser()
    return Path(user_data_dir("pysephone", appauthor=False))

def get_ee_project(explicit: str | None = None) -> str | None:
    """
    Resolve the Earth Engine GCP project to use.

    Resolution order (all optional — returns None if nothing is set, in which
    case Earth Engine is left to resolve its own default project):
      1. the explicit argument (e.g. an ee_project= passed by the caller)
      2. the PYSEPHONE_EE_PROJECT environment variable
      3. Earth Engine's native EARTHENGINE_PROJECT environment variable
    """
    if explicit:
        return explicit
    return (
        os.environ.get(ENV_EE_PROJECT)
        or os.environ.get(ENV_EE_PROJECT_NATIVE)
        or None
    )

"""
    Paths of main folder structure (relative to root)

    DATA_ROOT/
    ├── data/
    │   ├── observations/              # Store phenology observations
    │   │   └── <source_name>
    │   └── products/                  # Store other data
    │       └── <product_name>
    ├── datasets/                      # Store processed datasets
    │   └── <dataset_name>/
    ├── models/                        # Store models
    │   └── <model_id>/
    └── runs/                          # Store runs
        └── <run_id>/

"""

def get_data_dir(root: Path) -> Path:
    return root / "data"

def get_observation_data_dir(root: Path) -> Path:
    return get_data_dir(root) / "observations"

def get_products_data_dir(root: Path) -> Path:
    return get_data_dir(root) / "products"

def get_runs_dir(root: Path) -> Path:
    return root / "runs"

def get_models_dir(root: Path) -> Path:
    return root / "outputs" / "models"

def get_evaluations_dir(root: Path) -> Path:
    return root / "outputs" / "evaluations"

def get_comparisons_dir(root: Path) -> Path:
    return root / "outputs" / "comparisons"

def get_datasets_dir(root: Path) -> Path:
    return root / "datasets"

"""
    Paths to individual items
"""

def get_observations_source_data_dir(root: Path, source_name: str) -> Path:
    assert is_valid_id(source_name)
    return get_observation_data_dir(root) / source_name

def get_dataset_dir(root: Path, dataset_name: str) -> Path:
    assert is_valid_id(dataset_name)
    return get_datasets_dir(root) / dataset_name

def get_run_dir(root: Path, run_id: str) -> Path:
    assert is_valid_id(run_id)
    return get_runs_dir(root) / run_id

def get_model_dir(root: Path, model_id: str) -> Path:
    assert is_valid_id(model_id)
    return get_models_dir(root) / model_id


"""
    Utility functions
"""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

def is_valid_id(name: str) -> bool:
    """
    Validate a folder-safe identifier.

    Rules:
    - non-empty
    - starts with a letter or digit
    - contains only letters, digits, '.', '_', '-'
    - no spaces, slashes, or special characters
    """
    return bool(_NAME_RE.fullmatch(name))
