import os
import re

from pathlib import Path


ENV_DATA_ROOT = "PYSEPHONE_DATA_ROOT"

"""
    Roots
"""

def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_root() -> Path:
    return Path(
        os.environ.get(ENV_DATA_ROOT, get_repo_root())
    ).expanduser()

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
