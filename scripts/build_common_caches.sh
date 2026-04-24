#!/usr/bin/env bash
# Build OpenMeteo feature caches for all datasets commonly used in notebooks.
#
# Run from the repository root:
#   bash scripts/build_common_caches.sh
#
# Pass --force to rebuild caches that already exist:
#   bash scripts/build_common_caches.sh --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BUILD="$PYTHON $SCRIPT_DIR/build_feature_cache.py"

# Forward any extra arguments (e.g. --force) to each build call
EXTRA_ARGS=("$@")

# ── PEP725 fruit trees (used in unusual_year_model_eval + species_lstm_comparison) ──
PEP725_DATASETS=(
    PEP725_Apple
    PEP725_Pear
    PEP725_Peach
    PEP725_Cherry
    PEP725_Plum
    PEP725_Almond
    PEP725_Apricot
)

# ── GMU Cherry (used in unusual_year_model_eval + cherry blossom notebooks) ──
GMU_DATASETS=(
    GMU_Cherry_Japan_Y
    GMU_Cherry_Japan_YS
    GMU_Cherry_Switzerland
    GMU_Cherry_South_Korea
)

# ── PEP725 composite (used in cf_models, dataset_adequacy, unusual_seasons) ──
COMPOSITE_DATASETS=(
    PEP725_fruit_trees
)

ALL_DATASETS=("${PEP725_DATASETS[@]}" "${GMU_DATASETS[@]}" "${COMPOSITE_DATASETS[@]}")

echo "Building temperature caches for ${#ALL_DATASETS[@]} datasets ..."
echo

for ds in "${ALL_DATASETS[@]}"; do
    echo "══════════════════════════════════════════════════════"
    echo "  $ds"
    echo "══════════════════════════════════════════════════════"
    $BUILD "$ds" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo
done

# ── species_lstm_comparison uses temperature + daylight_duration ──────────────
echo "Building temperature+daylight caches for species LSTM comparison ..."
echo
LSTM_DATASETS=(
    PEP725_fruit_trees
    PEP725_Apple
    PEP725_Pear
    PEP725_Peach
    PEP725_Almond
    PEP725_Cherry
    PEP725_Apricot
    PEP725_Plum
)
for ds in "${LSTM_DATASETS[@]}"; do
    echo "══════════════════════════════════════════════════════"
    echo "  $ds  [temperature + daylight]"
    echo "══════════════════════════════════════════════════════"
    $BUILD "$ds" --keys temperature_2m_mean daylight_duration "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo
done

echo "All caches built."
