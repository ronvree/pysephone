#!/usr/bin/env bash
set -euo pipefail

${PYTHON:-python3} -m pysephone.run.fit_eval \
    --run_name pvtt_pep725_winter_wheat \
    --target BBCH_51 \
    --dataset_name CPF_PEP725_winter_wheat \
    --split_years cutoff --split_years_cutoff_year 2010 \
    --model_cls_path pysephone.models.pvtt.PVTTModel \
    --threshold_pvtt 800.0 \
    --threshold_vern 30.0 \
    --t_base 1.0 \
    --t_limit 32.0 \
    --t_upper 40.0 \
    --p_base 7.0 \
    --p_saturation 17.0 \
    --key_sow BBCH_0 \
    --verbose
