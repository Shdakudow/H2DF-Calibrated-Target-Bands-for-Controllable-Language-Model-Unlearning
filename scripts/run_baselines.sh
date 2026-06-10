#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/tofu/h2df_lite.yaml}"
TASK="${2:-tofu}"
SEED="${3:-42}"

# Calibration is shared because all methods use original-model per-example losses.
h2df-calibrate --config "$CONFIG" --set "experiment.seed=$SEED"

for METHOD in ga grad_diff npo simnpo rmu h2df_lite h2df_lite_retain; do
  OUTPUT="outputs/${TASK}/${METHOD}/seed_${SEED}"
  h2df-train --config "$CONFIG" \
    --set "experiment.seed=$SEED" \
    --set "method.name=$METHOD" \
    --set "output.dir=$OUTPUT"
  h2df-evaluate --config "$CONFIG" \
    --set "experiment.seed=$SEED" \
    --set "method.name=$METHOD" \
    --set "output.dir=$OUTPUT"
done
