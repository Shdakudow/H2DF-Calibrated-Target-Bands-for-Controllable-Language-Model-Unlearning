#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/tofu/h2df_lite.yaml}"
RUN_DIR="${2:-outputs/tofu/h2df_lite_lr1e5/seed_42}"

for STEP in 10 20 30 40 50 60 70 80 90 100; do
  CHECKPOINT="${RUN_DIR}/checkpoint-${STEP}"
  if [[ ! -f "${CHECKPOINT}/adapter_config.json" ]]; then
    echo "Skipping missing checkpoint: ${CHECKPOINT}"
    continue
  fi

  OUTPUT="outputs/tofu/checkpoint_fast_${STEP}"
  mkdir -p "${OUTPUT}"
  ln -sfn "$(realpath "${CHECKPOINT}")" "${OUTPUT}/adapter"

  h2df-evaluate \
    --config "${CONFIG}" \
    --set evaluation.generate_qa=false \
    --set output.dir="${OUTPUT}" \
    --set "data.retain.split=train[100:200]" \
    --set "data.calibration.split=train[:100]"
done
