#!/usr/bin/env bash
  set -e

  run_eval () {
    CONFIG=$1
    METHOD=$2
    SOURCE=$3
    OUTPUT=$4

    mkdir -p "$OUTPUT"
    cp -aL "$SOURCE" "$OUTPUT/adapter"

    h2df-evaluate \
      --config "$CONFIG" \
      --set method.name="$METHOD" \
      --set output.dir="$OUTPUT"
  }

  run_eval configs/tofu/sft.yaml sft \
    outputs/tofu/sft/seed_42/adapter \
    outputs/tofu/full_qa/sft

  run_eval configs/tofu/h2df_lite.yaml h2df_lite_retain \
    outputs/tofu/sweep/lr0.00003_b0.25/checkpoint-180 \
    outputs/tofu/full_qa/h2df

  run_eval configs/tofu/h2df_lite.yaml ga \
    outputs/tofu/baselines/ga/seed_42/checkpoint-140 \
    outputs/tofu/full_qa/ga

  run_eval configs/tofu/h2df_lite.yaml grad_diff \
    outputs/tofu/baselines/grad_diff/seed_42/checkpoint-160 \
    outputs/tofu/full_qa/grad_diff

  run_eval configs/tofu/h2df_lite.yaml npo \
    outputs/tofu/baselines/npo/seed_42/checkpoint-140 \
    outputs/tofu/full_qa/npo

  run_eval configs/tofu/h2df_lite.yaml simnpo \
    outputs/tofu/baselines/simnpo/seed_42/checkpoint-140 \
    outputs/tofu/full_qa/simnpo
