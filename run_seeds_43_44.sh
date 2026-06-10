  #!/usr/bin/env bash
  set -euo pipefail

  for SEED in 43 44
  do
    SFT="outputs/tofu/sft/seed_$SEED"
    ART="artifacts/tofu/sft_seed_$SEED"

    python -m h2df.sft \
      --config configs/tofu/sft.yaml \
      --set experiment.seed=$SEED \
      --set output.dir=$SFT

    h2df-calibrate \
      --config configs/tofu/h2df_lite.yaml \
      --set experiment.seed=$SEED \
      --set model.base_adapter=$SFT/adapter \
      --set calibration.output_dir=$ART

    for METHOD in ga grad_diff npo
    do
      OUT="outputs/tofu/multiseed/$METHOD/seed_$SEED"

      h2df-train \
        --config configs/tofu/h2df_lite.yaml \
        --set experiment.seed=$SEED \
        --set model.base_adapter=$SFT/adapter \
        --set calibration.output_dir=$ART \
        --set method.name=$METHOD \
        --set method.retain_beta=0.25 \
        --set training.learning_rate=0.00003 \
        --set training.max_steps=200 \
        --set output.save_every=20 \
        --set output.dir=$OUT
    done

    OUT="outputs/tofu/multiseed/h2df/seed_$SEED"

    h2df-train \
      --config configs/tofu/h2df_lite.yaml \
      --set experiment.seed=$SEED \
      --set model.base_adapter=$SFT/adapter \
      --set calibration.output_dir=$ART \
      --set method.name=h2df_lite_retain \
      --set method.retain_every=1 \
      --set method.retain_beta=0.25 \
      --set training.learning_rate=0.00003 \
      --set training.max_steps=200 \
      --set output.save_every=20 \
      --set output.dir=$OUT

    for SPEC in h2df:180 ga:140 grad_diff:160 npo:140
    do
      METHOD=${SPEC%:*}
      STEP=${SPEC#*:}
      SRC="outputs/tofu/multiseed/$METHOD/seed_$SEED/checkpoint-$STEP"
      EVAL="outputs/tofu/full_qa_seed_$SEED/$METHOD"

      mkdir -p "$EVAL"
      cp -aL "$SRC" "$EVAL/adapter"

      h2df-evaluate \
        --config configs/tofu/h2df_lite.yaml \
        --set experiment.seed=$SEED \
        --set model.base_adapter=$SFT/adapter \
        --set calibration.output_dir=$ART \
        --set method.name=$METHOD \
        --set output.dir=$EVAL
    done
  done
