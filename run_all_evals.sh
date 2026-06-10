  #!/usr/bin/env bash
  set -e

  for D in outputs/tofu/sweep/lr*
  do
    for S in 20 40 60 80 100 120 140 160 180 200
    do
      E="$D/eval_$S"
      mkdir -p "$E"
      cp -aL "$D/checkpoint-$S" "$E/adapter"

      h2df-evaluate \
        --config configs/tofu/h2df_lite.yaml \
        --set evaluation.generate_qa=false \
        --set output.dir="$E" \
        --set 'data.retain.split=train[100:200]' \
        --set 'data.calibration.split=train[:100]'
    done
  done
