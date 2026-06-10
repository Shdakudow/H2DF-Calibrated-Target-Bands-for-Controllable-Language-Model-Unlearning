# Curated Results

This directory contains lightweight evaluation metrics used by the H2DF workshop
manuscript. It intentionally excludes model weights, adapters, predictions, datasets,
and caches.

## Layout

- `tofu/full_qa/`: selected seed-42 full QA evaluations.
- `tofu/multiseed/`: available full QA evaluations for seeds 43 and 44.
- `muse/seed42/`: full held-out loss diagnostics for the selected methods.
- `muse/official/`: official MUSE-News metrics for Original, H2DF, and SimNPO.
- `muse/multiseed/`: fixed 200-example screens for corrected H2DF seeds 43 and 44.
- `muse/ablations/`: lower-bound and retain-replay screen evaluations.

The manuscript tables and interpretation are summarized in
`docs/workshop_results_memo.md`. The raw experiment tree remains under the ignored
local `outputs/` directory.

## Important qualifications

- Official MUSE metrics were evaluated only at seed 42.
- H2DF seed 43 did not reach the intended target region in the 50-step budget.
- The MUSE multiseed and ablation JSONs are checkpoint-screen metrics on a fixed
  200-example subset, not full held-out evaluations.
- The RMU implementation is a LoRA-matched comparison rather than the original
  full-parameter method.
