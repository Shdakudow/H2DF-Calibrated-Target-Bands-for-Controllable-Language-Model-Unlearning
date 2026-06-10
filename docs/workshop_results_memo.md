# H2DF Workshop Results Memo

## Scope

- Benchmarks: TOFU and MUSE-News
- MUSE target model: `muse-bench/MUSE-News_target`
- MUSE tokenizer: `NousResearch/Llama-2-7b-hf`
- Parameterization: LoRA rank 8 for MUSE
- Main MUSE budget: 50 optimizer steps
- Checkpoint selection: fixed 200-example validation screen, selecting the
  checkpoint closest to 50% below the calibrated lower target

## MUSE-News Seed 42

Full held-out evaluation:

| Method | Forget NLL | Retain NLL | Loss MIA AUC | Inside band |
|---|---:|---:|---:|---:|
| Original | 0.576 | 0.761 | 0.981 | 0.015 |
| H2DF, step 40 | 1.557 | 1.666 | 0.949 | 0.131 |
| GA, step 40 | 1.865 | 2.018 | 0.934 | 0.071 |
| NPO, step 40 | 1.858 | 2.010 | 0.934 | 0.072 |
| GradDiff, step 50 | 1.379 | 1.546 | 0.961 | 0.081 |
| SimNPO, step 50 | 1.513 | 1.597 | 0.955 | 0.144 |

Interpretation:

- H2DF preserves retention better than GA and NPO, but forgets less.
- H2DF and SimNPO are close. SimNPO has slightly better retention and
  in-band rate on the internal loss metrics.
- GradDiff is less aggressive and retains more utility.

## Corrected H2DF Multiseed Screen

All seeds use the same 50-step schedule and checkpoint set.

| Seed | Selected step | Forget NLL | Retain NLL | Below target | Inside band |
|---|---:|---:|---:|---:|---:|
| 42 | 40 | 1.530 | 1.598 | 0.575 | 0.125 |
| 43 | 50 | 0.956 | 1.038 | 0.930 | 0.030 |
| 44 | 50 | 1.531 | 1.568 | 0.705 | 0.105 |

H2DF remains seed-sensitive after correcting the scheduler mismatch. Seed 43
does not reach the intended forgetting region within 50 steps.

## Official MUSE-News Metrics

Lower is better for `VerbMem Forget` and `KnowMem Forget`; higher is better
for `KnowMem Retain`.

| Model | VerbMem Forget | KnowMem Forget | KnowMem Retain | PrivLeak |
|---|---:|---:|---:|---:|
| Original | 57.42 | 64.09 | 54.12 | -99.81 |
| H2DF | 20.71 | 45.83 | 38.72 | -96.52 |
| SimNPO | 19.27 | 45.74 | 38.27 | -98.32 |

H2DF and SimNPO are effectively tied on knowledge forgetting. SimNPO has
slightly stronger verbatim forgetting, while H2DF retains slightly more
knowledge utility.

The official repository's current main branch required two runtime fixes:

- PrivLeak metric keys must be read from the first result record.
- The retrain AUC normalization table must be selected by corpus.

Metric formulas and datasets were otherwise unchanged.

## Ablations

Fixed 200-example screen:

| Variant | Step | Forget NLL | Retain NLL | Below | Inside | Above |
|---|---:|---:|---:|---:|---:|---:|
| Two-sided band | 40 | 1.530 | 1.598 | 0.575 | 0.125 | 0.300 |
| Two-sided fixed final | 50 | 1.800 | 1.852 | 0.280 | 0.160 | 0.560 |
| Lower-bound only | 40 | 1.711 | 1.790 | 0.435 | 0.095 | 0.470 |
| Lower-bound only | 50 | 2.107 | 2.171 | 0.125 | 0.080 | 0.795 |
| Two-sided + retain 0.1 | 50 | 1.107 | 1.195 | 0.905 | 0.020 | 0.075 |

Conclusions:

- The upper band reduces overshoot and retention damage.
- Retain replay with weight 0.1 suppresses forgetting too strongly.
- Validation-based checkpoint selection avoids the substantial overshoot of
  the fixed final checkpoint.

## Defensible Workshop Claims

1. Calibrated two-sided loss bands provide explicit control over forgetting
   strength and reduce overshoot relative to a lower-bound-only objective.
2. H2DF achieves a competitive forgetting-retention tradeoff on TOFU and
   MUSE-News, but does not dominate SimNPO.
3. Checkpoint selection and seed sensitivity are central practical issues for
   short-budget LoRA unlearning.
4. Loss-distribution metrics correlate with, but do not replace, the official
   MUSE evaluation suite.

## Limitations

- H2DF is not robust across all three MUSE seeds at a fixed 50-step budget.
- SimNPO is at least as competitive on MUSE-News.
- Official MUSE metrics were run for seed 42 only.
- RMU is not implementation-equivalent to the original full-parameter setup.
- MUSE-Books and sequential/scalability evaluations are not included.
- TOFU and MUSE use different base-model and evaluation regimes.

## Recommended Workshop Framing

Present H2DF as a calibrated target-band formulation and an empirical study of
controllable forgetting, overshoot, and checkpoint selection. Avoid a
state-of-the-art superiority claim.
