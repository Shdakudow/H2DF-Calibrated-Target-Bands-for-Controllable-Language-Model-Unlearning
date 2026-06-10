# H2DF Experiments

Reference implementation and curated results for the experiments reported in
[`main.tex`](main.tex):

- TOFU H2DF-Lite with calibrated, one-sided answer-loss margins.
- MUSE H2DF-Lite with matched non-member, two-sided LM-loss bands.
- H2DF-Lite+R with sparse retain replay.
- Comparable Original, GA, GradDiff, NPO, SimNPO, and RMU runs.
- Per-step diagnostics and loss-distribution evaluation.

The code uses Qwen2.5-1.5B-Instruct with LoRA by default. Dataset fields are
configured rather than hard-coded because TOFU/MUSE mirrors and local exports use
different schemas.

## Repository layout

```text
configs/              Reproducible TOFU and MUSE YAML configurations
scripts/              Shell entry points for complete experiment runs
src/h2df/             Calibration, training, evaluation, and objective code
tests/                Unit tests that do not require model downloads
results/              Curated lightweight metrics used in the paper
docs/                 Result memo and experiment interpretation
main.tex              Workshop manuscript
references.bib        Bibliography for the manuscript
```

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Authenticate with Hugging Face if the selected model or dataset requires it:

```bash
huggingface-cli login
```

## Data configuration

Each split can be either a Hugging Face dataset or a local `.json`/`.jsonl` file.
Edit the `data` block in a config:

```yaml
data:
  forget:
    path: locuslab/TOFU
    name: forget05
    split: train
  retain:
    path: locuslab/TOFU
    name: retain95
    split: "train[10%:]"
  calibration:
    path: locuslab/TOFU
    name: retain95
    split: "train[:10%]"
  question_column: question
  answer_column: answer
```

For MUSE, set `text_column` and optionally `domain_column`. Calibration data must
be held-out non-member text from the same domain as the forget data.

## Run an experiment

The first command scores the original model and creates fixed per-example targets.
The second performs LoRA unlearning. The third reports forget/retain/calibration
metrics.

```bash
h2df-calibrate --config configs/tofu/h2df_lite.yaml
h2df-train --config configs/tofu/h2df_lite.yaml
h2df-evaluate --config configs/tofu/h2df_lite.yaml
```

Or run all stages:

```bash
bash scripts/run_tofu.sh
bash scripts/run_muse.sh
```

Override YAML values from the command line with dotted keys:

```bash
h2df-train --config configs/tofu/h2df_lite.yaml \
  --set method.name=ga training.max_steps=200 output.dir=outputs/tofu_ga
```

## Implemented objectives

`h2df_lite`
: The paper's smooth one-sided TOFU objective or two-sided MUSE objective.

`h2df_lite_retain`
: H2DF-Lite plus retain cross-entropy every `retain_every` steps.

`ga`
: Unbounded gradient ascent on per-example forget loss.

`grad_diff`
: GA plus a retain cross-entropy term every step.

`npo`
: Negative preference optimization using original-model per-example losses
recorded during calibration.

`simnpo`
: Reference-free negative preference optimization. The implementation follows the
official length-normalized form:
`-2 / beta * log(sigmoid(beta * (NLL - gamma)))`.

`rmu`
: Representation Misdirection for Unlearning. Forget activations at `rmu_layer` are
matched to a seeded random unit vector scaled by `steering_coefficient`; retain
activations are matched to a frozen original model. For an equal-parameter comparison,
this repository applies RMU through the same LoRA modules used by all other methods.
The original WMDP implementation instead selects full transformer parameters.

Dedicated examples are available in `configs/tofu/simnpo.yaml` and
`configs/tofu/rmu.yaml`. RMU layer, steering coefficient, and retain coefficient
must be tuned; they are not architecture-independent constants.

## Outputs

Calibration writes `targets.jsonl` and `calibration_metadata.json`. Training writes
LoRA adapter checkpoints, `train_metrics.jsonl`, and the resolved config. Evaluation
writes `evaluation.json` with NLL/PPL, target-band diagnostics, QA exact match when
applicable, and MUSE loss-distribution/privacy proxies.

For paper results, run at least three seeds and report both equal-step and
equal-wall-clock comparisons. Do not treat loss-only membership AUC as a replacement
for the official MUSE evaluation suite.

## Published results

The compact JSON metrics used for the workshop manuscript are tracked in `results/`.
They include selected TOFU evaluations, the MUSE-News seed-42 baseline comparison,
official MUSE metrics, corrected H2DF multiseed screens, and objective ablations.
See `results/README.md` and `docs/workshop_results_memo.md`.

Raw `outputs/`, model adapters, calibration artifacts, downloaded datasets, and the
local virtual environment are intentionally excluded from Git. Together they occupy
about 33 GB and include files that exceed GitHub's normal file-size limit. Reproduce
them with the checked-in configs and scripts rather than committing generated weights.

Current MUSE conclusion: H2DF-Lite improves the forgetting-retention balance relative
to GA and NPO and its upper target boundary reduces overshoot. It is competitive with,
but does not dominate, SimNPO and remains sensitive to random seed under a fixed
50-step budget.

## Model artifacts

This repository does not include trained LoRA adapters, full model checkpoints, or
downloaded base-model weights. These files are excluded because they are large and
may be subject to the licenses or access conditions of their upstream models and
datasets.

The checked-in configurations and scripts reproduce the reported adapters from the
corresponding upstream base models. If pretrained H2DF adapters are released
separately, their permanent Hugging Face or archival URL should be listed here and
in the paper's artifact-availability statement.

## Paper

The manuscript source is [`main.tex`](main.tex), with citations in
[`references.bib`](references.bib). For Overleaf, upload both files and set
`main.tex` as the main document. A venue-specific class or style file must also
be uploaded if the manuscript is converted to an official conference template.

## Verification

Run the CPU-only unit tests with:

```bash
pytest -q
```

The current repository test suite contains 14 tests.

## Citation and license

Citation metadata is provided in `CITATION.cff`. Code is released under the MIT
License. Dataset, model, and benchmark files retain their original licenses and are
not redistributed by this repository.

## RTX 4090 runtime

For Qwen2.5-1.5B, BF16, LoRA rank 8, batch size 4, sequence length up to 512, and
500 optimizer steps on one RTX 4090:

| Stage | GA / SimNPO / H2DF-Lite | H2DF-Lite+R | RMU |
|---|---:|---:|---:|
| Calibration | 1-4 min | 1-4 min | 1-4 min |
| Training | 4-10 min | 5-12 min | 12-30 min |
| NLL/PPL evaluation | 2-8 min | 2-8 min | 2-8 min |
| TOFU greedy QA generation | 15-45 min | 15-45 min | 15-45 min |

These are planning ranges, not measured benchmarks. Sequence-length distribution,
FlashAttention availability, thermal/power limits, and the number of evaluation
examples can move them substantially. The i9-14900K is sufficient; these runs are
primarily GPU-bound after tokenization. RMU is expected to use roughly 12-20 GB VRAM
at the default settings because it holds both trainable and frozen 1.5B models. Reduce
both batch sizes to 2 if it exceeds 24 GB.

A complete seven-method comparison is approximately 3-7 hours for one seed and
9-21 hours for three seeds, mainly because TOFU answer generation is repeated for
every checkpoint. A broad paper-level hyperparameter sweep can take 2-6 GPU-days.
Run short 50-step pilots first, prune unstable settings, and evaluate full QA
generation only for selected checkpoints.
