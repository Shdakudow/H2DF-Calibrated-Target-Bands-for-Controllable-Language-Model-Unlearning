from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score

from h2df.config import build_parser, dump_json, load_config, should_generate_qa
from h2df.data import read_jsonl, write_jsonl
from h2df.modeling import load_adapter_for_evaluation, load_model, load_tokenizer
from h2df.runtime import make_dataset, make_loader, mean, perplexity, score_loader, set_seed


def normalize_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"[^\w\s]", "", text)


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def qa_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    template = config["data"].get("prompt_template", "Question: {question}\nAnswer:")
    max_new_tokens = int(config["evaluation"].get("max_new_tokens", 64))
    max_examples = config["evaluation"].get("qa_max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]
    rows = []
    model.eval()
    for record in records:
        prompt = template.format(question=record["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        exact_match = normalize_answer(prediction) == normalize_answer(record["answer"])
        rows.append(
            {
                "question": record["question"],
                "reference": record["answer"],
                "prediction": prediction,
                "exact_match": exact_match,
                "token_f1": token_f1(prediction, record["answer"]),
            }
        )
    metrics: dict[str, float | int] = {
        "qa_examples": len(rows),
        "qa_exact_match": mean([float(row["exact_match"]) for row in rows]),
        "qa_token_f1": mean([float(row["token_f1"]) for row in rows]),
    }
    return metrics, rows


def split_metrics(scores: list[dict[str, float]]) -> dict[str, float]:
    losses = [row["loss"] for row in scores]
    nll = mean(losses)
    return {"nll": nll, "ppl": perplexity(nll), "examples": len(losses)}


def run(config: dict[str, Any]) -> None:
    accelerator = Accelerator()
    set_seed(int(config["experiment"].get("seed", 42)))
    tokenizer = load_tokenizer(config["model"])
    model = load_model(config["model"], trainable=False)
    method = config["method"]["name"]
    if method != "original":
        model = load_adapter_for_evaluation(model, str(Path(config["output"]["dir"]) / "adapter"))

    targets = read_jsonl(Path(config["calibration"]["output_dir"]) / "targets.jsonl")
    datasets = {}
    records = {}
    for split in ("forget", "retain", "calibration"):
        split_targets = targets if split == "forget" else None
        source_name = split
        if split == "retain" and "retain_eval" in config["data"]:
            source_name = "retain_eval"
        elif split == "calibration" and "calibration_eval" in config["data"]:
            source_name = "calibration_eval"
        records[split], datasets[split] = make_dataset(
            config, tokenizer, split, split_targets, source_name=source_name
        )
    loaders = {
        split: make_loader(
            dataset,
            tokenizer,
            int(config["evaluation"].get("batch_size", config["training"]["batch_size"])),
            False,
        )
        for split, dataset in datasets.items()
    }
    model, loaders["forget"], loaders["retain"], loaders["calibration"] = accelerator.prepare(
        model, loaders["forget"], loaders["retain"], loaders["calibration"]
    )
    scores = {
        split: score_loader(model, loader, accelerator)
        for split, loader in loaders.items()
    }
    if not accelerator.is_main_process:
        return

    result: dict[str, Any] = {
        "method": method,
        "task": config["experiment"]["task"],
        "forget": split_metrics(scores["forget"]),
        "retain": split_metrics(scores["retain"]),
        "calibration": split_metrics(scores["calibration"]),
    }
    forget_losses = np.asarray([row["loss"] for row in scores["forget"]])
    lower = np.asarray([row["lower_target"] for row in targets])
    result["target_diagnostics"] = {
        "fraction_below_target": float(np.mean(forget_losses < lower)),
    }
    if "upper_target" in targets[0]:
        upper = np.asarray([row["upper_target"] for row in targets])
        result["target_diagnostics"].update(
            {
                "fraction_above_target": float(np.mean(forget_losses > upper)),
                "fraction_inside_band": float(
                    np.mean((forget_losses >= lower) & (forget_losses <= upper))
                ),
            }
        )

    task = config["experiment"]["task"]
    if should_generate_qa(config):
        unwrapped = accelerator.unwrap_model(model)
        forget_metrics, forget_predictions = qa_metrics(
            unwrapped, tokenizer, records["forget"], config, accelerator.device
        )
        retain_metrics, retain_predictions = qa_metrics(
            unwrapped, tokenizer, records["retain"], config, accelerator.device
        )
        result["forget"].update(forget_metrics)
        result["retain"].update(retain_metrics)
        output_dir = Path(config["output"]["dir"])
        write_jsonl(forget_predictions, output_dir / "forget_predictions.jsonl")
        write_jsonl(retain_predictions, output_dir / "retain_predictions.jsonl")
    elif task == "tofu":
        result["qa_generation_skipped"] = True
    else:
        calibration_losses = np.asarray([row["loss"] for row in scores["calibration"]])
        labels = np.concatenate(
            [np.ones(len(forget_losses)), np.zeros(len(calibration_losses))]
        )
        attack_scores = -np.concatenate([forget_losses, calibration_losses])
        auc = float(roc_auc_score(labels, attack_scores))
        result["distribution"] = {
            "forget_calibration_ks": float(
                ks_2samp(forget_losses, calibration_losses).statistic
            ),
            "forget_calibration_wasserstein": float(
                wasserstein_distance(forget_losses, calibration_losses)
            ),
            "loss_membership_auc": auc,
            "loss_membership_advantage": abs(2.0 * auc - 1.0),
        }

    output_path = Path(config["output"]["dir"]) / "evaluation.json"
    dump_json(result, output_path)
    print(f"Wrote evaluation to {output_path}")


def main() -> None:
    parser = build_parser("Evaluate an H2DF or baseline checkpoint")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
