from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from h2df.data import CausalCollator, TokenizedDataset, load_records, normalize_records
from h2df.losses import per_example_causal_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in ("input_ids", "attention_mask", "labels")
        if key in batch
    }


def make_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    split_name: str,
    targets: list[dict[str, Any]] | None = None,
    source_name: str | None = None,
) -> tuple[list[dict[str, Any]], TokenizedDataset]:
    task = config["experiment"]["task"]
    records = normalize_records(
        load_records(config["data"][source_name or split_name]),
        config["data"],
        task,
    )
    dataset = TokenizedDataset(
        records,
        tokenizer,
        task,
        int(config["training"]["max_length"]),
        targets=targets,
        prompt_template=config["data"].get(
            "prompt_template", "Question: {question}\nAnswer:"
        ),
    )
    return records, dataset


def make_loader(
    dataset: TokenizedDataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def score_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Any,
) -> list[dict[str, float]]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        outputs = model(**model_inputs(batch))
        losses = per_example_causal_loss(outputs.logits.float(), batch["labels"])
        indices = batch["index"]
        gathered_indices = accelerator.gather_for_metrics(indices).detach().cpu().tolist()
        gathered_losses = accelerator.gather_for_metrics(losses).detach().cpu().tolist()
        if accelerator.is_main_process:
            rows.extend(
                {"index": int(index), "loss": float(loss)}
                for index, loss in zip(gathered_indices, gathered_losses)
            )
    if accelerator.is_main_process:
        unique = {row["index"]: row for row in rows}
        return [unique[index] for index in sorted(unique)]
    return []


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def perplexity(nll: float) -> float:
    return float(math.exp(min(nll, 20.0)))
