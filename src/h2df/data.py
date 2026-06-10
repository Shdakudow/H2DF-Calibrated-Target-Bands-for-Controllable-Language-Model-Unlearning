from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def load_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Load one configured split from local JSON/JSONL or Hugging Face datasets."""
    path = source["path"]
    local_path = Path(path)
    if local_path.exists():
        if local_path.suffix == ".jsonl":
            with local_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        elif local_path.suffix == ".json":
            with local_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            split = source.get("split")
            records = payload[split] if split and isinstance(payload, dict) else payload
        else:
            raise ValueError(f"Local data must be .json or .jsonl: {local_path}")
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of records in {local_path}")
        return records

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the project dependencies to load Hugging Face data") from exc

    dataset = load_dataset(
        path,
        source.get("name"),
        split=source.get("split", "train"),
        revision=source.get("revision"),
    )
    return [dict(row) for row in dataset]


def question_type(question: str) -> str:
    stripped = question.strip().lower()
    if not stripped:
        return "empty"
    first = stripped.split(maxsplit=1)[0].strip("\"'([{")
    return first if first in {"who", "what", "when", "where", "why", "how", "which"} else "other"


def normalize_record(record: dict[str, Any], data_config: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "tofu":
        question = str(record[data_config["question_column"]])
        answer = str(record[data_config["answer_column"]])
        return {
            "question": question,
            "answer": answer,
            "question_type": question_type(question),
            "domain": str(record.get(data_config.get("domain_column", ""), "default")),
        }
    if task == "muse":
        text = str(record[data_config["text_column"]])
        return {
            "text": text,
            "domain": str(record.get(data_config.get("domain_column", ""), "default")),
        }
    raise ValueError(f"Unsupported task: {task}")


def normalize_records(
    records: Iterable[dict[str, Any]], data_config: dict[str, Any], task: str
) -> list[dict[str, Any]]:
    return [normalize_record(record, data_config, task) for record in records]


def _trim_pair(
    prompt_ids: list[int], answer_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    if len(answer_ids) >= max_length:
        return [], answer_ids[:max_length]
    prompt_budget = max_length - len(answer_ids)
    return prompt_ids[-prompt_budget:], answer_ids


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
    task: str,
    max_length: int,
    prompt_template: str = "Question: {question}\nAnswer:",
) -> dict[str, Any]:
    if task == "tofu":
        prompt = prompt_template.format(question=record["question"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode(record["answer"], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
        prompt_ids, answer_ids = _trim_pair(prompt_ids, answer_ids, max_length)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
    else:
        input_ids = tokenizer.encode(
            record["text"], add_special_tokens=True, truncation=True, max_length=max_length
        )
        labels = list(input_ids)

    if len(input_ids) < 2:
        raise ValueError("Each tokenized example must contain at least two tokens")
    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "token_length": len(input_ids),
    }
    result.update({key: value for key, value in record.items() if key not in {"text"}})
    return result


class TokenizedDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        task: str,
        max_length: int,
        targets: list[dict[str, Any]] | None = None,
        prompt_template: str = "Question: {question}\nAnswer:",
    ) -> None:
        self.examples = [
            tokenize_record(record, tokenizer, task, max_length, prompt_template)
            for record in records
        ]
        for index, example in enumerate(self.examples):
            example["index"] = index
        if targets is not None:
            if len(targets) != len(self.examples):
                raise ValueError("Target count does not match forget-example count")
            for index, target in enumerate(targets):
                if int(target["index"]) != index:
                    raise ValueError("Targets must be ordered by contiguous example index")
                self.examples[index].update(target)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(example["input_ids"]) for example in examples)
        batch: dict[str, Any] = {}
        for key, pad_value in (
            ("input_ids", self.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            values = [
                example[key] + [pad_value] * (max_length - len(example[key]))
                for example in examples
            ]
            batch[key] = torch.tensor(values, dtype=torch.long)

        numeric_keys = {"index", "original_loss", "lower_target", "upper_target"}
        for key in numeric_keys:
            if key in examples[0]:
                dtype = torch.long if key == "index" else torch.float32
                batch[key] = torch.tensor([example[key] for example in examples], dtype=dtype)
        return batch


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
