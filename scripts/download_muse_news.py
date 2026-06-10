from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


DATASET = "muse-bench/MUSE-News"
OUTPUT_DIR = Path("data/muse")
RAW_SPLITS = ("forget", "retain1", "retain2", "holdout")


def write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text, "domain": "news"}) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, list[str]] = {}
    dropped_empty: dict[str, list[int]] = {}
    for split in RAW_SPLITS:
        dataset = load_dataset(DATASET, "raw", split=split)
        if list(dataset.features) != ["text"]:
            raise ValueError(f"Unexpected fields for {split}: {list(dataset.features)}")
        source_texts = [str(text) for text in dataset["text"]]
        dropped_empty[split] = [
            index for index, text in enumerate(source_texts) if not text.strip()
        ]
        texts = [text for text in source_texts if text.strip()]
        if not texts:
            raise ValueError(f"Split {split} has no valid text")
        loaded[split] = texts
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", texts)

    holdout = loaded["holdout"]
    calibration = holdout[::2]
    evaluation = holdout[1::2]
    if not calibration or not evaluation:
        raise ValueError("Holdout split is too small for disjoint calibration/evaluation")
    write_jsonl(OUTPUT_DIR / "holdout_calibration.jsonl", calibration)
    write_jsonl(OUTPUT_DIR / "holdout_evaluation.jsonl", evaluation)

    metadata = {
        "dataset": DATASET,
        "config": "raw",
        "counts": {name: len(texts) for name, texts in loaded.items()},
        "dropped_empty_source_indices": dropped_empty,
        "holdout_partition": {
            "method": "alternating source order",
            "calibration_examples": len(calibration),
            "evaluation_examples": len(evaluation),
            "overlap": 0,
        },
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
