from __future__ import annotations

from pathlib import Path

from accelerate import Accelerator

from h2df.calibration import build_targets
from h2df.config import build_parser, dump_json, load_config, save_resolved_config
from h2df.data import write_jsonl
from h2df.modeling import load_model, load_tokenizer
from h2df.runtime import make_dataset, make_loader, score_loader, set_seed


def run(config: dict) -> None:
    accelerator = Accelerator()
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    output_dir = Path(config["calibration"]["output_dir"])

    tokenizer = load_tokenizer(config["model"])
    forget_records, forget_dataset = make_dataset(config, tokenizer, "forget")
    calibration_records, calibration_dataset = make_dataset(
        config, tokenizer, "calibration"
    )
    batch_size = int(config["calibration"].get("batch_size", config["training"]["batch_size"]))
    forget_loader = make_loader(forget_dataset, tokenizer, batch_size, False)
    calibration_loader = make_loader(calibration_dataset, tokenizer, batch_size, False)

    model = load_model(config["model"], trainable=False)
    model, forget_loader, calibration_loader = accelerator.prepare(
        model, forget_loader, calibration_loader
    )
    forget_scores = score_loader(model, forget_loader, accelerator)
    calibration_scores = score_loader(model, calibration_loader, accelerator)

    if not accelerator.is_main_process:
        return

    forget_rows = []
    for record, example, score in zip(
        forget_records, forget_dataset.examples, forget_scores
    ):
        forget_rows.append(
            {
                **record,
                "token_length": example["token_length"],
                "original_loss": score["loss"],
            }
        )
    calibration_rows = []
    for record, example, score in zip(
        calibration_records, calibration_dataset.examples, calibration_scores
    ):
        calibration_rows.append(
            {
                **record,
                "token_length": example["token_length"],
                "original_loss": score["loss"],
            }
        )

    targets, metadata = build_targets(
        forget_rows,
        calibration_rows,
        config["experiment"]["task"],
        config["calibration"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(targets, output_dir / "targets.jsonl")
    write_jsonl(calibration_rows, output_dir / "calibration_scores.jsonl")
    dump_json(metadata, output_dir / "calibration_metadata.json")
    save_resolved_config(config, output_dir)
    print(f"Wrote {len(targets)} targets to {output_dir / 'targets.jsonl'}")


def main() -> None:
    parser = build_parser("Calibrate H2DF target margins or loss bands")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
