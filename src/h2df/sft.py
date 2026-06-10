from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import ConcatDataset
from transformers import get_scheduler

from h2df.config import build_parser, load_config, save_resolved_config
from h2df.losses import per_example_causal_loss
from h2df.modeling import add_lora, load_model, load_tokenizer
from h2df.runtime import (
    append_jsonl,
    infinite,
    make_dataset,
    make_loader,
    model_inputs,
    set_seed,
)


def run(config: dict[str, Any]) -> None:
    training = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=training.get("mixed_precision"),
    )
    set_seed(int(config["experiment"].get("seed", 42)))
    output_dir = Path(config["output"]["dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir)

    tokenizer = load_tokenizer(config["model"])
    _, forget_dataset = make_dataset(config, tokenizer, "forget")
    _, retain_dataset = make_dataset(config, tokenizer, "retain")
    train_dataset = ConcatDataset([forget_dataset, retain_dataset])
    train_loader = make_loader(
        train_dataset,
        tokenizer,
        int(training["batch_size"]),
        True,
        int(training.get("num_workers", 0)),
    )

    model = add_lora(load_model(config["model"], trainable=True), config["lora"])
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    max_steps = int(training["max_steps"])
    scheduler = get_scheduler(
        training.get("scheduler", "cosine"),
        optimizer,
        num_warmup_steps=int(training.get("warmup_steps", 0)),
        num_training_steps=max_steps,
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    metrics_path = output_dir / "train_metrics.jsonl"
    if accelerator.is_main_process and metrics_path.exists():
        metrics_path.unlink()

    iterator = infinite(train_loader)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    step = 0
    while step < max_steps:
        batch = next(iterator)
        with accelerator.accumulate(model):
            outputs = model(**model_inputs(batch))
            loss = per_example_causal_loss(outputs.logits.float(), batch["labels"]).mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients and training.get("max_grad_norm") is not None:
                accelerator.clip_grad_norm_(
                    model.parameters(), float(training["max_grad_norm"])
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        step += 1
        if step % int(config["output"].get("log_every", 10)) == 0 or step == 1:
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                "steps_per_second": step / elapsed,
            }
            if accelerator.is_main_process:
                append_jsonl(row, metrics_path)
                print(f"step={step} loss={row['loss']:.4f}")

        save_every = int(config["output"].get("save_every", 0))
        if save_every and step % save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(
                    output_dir / f"checkpoint-{step}"
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")


def main() -> None:
    parser = build_parser("Supervised fine-tune a TOFU base model")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
