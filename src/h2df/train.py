from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_scheduler

from h2df.config import build_parser, load_config, save_resolved_config
from h2df.data import read_jsonl
from h2df.losses import (
    ga_loss,
    h2df_loss,
    lora_l2,
    masked_representation_mse,
    npo_loss,
    per_example_causal_loss,
    simnpo_loss,
)
from h2df.modeling import add_lora, load_model, load_tokenizer
from h2df.runtime import (
    append_jsonl,
    infinite,
    make_dataset,
    make_loader,
    model_inputs,
    set_seed,
)


SUPPORTED_METHODS = {
    "ga",
    "grad_diff",
    "npo",
    "simnpo",
    "rmu",
    "h2df_lite",
    "h2df_lite_retain",
}


def forget_objective(
    method: str,
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    method_config: dict[str, Any],
) -> torch.Tensor:
    if method in {"ga", "grad_diff"}:
        return ga_loss(scores)
    if method == "npo":
        return npo_loss(scores, batch["original_loss"], float(method_config.get("beta", 0.1)))
    if method == "simnpo":
        return simnpo_loss(
            scores,
            beta=float(method_config.get("beta", 2.5)),
            gamma=float(method_config.get("simnpo_gamma", 0.0)),
        )
    if method in {"h2df_lite", "h2df_lite_retain"}:
        return h2df_loss(
            scores,
            batch["lower_target"],
            tau=float(method_config.get("tau", 0.25)),
            upper_targets=batch.get("upper_target"),
            gamma=float(method_config.get("gamma", 0.0)),
        )
    raise ValueError(f"Unsupported method: {method}")


def resolve_hidden_layer(hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    try:
        return hidden_states[layer]
    except IndexError as exc:
        raise ValueError(
            f"RMU layer {layer} is invalid for {len(hidden_states) - 1} transformer layers"
        ) from exc


def grad_norm(model: torch.nn.Module) -> float:
    squared = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().item())


def run(config: dict[str, Any]) -> None:
    training = config["training"]
    method_config = config["method"]
    method = method_config["name"]
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method.name must be one of {sorted(SUPPORTED_METHODS)}")

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
    targets = read_jsonl(Path(config["calibration"]["output_dir"]) / "targets.jsonl")
    _, forget_dataset = make_dataset(config, tokenizer, "forget", targets)
    forget_loader = make_loader(
        forget_dataset,
        tokenizer,
        int(training["batch_size"]),
        True,
        int(training.get("num_workers", 0)),
    )

    needs_retain = method in {"grad_diff", "rmu", "h2df_lite_retain"}
    retain_loader = None
    if needs_retain:
        _, retain_dataset = make_dataset(config, tokenizer, "retain")
        retain_loader = make_loader(
            retain_dataset,
            tokenizer,
            int(training.get("retain_batch_size", training["batch_size"])),
            True,
            int(training.get("num_workers", 0)),
        )

    model = add_lora(load_model(config["model"], trainable=True), config["lora"])
    reference_model = None
    if method == "rmu":
        reference_model = load_model(config["model"], trainable=False)
        reference_model.requires_grad_(False)
        reference_model.eval()
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

    if retain_loader is None:
        model, optimizer, forget_loader, scheduler = accelerator.prepare(
            model, optimizer, forget_loader, scheduler
        )
    elif reference_model is not None:
        (
            model,
            reference_model,
            optimizer,
            forget_loader,
            retain_loader,
            scheduler,
        ) = accelerator.prepare(
            model,
            reference_model,
            optimizer,
            forget_loader,
            retain_loader,
            scheduler,
        )
    else:
        model, optimizer, forget_loader, retain_loader, scheduler = accelerator.prepare(
            model, optimizer, forget_loader, retain_loader, scheduler
        )
    forget_iterator = infinite(forget_loader)
    retain_iterator = infinite(retain_loader) if retain_loader is not None else None
    metrics_path = output_dir / "train_metrics.jsonl"
    if accelerator.is_main_process and metrics_path.exists():
        metrics_path.unlink()

    model.train()
    control_vector = None
    if method == "rmu":
        hidden_size = int(model.config.hidden_size)
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(
            int(config["experiment"].get("seed", 42))
            + int(method_config.get("control_seed_offset", 10_000))
        )
        control_vector = torch.rand(
            hidden_size,
            generator=generator,
            device=accelerator.device,
            dtype=torch.float32,
        )
        control_vector = (
            control_vector / control_vector.norm().clamp_min(1e-12)
            * float(method_config.get("steering_coefficient", 20.0))
        )
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    while step < max_steps:
        batch = next(forget_iterator)
        with accelerator.accumulate(model):
            outputs = model(
                **model_inputs(batch),
                output_hidden_states=method == "rmu",
            )
            scores = per_example_causal_loss(outputs.logits.float(), batch["labels"])
            if method == "rmu":
                assert control_vector is not None
                layer = int(method_config.get("rmu_layer", -1))
                forget_hidden = resolve_hidden_layer(outputs.hidden_states, layer)
                target = control_vector.view(1, 1, -1).expand_as(forget_hidden)
                objective = masked_representation_mse(
                    forget_hidden, target, batch["attention_mask"]
                )
            else:
                objective = forget_objective(method, scores, batch, method_config)
            regularizer = float(method_config.get("lora_lambda", 0.0)) * lora_l2(model)
            loss = objective + regularizer

            retain_value = None
            replay = method in {"grad_diff", "rmu"} or (
                method == "h2df_lite_retain"
                and (step + 1) % int(method_config.get("retain_every", 10)) == 0
            )
            if replay:
                assert retain_iterator is not None
                retain_batch = next(retain_iterator)
                retain_outputs = model(
                    **model_inputs(retain_batch),
                    output_hidden_states=method == "rmu",
                )
                if method == "rmu":
                    assert reference_model is not None
                    with torch.no_grad():
                        reference_outputs = reference_model(
                            **model_inputs(retain_batch),
                            output_hidden_states=True,
                        )
                    layer = int(method_config.get("rmu_layer", -1))
                    retain_loss = masked_representation_mse(
                        resolve_hidden_layer(retain_outputs.hidden_states, layer),
                        resolve_hidden_layer(reference_outputs.hidden_states, layer),
                        retain_batch["attention_mask"],
                    )
                else:
                    retain_loss = per_example_causal_loss(
                        retain_outputs.logits.float(), retain_batch["labels"]
                    ).mean()
                retain_value = retain_loss.detach()
                loss = loss + float(method_config.get("retain_beta", 1.0)) * retain_loss

            accelerator.backward(loss)
            clip = training.get("max_grad_norm")
            raw_grad_norm = 0.0
            if accelerator.sync_gradients:
                accelerator.unscale_gradients(optimizer)
                raw_grad_norm = grad_norm(model)
                if clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        step += 1
        if step % int(config["output"].get("log_every", 10)) == 0 or step == 1:
            elapsed = time.perf_counter() - started
            below = (scores.detach() < batch["lower_target"]).float().mean()
            above = (
                (scores.detach() > batch["upper_target"]).float().mean()
                if "upper_target" in batch
                else torch.zeros_like(below)
            )
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "forget_nll": float(scores.detach().mean().item()),
                "retain_nll": float(retain_value.item()) if retain_value is not None else None,
                "fraction_below_target": float(below.item()),
                "fraction_above_target": float(above.item()),
                "lora_l2": float(lora_l2(model).detach().item()),
                "gradient_norm": raw_grad_norm,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                "steps_per_second": step / elapsed,
            }
            if accelerator.is_main_process:
                append_jsonl(row, metrics_path)
                print(
                    f"step={step} loss={row['loss']:.4f} "
                    f"forget_nll={row['forget_nll']:.4f} "
                    f"below={row['fraction_below_target']:.3f}"
                )

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
    parser = build_parser("Run an H2DF or baseline unlearning experiment")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
