from __future__ import annotations

from typing import Any

import torch


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_tokenizer(model_config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("tokenizer_name", model_config["name"]),
        revision=model_config.get("tokenizer_revision", model_config.get("revision")),
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_config: dict[str, Any], trainable: bool) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs = {
        "revision": model_config.get("revision"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "torch_dtype": resolve_dtype(model_config.get("dtype", "bfloat16")),
    }
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **kwargs)
    base_adapter = model_config.get("base_adapter")
    if base_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, base_adapter).merge_and_unload()
    model.config.use_cache = not trainable
    if model_config.get("gradient_checkpointing", False) and trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def add_lora(model: Any, lora_config: dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config.get("alpha", 2 * int(lora_config["rank"]))),
        lora_dropout=float(lora_config.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    return get_peft_model(model, config)


def load_adapter_for_evaluation(model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)
