"""LoRA supervised fine-tuning."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "results/models/qwen25_05b_behavior_lora"
    cache_dir: str | None = None
    max_sequence_length: int = 8192
    seed: int = 42
    n_samples_per_split: int | None = None

    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"

    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def tokenize_sft_batch(
    batch: dict[str, list[Any]], tokenizer: Any, max_sequence_length: int
) -> dict[str, list[Any]]:
    """Tokenize SFT rows and mask every prompt token from the loss."""

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    keep: list[bool] = []
    sequence_length: list[int] = []

    for prompt_text, target_text in zip(
        batch["prompt_text"], batch["target_text"], strict=True
    ):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            target_text + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        ids = prompt_ids + target_ids
        row_labels = [-100] * len(prompt_ids) + target_ids

        input_ids.append(ids)
        labels.append(row_labels)
        attention_mask.append([1] * len(ids))
        sequence_length.append(len(ids))
        keep.append(len(ids) <= max_sequence_length and bool(target_ids))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "sequence_length": sequence_length,
        "keep": keep,
    }


def _filter_summary(dataset_dict: Any, max_sequence_length: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "max_sequence_length": max_sequence_length,
        "splits": {},
    }
    for split_name, dataset in dataset_dict.items():
        total = len(dataset)
        kept = sum(bool(value) for value in dataset["keep"])
        summary["splits"][split_name] = {
            "total_rows": total,
            "kept_rows": kept,
            "dropped_rows": total - kept,
            "dropped_rate": (total - kept) / total if total else 0.0,
        }
    return summary


def run_sft_training(dataset_dict: Any, config: SFTConfig) -> dict[str, Any]:
    """Train and save a LoRA adapter from train/validation/test SFT datasets.
    Effective training batch size: train_batch_size * gradient_accumulation_steps * number_of_devices
    """

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install training dependencies with `uv sync --extra train`."
        ) from error

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset_dict)
    if missing_splits:
        raise KeyError(f"SFT dataset is missing splits: {sorted(missing_splits)}.")

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", asdict(config))

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, use_fast=True, cache_dir=config.cache_dir
    )
    if tokenizer.eos_token is None:
        raise ValueError(f"Tokenizer for {config.base_model} has no EOS token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset_dict.map(
        lambda batch: tokenize_sft_batch(batch, tokenizer, config.max_sequence_length),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing SFT examples",
    )
    summary = _filter_summary(tokenized, config.max_sequence_length)
    _write_json(output_dir / "sequence_length_summary.json", summary)

    # Remove tokens that exceed max_sequence_length
    tokenized = tokenized.filter(
        lambda example: bool(example["keep"]),
        desc="Filtering overlength examples",
    ).remove_columns(["keep", "sequence_length"])

    if not len(tokenized["train"]):
        raise ValueError("No training examples remain after length filtering.")
    if not len(tokenized["validation"]):
        raise ValueError("No validation examples remain after length filtering.")

    dtype = None
    if config.bf16:
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        cache_dir=config.cache_dir,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            module.strip()
            for module in config.lora_target_modules.split(",")
            if module.strip()
        ],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.scheduler,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to=[] if config.report_to == "none" else config.report_to.split(","),
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    if trainer.is_world_process_zero():
        model.print_trainable_parameters()

    trainer.train()
    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized["validation"], metric_key_prefix="validation"
    )
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"], metric_key_prefix="test"
    )
    _add_perplexity(validation_metrics, "validation_loss")
    _add_perplexity(test_metrics, "test_loss")

    if trainer.is_world_process_zero():
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        _write_json(output_dir / "validation_metrics.json", validation_metrics)
        _write_json(output_dir / "test_metrics.json", test_metrics)
        _write_json(output_dir / "log_history.json", trainer.state.log_history)

    return {"validation": validation_metrics, "test": test_metrics}


def _add_perplexity(metrics: dict[str, Any], loss_key: str) -> None:
    """Add perplexity derived from evaluation loss to the metrics dictionary."""

    if loss_key not in metrics:
        return
    try:
        metrics[f"{loss_key.removesuffix('_loss')}_perplexity"] = math.exp(
            metrics[loss_key]
        )
    except OverflowError:
        metrics[f"{loss_key.removesuffix('_loss')}_perplexity"] = float("inf")


def _write_json(path: Path, payload: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
